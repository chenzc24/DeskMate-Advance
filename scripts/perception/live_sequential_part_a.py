"""Run the Laptop-only sequential identity and multimodal action loop."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import queue
import time

import cv2

from poker_dealer.domain import (
    ActionEvidenceState,
    DealerCommand,
    DealerCommandType,
    PlayerActionType,
    SEAT_ORDER,
    Seat,
)
from poker_dealer.game import (
    ActionPromoter,
    HandEngine,
    PromotionPolicy,
    SimulatedDealer,
)
from poker_dealer.io.camera import CameraConfig, CameraError, CameraReadStatus, OpenCVCamera
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GesturePilotConfig,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    MultimodalActionWindow,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    VoskSpeechRecognizer,
    observation_to_dict,
)
from poker_dealer.perception.identity import (
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
)
from poker_dealer.runtime import PartAPhase, SequentialPartACoordinator

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - clear CLI diagnostic
    sd = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SEAT_KEYS = {ord("1"): Seat.A, ord("2"): Seat.B, ord("3"): Seat.C, ord("4"): Seat.D}
PLAYER_BY_SEAT = {seat: f"player_{seat.value[-1]}" for seat in SEAT_ORDER}


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-config",
        type=Path,
        default=ROOT / "configs/perception/face_identity_session.json",
    )
    parser.add_argument(
        "--gesture-config",
        type=Path,
        default=ROOT / "configs/perception/actions_laptop_pilot.json",
    )
    parser.add_argument(
        "--speech-config",
        type=Path,
        default=ROOT / "configs/perception/actions_speech_pilot.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument("--speech-device", type=_device)
    parser.add_argument("--disable-speech", action="store_true")
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument(
        "--player-mode",
        choices=("four_player_core", "two_player_pilot"),
        default="four_player_core",
    )
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rotation-delay-ms", type=int, default=350)
    parser.add_argument("--identity-grace-ms", type=int, default=1000)
    parser.add_argument("--session-id", default="sequential-part-a-pilot")
    parser.add_argument("--hand-id", default="sequential-part-a-hand")
    parser.add_argument("--button", choices=tuple(seat.value for seat in Seat), default=Seat.A.value)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def _clear_queue(audio_queue: queue.Queue[bytes]) -> int:
    cleared = 0
    while True:
        try:
            audio_queue.get_nowait()
            cleared += 1
        except queue.Empty:
            return cleared


def _emit(event_type: str, **payload: object) -> None:
    print(
        json.dumps({"type": event_type, **payload}, ensure_ascii=True),
        flush=True,
    )


def _two_player_pilot_start(enrolled_seats: set[Seat]) -> tuple[Seat, Seat]:
    """Choose the longest enrolled clockwise run and a Button before it."""

    if not enrolled_seats:
        raise ValueError("two-player pilot requires at least one enrolled seat")
    best_start = next(seat for seat in SEAT_ORDER if seat in enrolled_seats)
    best_length = 0
    for start in SEAT_ORDER:
        if start not in enrolled_seats:
            continue
        index = SEAT_ORDER.index(start)
        length = 0
        for offset in range(len(SEAT_ORDER)):
            seat = SEAT_ORDER[(index + offset) % len(SEAT_ORDER)]
            if seat not in enrolled_seats:
                break
            length += 1
        if length > best_length:
            best_start, best_length = start, length
    button = SEAT_ORDER[(SEAT_ORDER.index(best_start) + 1) % len(SEAT_ORDER)]
    return button, best_start


def _registered_player(
    gallery: SessionFaceGallery, seat: Seat | None
) -> str | None:
    if seat is None:
        return None
    for item in gallery.metadata():
        if item["seat"] == seat.value:
            return str(item["player_id"])
    return None


def _resolve_start_plan(
    player_mode: str,
    enrolled_seats: set[Seat],
    configured_button: Seat,
) -> tuple[Seat, Seat | None, str | None]:
    if player_mode == "four_player_core":
        missing = set(SEAT_ORDER) - enrolled_seats
        if missing or len(enrolled_seats) != 4:
            return (
                configured_button,
                None,
                "four_player_core requires A/B/C/D; missing "
                + ",".join(sorted(seat.value for seat in missing)),
            )
        return configured_button, None, None
    if player_mode != "two_player_pilot":
        raise ValueError("unsupported player mode")
    if len(enrolled_seats) != 2:
        return (
            configured_button,
            None,
            "two_player_pilot requires exactly two registered seats",
        )
    button, expected_first = _two_player_pilot_start(enrolled_seats)
    next_after_first = SEAT_ORDER[
        (SEAT_ORDER.index(expected_first) + 1) % len(SEAT_ORDER)
    ]
    if next_after_first not in enrolled_seats:
        return (
            button,
            expected_first,
            "two_player_pilot requires adjacent clockwise seats",
        )
    return button, expected_first, None


def main() -> int:
    args = parse_args()
    if (
        args.max_seconds <= 0
        or args.rotation_delay_ms < 0
        or args.identity_grace_ms < 0
    ):
        raise SystemExit("runtime duration must be positive and rotation delay non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if not args.disable_speech and sd is None:
        print(json.dumps({"type": "error", "error": "sounddevice is unavailable"}))
        return 2

    identity_config = FaceIdentityConfig.from_json(args.identity_config)
    gesture_config = GesturePilotConfig.from_json(args.gesture_config)
    speech_config = SpeechPilotConfig.from_json(args.speech_config)
    camera_values = identity_config.camera
    camera_config = CameraConfig(
        device_index=int(camera_values["device_index"]) if args.index is None else args.index,
        source_id="sequential_part_a_pilot",
        backend=str(camera_values["backend"]) if args.backend is None else args.backend,
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fps=float(camera_values["fps"]),
    )

    audio_queue: queue.Queue[bytes] = queue.Queue(
        maxsize=int(speech_config.audio["queue_max_blocks"])
    )
    dropped_audio_blocks = 0

    def audio_callback(
        indata: bytes, _frames: int, _time_info: object, status: object
    ) -> None:
        nonlocal dropped_audio_blocks
        if status:
            dropped_audio_blocks += 1
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            dropped_audio_blocks += 1

    stream_context = nullcontext()
    microphone_summary: dict[str, object] = {"enabled": False}
    if not args.disable_speech:
        assert sd is not None
        selected = sd.query_devices(args.speech_device, "input")
        microphone_summary = {
            "enabled": True,
            "requested": args.speech_device,
            "name": str(selected["name"]),
            "sample_rate_hz": int(speech_config.audio["sample_rate_hz"]),
        }
        stream_context = sd.RawInputStream(
            samplerate=int(speech_config.audio["sample_rate_hz"]),
            blocksize=int(speech_config.audio["blocksize_frames"]),
            device=args.speech_device,
            dtype=str(speech_config.audio["dtype"]),
            channels=1,
            callback=audio_callback,
        )

    dealer = SimulatedDealer()
    now_ns = time.monotonic_ns()
    home = DealerCommand("part-a-sim-home", now_ns, DealerCommandType.HOME)
    dealer.execute(home, now_ns + 1)
    coordinator: SequentialPartACoordinator | None = None
    pending_rotation_due_ns: int | None = None
    registration_target = Seat.A
    enrollment_active = False
    enrollment_samples = []
    last_enrollment_sample_ns: int | None = None
    identity_temporal = FaceIdentityTemporalAdapter(identity_config)
    gesture_temporal = GestureTemporalAdapter(gesture_config)
    multimodal = MultimodalActionWindow(
        decision_wait_ms=500,
        max_skew_ms=3000,
        allow_speech_single_source=args.player_mode != "four_player_core",
    )
    speech_recognizer: VoskSpeechRecognizer | None = None
    speech_adapter = SpeechObservationAdapter(speech_config)
    frames = 0
    missing_reads = 0
    accepted_actions = 0
    rejected_actions = 0
    identity_matches = 0
    simulated_rotation_acks = 0
    status_text = "SETUP: choose 1-4, E enroll, S start"
    last_gesture = "no hand"
    last_identity_log_key: tuple[object, ...] | None = None
    last_gesture_log_key: tuple[object, ...] | None = None
    identity_guard_last_valid_ns: int | None = None
    speech_confirm_requested = False
    two_player_remaining: set[Seat] = set()
    started_ns = time.monotonic_ns()

    try:
        if not args.disable_speech:
            speech_recognizer = VoskSpeechRecognizer(speech_config)
        with SessionFaceGallery(identity_config, args.session_id) as gallery:
            with OpenCVCamera(camera_config) as camera, MediaPipeGestureAdapter(
                gesture_config
            ) as gesture_model, stream_context:
                face_model = OpenCvFaceIdentityAdapter(identity_config)
                camera_summary = camera.negotiated_properties()
                print(
                    json.dumps(
                        {
                            "type": "ready",
                            "runtime": "sequential_part_a_vertical_loop",
                            "player_mode": args.player_mode,
                            "camera": camera_summary,
                            "microphone": microphone_summary,
                            "rotation_adapter": "simulated_dealer_only",
                            "consent_confirmed": args.consent_confirmed,
                            "frames_saved": 0,
                            "audio_saved": False,
                            "embeddings_persisted": False,
                            "four_player_speech_authority": (
                                "gesture_agreement_or_ui_confirmation_required"
                            ),
                        },
                        ensure_ascii=True,
                    )
                )

                while (time.monotonic_ns() - started_ns) / 1_000_000_000 < args.max_seconds:
                    if args.max_frames is not None and frames >= args.max_frames:
                        break
                    read = camera.read()
                    if read.status is not CameraReadStatus.OK or read.frame is None:
                        missing_reads += 1
                        if read.status is CameraReadStatus.DISCONNECTED:
                            status_text = "camera disconnected"
                            break
                        continue
                    frames += 1
                    frame = read.frame
                    now_ns = time.monotonic_ns()
                    face_evidence = None
                    gesture_evidence = None

                    if (
                        coordinator is not None
                        and coordinator.phase is PartAPhase.WAITING_ROTATION_ACK
                        and coordinator.pending_rotation is not None
                        and pending_rotation_due_ns is not None
                        and now_ns >= pending_rotation_due_ns
                    ):
                        pending_command = coordinator.pending_rotation
                        ack = dealer.execute(pending_command, now_ns)
                        simulated_rotation_acks += 1
                        accepted_ack = coordinator.accept_rotation_ack(ack)
                        _emit(
                            "rotation_ack",
                            command_id=ack.command_id,
                            target_slot=(
                                ack.target_slot.value if ack.target_slot else None
                            ),
                            status=ack.status.value,
                            accepted=accepted_ack,
                            state_version=coordinator.engine.state.state_version,
                        )
                        pending_rotation_due_ns = None
                        identity_temporal = FaceIdentityTemporalAdapter(identity_config)
                        status_text = coordinator.last_reason

                    phase = coordinator.phase if coordinator is not None else None
                    if phase in {
                        None,
                        PartAPhase.VERIFYING_IDENTITY,
                        PartAPhase.WAITING_PLAYER_ACTION,
                    }:
                        face_evidence = face_model.analyze(frame)

                        target = (
                            registration_target
                            if coordinator is None
                            else coordinator.focus_seat
                        )
                        assert target is not None
                        if enrollment_active:
                            can_sample = (
                                face_evidence.detected_face_count == 1
                                and len(face_evidence.features) == 1
                                and (
                                    last_enrollment_sample_ns is None
                                    or face_evidence.observed_at_ns - last_enrollment_sample_ns
                                    >= 150_000_000
                                )
                            )
                            if can_sample:
                                enrollment_samples.append(face_evidence.features[0])
                                last_enrollment_sample_ns = face_evidence.observed_at_ns
                            status_text = (
                                f"ENROLL {PLAYER_BY_SEAT[target]} {len(enrollment_samples)}/"
                                f"{identity_config.minimum_samples}"
                            )
                            if len(enrollment_samples) >= identity_config.minimum_samples:
                                try:
                                    gallery.enroll(
                                        PLAYER_BY_SEAT[target],
                                        target,
                                        enrollment_samples,
                                        consent_granted=args.consent_confirmed,
                                    )
                                    status_text = f"enrolled {PLAYER_BY_SEAT[target]} at {target.value}"
                                    _emit(
                                        "enrollment_completed",
                                        player_id=PLAYER_BY_SEAT[target],
                                        seat=target.value,
                                        sample_count=len(enrollment_samples),
                                        gallery_size=gallery.size,
                                    )
                                except (PermissionError, ValueError) as exc:
                                    status_text = f"enrollment rejected: {exc}"
                                    _emit(
                                        "enrollment_rejected",
                                        player_id=PLAYER_BY_SEAT[target],
                                        seat=target.value,
                                        reason=str(exc),
                                    )
                                enrollment_active = False
                                enrollment_samples = []
                                last_enrollment_sample_ns = None
                                identity_temporal = FaceIdentityTemporalAdapter(identity_config)

                        if coordinator is not None and coordinator.phase is PartAPhase.VERIFYING_IDENTITY:
                            focus_seat = coordinator.focus_seat
                            assert focus_seat is not None
                            match = gallery.match_expected_seat(
                                face_evidence, focus_seat
                            )
                            identity_observation = identity_temporal.process(
                                match,
                                face_evidence.observed_at_ns,
                                FaceIdentityContext(
                                    args.session_id,
                                    coordinator.engine.state.state_version,
                                    coordinator.focus_seat,  # type: ignore[arg-type]
                                ),
                            )
                            opened = coordinator.accept_identity(identity_observation)
                            identity_log_key = (
                                identity_observation.identity_state,
                                identity_observation.player_id,
                                identity_observation.registered_seat,
                                coordinator.engine.state.state_version,
                                coordinator.focus_seat,
                            )
                            if args.emit_all or identity_log_key != last_identity_log_key:
                                _emit(
                                    "identity_observation",
                                    state=identity_observation.identity_state.value,
                                    player_id=identity_observation.player_id,
                                    registered_seat=(
                                        identity_observation.registered_seat.value
                                        if identity_observation.registered_seat
                                        else None
                                    ),
                                    focus_seat=focus_seat.value,
                                    state_version=coordinator.engine.state.state_version,
                                    similarity=identity_observation.similarity,
                                    stable_frames=identity_observation.stable_frames,
                                    stable_duration_ms=identity_observation.stable_duration_ms,
                                    quality_flags=list(identity_observation.quality_flags),
                                )
                                last_identity_log_key = identity_log_key
                            expected_player = _registered_player(
                                gallery, coordinator.focus_seat
                            ) or PLAYER_BY_SEAT[coordinator.focus_seat]  # type: ignore[index]
                            if opened:
                                status_text = coordinator.last_reason
                            elif identity_observation.identity_state is FaceIdentityState.SEAT_MISMATCH:
                                status_text = (
                                    f"MISMATCH saw {identity_observation.player_id} at "
                                    f"{identity_observation.registered_seat.value if identity_observation.registered_seat else '?'}; "
                                    f"expected {expected_player}"
                                )
                            elif identity_observation.identity_state is FaceIdentityState.ENROLLMENT_REQUIRED:
                                status_text = f"ENROLLMENT REQUIRED: expected {expected_player}"
                            elif (
                                identity_observation.identity_state
                                is FaceIdentityState.EXPECTED_SEAT_UNENROLLED
                            ):
                                status_text = (
                                    f"EXPECTED SEAT UNENROLLED: {coordinator.focus_seat.value}"
                                )
                            else:
                                status_text = (
                                    f"{identity_observation.identity_state.value}: "
                                    f"expected {expected_player}"
                                )
                            if opened:
                                identity_matches += 1
                                identity_guard_last_valid_ns = (
                                    identity_observation.observed_at_ns
                                )
                                gesture_temporal = GestureTemporalAdapter(gesture_config)
                                speech_adapter = SpeechObservationAdapter(speech_config)
                                multimodal.clear()
                                _clear_queue(audio_queue)
                                if speech_recognizer is not None:
                                    speech_recognizer.reset_window()
                                print(
                                    json.dumps(
                                        {
                                            "type": "identity_gate_opened",
                                            "player_id": coordinator.verified_player_id,
                                            "focus_seat": coordinator.focus_seat.value,  # type: ignore[union-attr]
                                            "state_version": coordinator.engine.state.state_version,
                                        },
                                        ensure_ascii=True,
                                    )
                                )

                    face_guard_current = False
                    if (
                        coordinator is not None
                        and coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION
                        and face_evidence is not None
                    ):
                        focus_seat = coordinator.focus_seat
                        assert focus_seat is not None
                        guard_match = gallery.match_expected_seat(
                            face_evidence, focus_seat
                        )
                        face_guard_current = (
                            guard_match.state is FaceIdentityState.MATCHED
                            and guard_match.player_id == coordinator.verified_player_id
                            and guard_match.registered_seat is focus_seat
                        )
                        if face_guard_current:
                            identity_guard_last_valid_ns = face_evidence.observed_at_ns
                        hard_identity_failure = (
                            guard_match.state is FaceIdentityState.MULTIPLE_FACES
                            or (
                                guard_match.state is FaceIdentityState.MATCHED
                                and not face_guard_current
                            )
                        )
                        grace_expired = (
                            identity_guard_last_valid_ns is None
                            or face_evidence.observed_at_ns
                            - identity_guard_last_valid_ns
                            > args.identity_grace_ms * 1_000_000
                        )
                        if hard_identity_failure or grace_expired:
                            reason = (
                                "different_or_multiple_player_detected"
                                if hard_identity_failure
                                else "face_missing_or_unknown_beyond_grace"
                            )
                            coordinator.revoke_identity(reason)
                            _emit(
                                "action_window_closed",
                                reason=reason,
                                focus_seat=focus_seat.value,
                                state_version=coordinator.engine.state.state_version,
                            )
                            identity_temporal = FaceIdentityTemporalAdapter(
                                identity_config
                            )
                            identity_guard_last_valid_ns = None
                            gesture_temporal = GestureTemporalAdapter(gesture_config)
                            speech_adapter = SpeechObservationAdapter(speech_config)
                            multimodal.clear()
                            _clear_queue(audio_queue)
                            if speech_recognizer is not None:
                                speech_recognizer.reset_window()
                            status_text = coordinator.last_reason

                    if coordinator is not None and coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION:
                        context = ActionObservationContext(
                            coordinator.engine.state.hand_id,
                            coordinator.engine.state.state_version,
                            coordinator.focus_seat,  # type: ignore[arg-type]
                        )
                        gesture_evidence = gesture_model.recognize(frame)
                        last_gesture = (
                            f"{gesture_evidence.gesture_label or 'None'} "
                            f"{gesture_evidence.gesture_score or 0.0:.2f}"
                        )
                        gesture_observation = gesture_temporal.process(
                            gesture_evidence, context
                        )
                        gesture_log_key = (
                            gesture_observation.evidence_state,
                            gesture_observation.candidate_action,
                            gesture_evidence.gesture_label,
                            coordinator.engine.state.state_version,
                        )
                        if args.emit_all or gesture_log_key != last_gesture_log_key:
                            _emit(
                                "gesture_observation",
                                label=gesture_evidence.gesture_label,
                                score=gesture_evidence.gesture_score,
                                evidence_state=gesture_observation.evidence_state.value,
                                candidate_action=(
                                    gesture_observation.candidate_action.value
                                    if gesture_observation.candidate_action
                                    else None
                                ),
                                face_guard_current=face_guard_current,
                                focus_seat=context.focus_seat.value,
                                state_version=context.expected_state_version,
                            )
                            last_gesture_log_key = gesture_log_key
                        fused = None
                        if face_guard_current and speech_confirm_requested:
                            fused = multimodal.confirm_pending_speech(now_ns)
                            speech_confirm_requested = False
                            if fused is not None:
                                _emit(
                                    "speech_ui_confirmation",
                                    candidate_action=(
                                        fused.candidate_action.value
                                        if fused.candidate_action
                                        else None
                                    ),
                                    focus_seat=context.focus_seat.value,
                                    state_version=context.expected_state_version,
                                )
                        if face_guard_current and fused is None:
                            fused = multimodal.add(gesture_observation)
                        while speech_recognizer is not None and face_guard_current:
                            try:
                                pcm = audio_queue.get_nowait()
                            except queue.Empty:
                                break
                            speech_evidence = speech_recognizer.accept_audio(
                                pcm, time.monotonic_ns()
                            )
                            if speech_evidence is None:
                                continue
                            speech_observation = speech_adapter.process(
                                speech_evidence, context
                            )
                            _emit(
                                "speech_observation",
                                transcript=speech_evidence.canonical_transcript,
                                confidence=speech_evidence.confidence,
                                evidence_state=speech_observation.evidence_state.value,
                                candidate_action=(
                                    speech_observation.candidate_action.value
                                    if speech_observation.candidate_action
                                    else None
                                ),
                                focus_seat=context.focus_seat.value,
                                state_version=context.expected_state_version,
                                quality_flags=list(speech_observation.quality_flags),
                            )
                            speech_fused = multimodal.add(speech_observation)
                            if speech_fused is not None:
                                fused = speech_fused
                        if not face_guard_current:
                            _clear_queue(audio_queue)
                        if fused is None and face_guard_current:
                            fused = multimodal.poll(now_ns)
                        if (
                            args.player_mode == "four_player_core"
                            and multimodal.pending_sources == ("speech",)
                        ):
                            status_text = (
                                "speech candidate pending: matching gesture or C confirmation required"
                            )
                        if fused is not None:
                            before_version = coordinator.engine.state.state_version
                            acted_seat = fused.focus_seat
                            outcome = coordinator.accept_action(fused)
                            print(
                                json.dumps(
                                    {
                                        "type": "multimodal_action_decision",
                                        **observation_to_dict(fused),
                                        "accepted": outcome.accepted,
                                        "reason": outcome.reason,
                                        "next_seat": outcome.next_seat.value if outcome.next_seat else None,
                                    },
                                    ensure_ascii=True,
                                )
                            )
                            if outcome.accepted:
                                accepted_actions += 1
                                status_text = coordinator.last_reason
                                if args.player_mode == "two_player_pilot":
                                    two_player_remaining.discard(acted_seat)
                                    if not two_player_remaining:
                                        coordinator.complete_pilot(
                                            "two_registered_players_completed_one_action"
                                        )
                                        status_text = coordinator.last_reason
                                _emit(
                                    "state_transition",
                                    accepted=True,
                                    action=(
                                        fused.candidate_action.value
                                        if fused.candidate_action
                                        else None
                                    ),
                                    acting_seat=acted_seat.value,
                                    before_version=before_version,
                                    after_version=coordinator.engine.state.state_version,
                                    next_seat=(
                                        coordinator.focus_seat.value
                                        if coordinator.focus_seat
                                        else None
                                    ),
                                    game_phase=coordinator.engine.state.phase.value,
                                    runtime_phase=coordinator.phase.value,
                                    legal_actions=[
                                        action.value
                                        for action in coordinator.engine.state.legal_actions
                                    ],
                                )
                                gesture_temporal = GestureTemporalAdapter(gesture_config)
                                speech_adapter = SpeechObservationAdapter(speech_config)
                                multimodal.clear()
                                identity_guard_last_valid_ns = None
                                _clear_queue(audio_queue)
                                if speech_recognizer is not None:
                                    speech_recognizer.reset_window()
                                if coordinator.phase is PartAPhase.WAITING_ROTATION_ACK:
                                    command = coordinator.request_rotation(now_ns)
                                    _emit(
                                        "rotation_requested",
                                        command_id=command.command_id,
                                        target_slot=command.target_slot.value,
                                        state_version=coordinator.engine.state.state_version,
                                    )
                                    pending_rotation_due_ns = (
                                        now_ns + args.rotation_delay_ms * 1_000_000
                                    )
                            else:
                                rejected_actions += 1
                                status_text = coordinator.last_reason
                    else:
                        _clear_queue(audio_queue)

                    if args.headless:
                        continue
                    display = frame.image.copy()
                    height, width = display.shape[:2]
                    if face_evidence is not None:
                        for feature in face_evidence.features:
                            x, y, box_width, box_height = feature.bbox_xywh
                            cv2.rectangle(
                                display,
                                (x, y),
                                (x + box_width, y + box_height),
                                (0, 200, 255),
                                2,
                            )
                    if gesture_evidence is not None and gesture_evidence.centroid_x is not None:
                        cv2.circle(
                            display,
                            (
                                int(gesture_evidence.centroid_x * width),
                                int(gesture_evidence.centroid_y * height),  # type: ignore[arg-type]
                            ),
                            8,
                            (0, 255, 0),
                            2,
                        )
                    if coordinator is None:
                        phase_text = "SETUP"
                        focus_text = registration_target.value
                        expected_player = _registered_player(
                            gallery, registration_target
                        ) or PLAYER_BY_SEAT[registration_target]
                        legal_text = "press S after enrollment"
                    else:
                        phase_text = coordinator.phase.value
                        focus_text = coordinator.focus_seat.value if coordinator.focus_seat else "none"
                        expected_player = _registered_player(
                            gallery, coordinator.focus_seat
                        ) or (
                            PLAYER_BY_SEAT[coordinator.focus_seat]
                            if coordinator.focus_seat is not None
                            else "none"
                        )
                        legal_text = ",".join(
                            action.value for action in coordinator.engine.state.legal_actions
                        ) or "Part A boundary reached"
                    cv2.putText(
                        display,
                        f"{args.player_mode} | {phase_text} | focus {focus_text} | expected {expected_player} | gallery {gallery.size}/4",
                        (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        f"legal: {legal_text} | gesture: {last_gesture} | pending: {','.join(multimodal.pending_sources) or 'none'}",
                        (18, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        status_text,
                        (18, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.64,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        "1-4 target | E enroll | S start | C confirm speech | X clear | Q quit",
                        (18, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Poker Dealer - Sequential Part A Vertical Loop", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if coordinator is None and key in SEAT_KEYS:
                        registration_target = SEAT_KEYS[key]
                        enrollment_active = False
                        enrollment_samples = []
                        status_text = f"registration target {registration_target.value}"
                    elif key == ord("e"):
                        target = (
                            registration_target
                            if coordinator is None
                            else coordinator.focus_seat
                        )
                        if target is None:
                            status_text = "no state-owned focus seat"
                        elif not args.consent_confirmed:
                            status_text = "enrollment blocked: restart with --consent-confirmed"
                        elif any(item["seat"] == target.value for item in gallery.metadata()):
                            status_text = f"{target.value} already enrolled"
                        elif coordinator is not None and coordinator.phase is not PartAPhase.VERIFYING_IDENTITY:
                            status_text = "active enrollment only allowed during identity verification"
                        else:
                            enrollment_active = True
                            enrollment_samples = []
                            last_enrollment_sample_ns = None
                            status_text = f"ENROLL {PLAYER_BY_SEAT[target]}: one face only"
                            _emit(
                                "enrollment_started",
                                player_id=PLAYER_BY_SEAT[target],
                                seat=target.value,
                                player_mode=args.player_mode,
                            )
                    elif key == ord("x"):
                        if (
                            coordinator is not None
                            and coordinator.phase is not PartAPhase.VERIFYING_IDENTITY
                        ):
                            status_text = "gallery clear blocked while an action window is open"
                        else:
                            gallery.clear()
                            enrollment_active = False
                            enrollment_samples = []
                            status_text = "session gallery cleared"
                            _emit("gallery_cleared", player_mode=args.player_mode)
                    elif key == ord("c"):
                        if (
                            coordinator is not None
                            and coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION
                            and multimodal.pending_sources == ("speech",)
                        ):
                            speech_confirm_requested = True
                            status_text = "speech UI confirmation requested"
                        else:
                            status_text = "no pending speech candidate to confirm"
                    elif key == ord("s") and coordinator is None:
                        enrolled_seats = {
                            Seat(str(item["seat"])) for item in gallery.metadata()
                        }
                        button, expected_first, start_block_reason = (
                            _resolve_start_plan(
                                args.player_mode,
                                enrolled_seats,
                                Seat(args.button),
                            )
                        )
                        start_mode = args.player_mode
                        if start_block_reason is not None:
                            status_text = start_block_reason
                            _emit(
                                "hand_start_blocked",
                                mode=args.player_mode,
                                reason=start_block_reason,
                                enrolled_seats=sorted(
                                    seat.value for seat in enrolled_seats
                                ),
                            )
                        else:
                            engine = HandEngine.start(args.hand_id, button)
                            if (
                                expected_first is not None
                                and engine.state.acting_seat is not expected_first
                            ):
                                raise RuntimeError(
                                    "two-player pilot failed to align first acting seat"
                                )
                            engine.promoter = ActionPromoter(
                                PromotionPolicy(
                                    minimum_confidence=gesture_config.confirmation.minimum_score,
                                    minimum_stable_frames=3,
                                    minimum_stable_duration_ms=200,
                                )
                            )
                            coordinator = SequentialPartACoordinator(
                                engine, args.session_id
                            )
                            if args.player_mode == "two_player_pilot":
                                two_player_remaining = set(enrolled_seats)
                            command = coordinator.request_rotation(now_ns)
                            _emit(
                                "rotation_requested",
                                command_id=command.command_id,
                                target_slot=command.target_slot.value,
                                state_version=coordinator.engine.state.state_version,
                            )
                            pending_rotation_due_ns = (
                                now_ns + args.rotation_delay_ms * 1_000_000
                            )
                            enrollment_active = False
                            status_text = (
                                f"{start_mode}: Button {button.value}, first "
                                f"{coordinator.focus_seat.value}"
                            )
                            print(
                                json.dumps(
                                    {
                                        "type": "hand_started",
                                        "mode": start_mode,
                                        "button": button.value,
                                        "first_acting_seat": coordinator.focus_seat.value,
                                        "enrolled_seats": sorted(
                                            seat.value for seat in enrolled_seats
                                        ),
                                    },
                                    ensure_ascii=True,
                                )
                            )
    except (CameraError, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=True,
            )
        )
        return 2
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    print(
        json.dumps(
            {
                "type": "summary",
                "status": "completed" if frames else "no_readable_frames",
                "frames": frames,
                "missing_reads": missing_reads,
                "elapsed_seconds": elapsed_s,
                "accepted_actions": accepted_actions,
                "rejected_actions": rejected_actions,
                "identity_matches": identity_matches,
                "simulated_rotation_acks": simulated_rotation_acks,
                "dropped_audio_blocks": dropped_audio_blocks,
                "final_phase": coordinator.phase.value if coordinator else "setup",
                "player_mode": args.player_mode,
                "frames_saved": 0,
                "audio_saved": False,
                "embeddings_persisted": False,
                "physical_robot_connected": False,
            },
            ensure_ascii=True,
        )
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
