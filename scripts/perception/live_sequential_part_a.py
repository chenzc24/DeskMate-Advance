"""Run the sequential identity and multimodal action loop."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import queue
import time

import cv2

from poker_dealer.domain import (
    ActionEvidenceState,
    ControlIntent,
    DealerCommand,
    DealerCommandType,
    LaptopControlAdapter,
    PlayerActionType,
    SEAT_ORDER,
    Seat,
    TableRole,
    role_for_seat,
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
    GestureFrameEvidence,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    MultimodalActionWindow,
    SpeakerVerificationConfig,
    SpeechConfirmationController,
    SpeechConfirmationStatus,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeechIntentKind,
    VoskSpeechRecognizer,
    classify_speech_intent,
    observation_to_dict,
)
from poker_dealer.perception.attribution import (
    ActorAttributionConfig,
    ActorBindingLease,
    AttributedActionCandidate,
    HandAttributionState,
    MediaPipePoseAdapter,
    SessionSpeakerGallery,
    SpeakerVerificationState,
    TargetPersonTracker,
    actor_binding_to_dict,
    attribute_hands_to_target,
)
from poker_dealer.perception.identity import (
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
)
from poker_dealer.runtime import (
    ConsoleAnnouncer,
    EventAnnouncer,
    PartAPhase,
    RegistrationPhase,
    RegistrationRuntime,
    SequentialPartACoordinator,
    VisualSettleGate,
    VisualSettleState,
    WindowsSpeechAnnouncer,
)

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - clear CLI diagnostic
    sd = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SEAT_KEYS = {ord("1"): Seat.A, ord("2"): Seat.B, ord("3"): Seat.C, ord("4"): Seat.D}
PLAYER_BY_SEAT = {seat: f"player_{seat.value[-1]}" for seat in SEAT_ORDER}
ROLE_KEYS = {
    ord("1"): TableRole.BUTTON,
    ord("2"): TableRole.SMALL_BLIND,
    ord("3"): TableRole.BIG_BLIND,
    ord("4"): TableRole.UNDER_THE_GUN,
}
ROLE_LABELS = {
    TableRole.BUTTON: "Button",
    TableRole.SMALL_BLIND: "Small Blind",
    TableRole.BIG_BLIND: "Big Blind",
    TableRole.UNDER_THE_GUN: "Under the Gun",
}
VOICE_ENROLLMENT_PHRASES = (
    "fold check call bet raise confirm cancel",
    "check call raise bet fold cancel confirm",
    "raise bet call check fold confirm cancel",
)
VOICE_PROMPT_GUARD_NS = 4_000_000_000
_EVENT_CONTEXT: dict[str, object] = {}
_EVENT_LOG_PATH: Path | None = None


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
    parser.add_argument(
        "--attribution-config",
        type=Path,
        default=ROOT / "configs/perception/actor_binding_session.json",
    )
    parser.add_argument(
        "--speaker-config",
        type=Path,
        default=ROOT / "configs/perception/speaker_verification_session.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument(
        "--stream-url",
        help="HTTP(S) MJPEG stream; mutually exclusive with --index",
    )
    parser.add_argument("--stream-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--stream-read-timeout-ms", type=int, default=2000)
    parser.add_argument("--speech-device", type=_device)
    parser.add_argument("--disable-speech", action="store_true")
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument(
        "--player-mode",
        choices=("four_player_core", "two_player_pilot", "single_player_pilot"),
        default="four_player_core",
    )
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rotation-delay-ms", type=int, default=350)
    parser.add_argument("--identity-grace-ms", type=int, default=1000)
    parser.add_argument("--session-id", default="sequential-part-a-pilot")
    parser.add_argument("--hand-id", default="sequential-part-a-hand")
    parser.add_argument("--acceptance-case", default="UNASSIGNED")
    parser.add_argument("--acceptance-session-group", default="UNASSIGNED")
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument(
        "--button",
        choices=tuple(seat.value for seat in Seat),
        help="required in four-player Core; physical seat holding Button this hand",
    )
    parser.add_argument(
        "--announcer",
        choices=("off", "console", "windows"),
        default="windows",
        help="committed-event feedback; Windows mode speaks without blocking the camera loop",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def _camera_config(
    args: argparse.Namespace, identity_config: FaceIdentityConfig
) -> CameraConfig:
    camera_values = identity_config.camera
    if args.stream_url is not None:
        if args.index is not None:
            raise ValueError("--stream-url and --index are mutually exclusive")
        if args.backend not in {None, "auto"}:
            raise ValueError("network streams use FFmpeg; omit --backend")
        return CameraConfig(
            device_index=0,
            stream_url=args.stream_url,
            source_id="robot_mjpeg_stream",
            backend="auto",
            width=None,
            height=None,
            fps=None,
            open_timeout_ms=args.stream_open_timeout_ms,
            read_timeout_ms=args.stream_read_timeout_ms,
        )
    return CameraConfig(
        device_index=(
            int(camera_values["device_index"])
            if args.index is None
            else args.index
        ),
        source_id="sequential_part_a_pilot",
        backend=(
            str(camera_values["backend"])
            if args.backend is None
            else args.backend
        ),
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fps=float(camera_values["fps"]),
    )


def _clear_queue(audio_queue: queue.Queue[bytes]) -> int:
    cleared = 0
    while True:
        try:
            audio_queue.get_nowait()
            cleared += 1
        except queue.Empty:
            return cleared


def _configure_event_output(
    *,
    session_id: str,
    hand_id: str,
    acceptance_case: str,
    log_path: Path | None,
    acceptance_session_group: str = "UNASSIGNED",
) -> None:
    global _EVENT_CONTEXT, _EVENT_LOG_PATH
    _EVENT_CONTEXT = {
        "session_id": session_id,
        "hand_id": hand_id,
        "acceptance_case": acceptance_case,
        "acceptance_session_group": acceptance_session_group,
    }
    _EVENT_LOG_PATH = log_path
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8"):
            pass


def _event_record(event_type: str, **payload: object) -> dict[str, object]:
    return {
        "type": event_type,
        **_EVENT_CONTEXT,
        "logged_at_monotonic_ns": time.monotonic_ns(),
        **payload,
    }


def _emit(event_type: str, **payload: object) -> None:
    rendered = json.dumps(_event_record(event_type, **payload), ensure_ascii=True)
    print(rendered, flush=True)
    if _EVENT_LOG_PATH is not None:
        with _EVENT_LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


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


def _role_label(button: Seat, seat: Seat | None) -> str:
    if seat is None:
        return "none"
    return ROLE_LABELS[role_for_seat(button, seat)]


def _registration_player_id(
    registration: RegistrationRuntime | None, seat: Seat
) -> str:
    if registration is not None:
        if registration.focus_seat is not seat:
            raise ValueError("registration runtime focus does not match capture seat")
        return registration.participant_id
    return PLAYER_BY_SEAT[seat]


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
    if player_mode == "single_player_pilot":
        if len(enrolled_seats) != 1:
            return (
                configured_button,
                None,
                "single_player_pilot requires exactly one registered seat",
            )
        expected_first = next(iter(enrolled_seats))
        button = SEAT_ORDER[
            (SEAT_ORDER.index(expected_first) + 1) % len(SEAT_ORDER)
        ]
        return button, expected_first, None
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


def _consume_pilot_action(
    player_mode: str, remaining: set[Seat], acted_seat: Seat
) -> str | None:
    """Return the explicit non-product completion reason, if reached."""

    if player_mode == "four_player_core":
        return None
    if player_mode not in {"single_player_pilot", "two_player_pilot"}:
        raise ValueError("unsupported player mode")
    remaining.discard(acted_seat)
    if remaining:
        return None
    if player_mode == "single_player_pilot":
        return "single_registered_player_completed_one_action"
    return "two_registered_players_completed_one_action"


def main() -> int:
    args = parse_args()
    if args.player_mode == "four_player_core" and args.button is None:
        raise SystemExit(
            "--button is required for four_player_core; select the physical seat "
            "that currently holds Button"
        )
    if (
        args.max_seconds <= 0
        or args.rotation_delay_ms < 0
        or args.identity_grace_ms < 0
    ):
        raise SystemExit("runtime duration must be positive and rotation delay non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    try:
        _configure_event_output(
            session_id=args.session_id,
            hand_id=args.hand_id,
            acceptance_case=args.acceptance_case,
            log_path=args.log_jsonl,
            acceptance_session_group=args.acceptance_session_group,
        )
    except FileExistsError as exc:
        raise SystemExit(f"--log-jsonl must not already exist: {exc.filename}") from exc
    if not args.disable_speech and sd is None:
        _emit("error", error="sounddevice is unavailable")
        return 2

    identity_config = FaceIdentityConfig.from_json(args.identity_config)
    gesture_config = GesturePilotConfig.from_json(args.gesture_config)
    speech_config = SpeechPilotConfig.from_json(args.speech_config)
    speaker_config = SpeakerVerificationConfig.from_json(args.speaker_config)
    attribution_config = ActorAttributionConfig.from_json(args.attribution_config)
    gesture_config = replace(
        gesture_config,
        model=replace(gesture_config.model, num_hands=attribution_config.max_hands),
        calibration_version=f"{gesture_config.calibration_version}-actor-bound-v1",
    )
    try:
        camera_config = _camera_config(args, identity_config)
    except ValueError as exc:
        raise SystemExit(f"invalid camera configuration: {exc}") from exc

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
    announcer_port = None
    if args.announcer == "console":
        announcer_port = ConsoleAnnouncer()
    elif args.announcer == "windows":
        announcer_port = WindowsSpeechAnnouncer()
    event_announcer = EventAnnouncer(announcer_port) if announcer_port is not None else None
    laptop_controls = LaptopControlAdapter()
    registration = (
        RegistrationRuntime(args.session_id, Seat(args.button))
        if args.player_mode == "four_player_core"
        else None
    )
    coordinator: SequentialPartACoordinator | None = None
    pending_rotation_due_ns: int | None = None
    registration_target = registration.focus_seat if registration is not None else Seat.A
    enrollment_active = False
    enrollment_samples = []
    last_enrollment_sample_ns: int | None = None
    speaker_enrollment_active = False
    speaker_enrollment_player_id: str | None = None
    speaker_enrollment_role: str | None = None
    speaker_enrollment_samples = []
    speaker_enrollment_listen_after_ns = 0
    identity_temporal = FaceIdentityTemporalAdapter(identity_config)
    gesture_temporal = GestureTemporalAdapter(gesture_config)
    actor_lease = ActorBindingLease(lease_ms=attribution_config.actor_lease_ms)
    person_tracker = TargetPersonTracker(attribution_config)
    visual_settle = VisualSettleGate()
    multimodal = MultimodalActionWindow(
        decision_wait_ms=500,
        max_skew_ms=3000,
        allow_speech_single_source=args.player_mode != "four_player_core",
    )
    speech_recognizer: VoskSpeechRecognizer | None = None
    speech_adapter = SpeechObservationAdapter(speech_config)
    speech_confirmation = SpeechConfirmationController(
        confirmation_timeout_ms=speaker_config.confirmation_timeout_ms,
        require_speaker_match=True,
    )
    frames = 0
    missing_reads = 0
    dropped_camera_frames = 0
    camera_reconnects = 0
    accepted_actions = 0
    rejected_actions = 0
    identity_matches = 0
    simulated_rotation_acks = 0
    status_text = (
        "SETUP: 1 Button, 2 Small Blind, 3 Big Blind, 4 UTG; E enroll; S start"
        if registration is not None
        else "SETUP: choose 1-4, E enroll, S start"
    )
    last_gesture = "no hand"
    last_identity_log_key: tuple[object, ...] | None = None
    last_gesture_log_key: tuple[object, ...] | None = None
    last_camera_read_log_key: tuple[object, ...] | None = None
    last_binding_log_key: tuple[object, ...] | None = None
    last_visual_settle_log_key: tuple[object, ...] | None = None
    last_hand_attribution_confidence: float | None = None
    verified_speech_similarity: float | None = None
    speech_confirm_requested = False
    pilot_remaining: set[Seat] = set()
    required_player_count = {
        "single_player_pilot": 1,
        "two_player_pilot": 2,
        "four_player_core": 4,
    }[args.player_mode]
    started_ns = time.monotonic_ns()

    try:
        if not args.disable_speech:
            speech_recognizer = VoskSpeechRecognizer(speech_config, speaker_config)
        with SessionFaceGallery(identity_config, args.session_id) as gallery, SessionSpeakerGallery(
            args.session_id,
            minimum_samples=speaker_config.minimum_samples,
            minimum_speaker_frames=speaker_config.minimum_speaker_frames,
            minimum_similarity=speaker_config.minimum_similarity,
            minimum_margin=speaker_config.minimum_margin,
        ) as speaker_gallery:
            with OpenCVCamera(camera_config) as camera, MediaPipeGestureAdapter(
                gesture_config
            ) as gesture_model, MediaPipePoseAdapter(
                attribution_config
            ) as pose_model, stream_context:
                face_model = OpenCvFaceIdentityAdapter(identity_config)
                camera_summary = camera.negotiated_properties()
                _emit(
                    "ready",
                    runtime="sequential_part_a_vertical_loop",
                    player_mode=args.player_mode,
                    camera=camera_summary,
                    microphone=microphone_summary,
                    rotation_adapter="simulated_dealer_only",
                    consent_confirmed=args.consent_confirmed,
                    frames_saved=0,
                    audio_saved=False,
                    embeddings_persisted=False,
                    speaker_embeddings_memory_only=True,
                    speaker_model=speaker_config.model.model_id,
                    actor_binding_required=True,
                    pose_model=attribution_config.pose_model_id,
                    max_hands=attribution_config.max_hands,
                    four_player_speech_authority=(
                        "same_speaker_spoken_confirm_or_gesture_or_ui_override"
                    ),
                    pilot_completion_actions_per_registered_player=(
                        1 if args.player_mode != "four_player_core" else None
                    ),
                    initial_button=(registration.button.value if registration else None),
                    public_identity_labels="roles_only",
                    control_sources=["laptop_keyboard", "robot_button_interface"],
                    announcer=args.announcer,
                )
                if event_announcer is not None and registration is not None:
                    event_announcer.publish(
                        "registration_focus_changed",
                        role=registration.focus_role.value,
                    )

                while (time.monotonic_ns() - started_ns) / 1_000_000_000 < args.max_seconds:
                    if args.max_frames is not None and frames >= args.max_frames:
                        break
                    read = camera.read()
                    current_reconnects = camera.network_reconnects
                    if current_reconnects > camera_reconnects:
                        camera_reconnects = current_reconnects
                        if (
                            coordinator is not None
                            and coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION
                        ):
                            coordinator.revoke_identity("camera_epoch_changed")
                            actor_lease.revoke("camera_epoch_changed")
                            person_tracker.clear()
                            gesture_temporal = GestureTemporalAdapter(gesture_config)
                            multimodal.clear()
                            speech_confirmation.clear()
                            verified_speech_similarity = None
                            _clear_queue(audio_queue)
                        _emit(
                            "camera_reconnected",
                            reconnect_count=camera_reconnects,
                            source_id=camera_config.source_id,
                            state_version=(
                                coordinator.engine.state.state_version
                                if coordinator is not None
                                else None
                            ),
                            phase=(coordinator.phase.value if coordinator else "setup"),
                        )
                    if read.status is not CameraReadStatus.OK or read.frame is None:
                        missing_reads += 1
                        camera_read_log_key = (
                            read.status,
                            read.reason,
                            camera.network_reconnecting,
                        )
                        if args.emit_all or camera_read_log_key != last_camera_read_log_key:
                            _emit(
                                "camera_read_status",
                                status=read.status.value,
                                reason=read.reason,
                                consecutive_failures=read.consecutive_failures,
                                reconnecting=camera.network_reconnecting,
                            )
                            last_camera_read_log_key = camera_read_log_key
                        if read.status is CameraReadStatus.DISCONNECTED:
                            status_text = "camera disconnected"
                            break
                        continue
                    last_camera_read_log_key = None
                    frames += 1
                    frame = read.frame
                    dropped_camera_frames += frame.dropped_before
                    now_ns = time.monotonic_ns()
                    face_evidence = None
                    pose_evidence = None
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
                        if (
                            accepted_ack
                            and coordinator.phase is PartAPhase.WAITING_VISUAL_SETTLE
                        ):
                            visual_settle.begin(
                                started_at_ns=now_ns,
                                sequence_watermark=frame.sequence_id,
                                camera_epoch=camera_reconnects,
                            )
                        status_text = coordinator.last_reason

                    phase = coordinator.phase if coordinator is not None else None
                    if phase is PartAPhase.WAITING_VISUAL_SETTLE:
                        settle_observation = visual_settle.observe(
                            frame, camera_epoch=camera_reconnects
                        )
                        settle_log_key = (
                            settle_observation.state,
                            settle_observation.reason,
                            settle_observation.stable_frames,
                        )
                        if args.emit_all or settle_log_key != last_visual_settle_log_key:
                            _emit(
                                "visual_settle",
                                state=settle_observation.state.value,
                                reason=settle_observation.reason,
                                sequence_id=settle_observation.sequence_id,
                                new_frames=settle_observation.new_frames,
                                stable_frames=settle_observation.stable_frames,
                                mean_absdiff=settle_observation.mean_absdiff,
                                camera_epoch=camera_reconnects,
                            )
                            last_visual_settle_log_key = settle_log_key
                        if settle_observation.state is VisualSettleState.SETTLED:
                            coordinator.accept_visual_settle()
                            visual_settle.clear()
                        elif settle_observation.state is VisualSettleState.TIMED_OUT:
                            coordinator.fail_visual_settle(settle_observation.reason)
                        status_text = coordinator.last_reason
                        phase = coordinator.phase
                    if phase in {
                        None,
                        PartAPhase.VERIFYING_IDENTITY,
                        PartAPhase.WAITING_PLAYER_ACTION,
                    }:
                        face_evidence = face_model.analyze(frame)
                        pose_evidence = pose_model.recognize(frame)

                        target = (
                            registration_target
                            if coordinator is None
                            else coordinator.focus_seat
                        )
                        assert target is not None
                        if enrollment_active:
                            registration_player_id = _registration_player_id(
                                registration, target
                            )
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
                                f"ENROLL {_role_label(registration.button, target) if registration else target.value} "
                                f"{len(enrollment_samples)}/"
                                f"{identity_config.minimum_samples}"
                            )
                            if len(enrollment_samples) >= identity_config.minimum_samples:
                                try:
                                    gallery.enroll(
                                        registration_player_id,
                                        target,
                                        enrollment_samples,
                                        consent_granted=args.consent_confirmed,
                                    )
                                    if registration is not None:
                                        participant = registration.complete_face_enrollment(
                                            len(enrollment_samples)
                                        )
                                        public_role = participant.initial_role.value
                                        status_text = (
                                            f"enrolled {ROLE_LABELS[participant.initial_role]}"
                                        )
                                    else:
                                        public_role = target.value
                                        status_text = f"enrolled {target.value}"
                                    _emit(
                                        "enrollment_completed",
                                        player_id=registration_player_id,
                                        seat=target.value,
                                        role=public_role,
                                        sample_count=len(enrollment_samples),
                                        gallery_size=gallery.size,
                                    )
                                    if event_announcer is not None:
                                        event_announcer.publish(
                                            "enrollment_completed", role=public_role
                                        )
                                        if gallery.size == required_player_count:
                                            event_announcer.publish("roster_ready")
                                except (PermissionError, ValueError) as exc:
                                    if (
                                        registration is not None
                                        and registration.phase
                                        is RegistrationPhase.CAPTURING_FACE
                                    ):
                                        registration.reject_face_enrollment()
                                    status_text = f"enrollment rejected: {exc}"
                                    _emit(
                                        "enrollment_rejected",
                                        player_id=registration_player_id,
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
                            expected_player = _role_label(
                                coordinator.engine.state.button,
                                coordinator.focus_seat,
                            )
                            if opened:
                                assert pose_evidence is not None
                                if len(face_evidence.features) != 1:
                                    coordinator.revoke_identity(
                                        "actor_binding_requires_one_target_face"
                                    )
                                    opened = False
                                else:
                                    target_track = person_tracker.acquire(
                                        pose_evidence.poses,
                                        face_bbox_xywh=face_evidence.features[0].bbox_xywh,
                                        frame_width=frame.width,
                                        frame_height=frame.height,
                                        observed_at_ns=face_evidence.observed_at_ns,
                                    )
                                    if target_track is None:
                                        coordinator.revoke_identity(
                                            "target_person_pose_not_acquired"
                                        )
                                        opened = False
                                    else:
                                        binding = actor_lease.open(
                                            identity_observation,
                                            hand_id=coordinator.engine.state.hand_id,
                                            person_track_id=target_track.track_id,
                                            camera_epoch=camera_reconnects,
                                        )
                                        coordinator.bind_actor(binding)
                                        _emit(
                                            "actor_binding_opened",
                                            **actor_binding_to_dict(binding),
                                            pose_detector_index=(
                                                target_track.pose.detector_index
                                            ),
                                        )
                            if opened:
                                status_text = coordinator.last_reason
                            elif identity_observation.identity_state is FaceIdentityState.SEAT_MISMATCH:
                                observed_role = _role_label(
                                    coordinator.engine.state.button,
                                    identity_observation.registered_seat,
                                )
                                status_text = (
                                    f"MISMATCH saw {observed_role}; "
                                    f"expected {expected_player}"
                                )
                            elif identity_observation.identity_state is FaceIdentityState.ENROLLMENT_REQUIRED:
                                status_text = f"ENROLLMENT REQUIRED: expected {expected_player}"
                            elif (
                                identity_observation.identity_state
                                is FaceIdentityState.EXPECTED_SEAT_UNENROLLED
                            ):
                                status_text = (
                                    f"EXPECTED ROLE UNENROLLED: {expected_player}"
                                )
                            elif identity_observation.identity_state is FaceIdentityState.MATCHED:
                                status_text = coordinator.last_reason
                            else:
                                status_text = (
                                    f"{identity_observation.identity_state.value}: "
                                    f"expected {expected_player}"
                                )
                            if opened:
                                identity_matches += 1
                                gesture_temporal = GestureTemporalAdapter(gesture_config)
                                speech_adapter = SpeechObservationAdapter(speech_config)
                                multimodal.clear()
                                speech_confirmation.clear()
                                verified_speech_similarity = None
                                _clear_queue(audio_queue)
                                if speech_recognizer is not None:
                                    speech_recognizer.reset_window()
                                _emit(
                                    "identity_gate_opened",
                                    player_id=coordinator.verified_player_id,
                                    focus_seat=coordinator.focus_seat.value,  # type: ignore[union-attr]
                                    state_version=coordinator.engine.state.state_version,
                                )

                    face_guard_current = False
                    if (
                        coordinator is not None
                        and coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION
                        and face_evidence is not None
                        and pose_evidence is not None
                    ):
                        focus_seat = coordinator.focus_seat
                        assert focus_seat is not None
                        guard_match = gallery.match_expected_seat(
                            face_evidence, focus_seat
                        )
                        guard_observation = identity_temporal.process(
                            guard_match,
                            face_evidence.observed_at_ns,
                            FaceIdentityContext(
                                args.session_id,
                                coordinator.engine.state.state_version,
                                focus_seat,
                            ),
                        )
                        actor_lease.observe_identity(guard_observation)
                        target_track = person_tracker.update(
                            pose_evidence.poses,
                            observed_at_ns=pose_evidence.observed_at_ns,
                        )
                        binding = actor_lease.binding
                        face_guard_current = (
                            binding is not None
                            and actor_lease.is_valid_at(
                                face_evidence.observed_at_ns,
                                camera_epoch=camera_reconnects,
                            )
                            and target_track is not None
                        )
                        if binding is not None:
                            coordinator.bind_actor(binding)
                        binding_log_key = (
                            actor_lease.state,
                            actor_lease.last_reason,
                            target_track.track_id if target_track else None,
                            coordinator.engine.state.state_version,
                        )
                        if args.emit_all or binding_log_key != last_binding_log_key:
                            _emit(
                                "actor_binding_status",
                                state=actor_lease.state.value,
                                reason=actor_lease.last_reason,
                                binding_id=(binding.binding_id if binding else None),
                                player_id=(binding.player_id if binding else None),
                                person_track_id=(
                                    target_track.track_id if target_track else None
                                ),
                                focus_seat=focus_seat.value,
                                state_version=coordinator.engine.state.state_version,
                            )
                            last_binding_log_key = binding_log_key
                        if not actor_lease.is_valid_at(face_evidence.observed_at_ns):
                            reason = actor_lease.last_reason
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
                            person_tracker.clear()
                            actor_lease.clear()
                            gesture_temporal = GestureTemporalAdapter(gesture_config)
                            speech_adapter = SpeechObservationAdapter(speech_config)
                            multimodal.clear()
                            speech_confirmation.clear()
                            verified_speech_similarity = None
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
                        assert pose_evidence is not None
                        raw_hands = gesture_model.recognize_all(frame)
                        target_track = (
                            person_tracker.track if face_guard_current else None
                        )
                        if target_track is None:
                            hand_attribution = None
                            gesture_evidence = GestureFrameEvidence(
                                observed_at_ns=frame.captured_at_ns,
                                hand_present=False,
                                hand_in_focus_roi=False,
                                gesture_label=None,
                                gesture_score=None,
                                quality_flags=("target_person_track_missing",),
                            )
                            last_hand_attribution_confidence = None
                        else:
                            hand_attribution = attribute_hands_to_target(
                                raw_hands,
                                pose_evidence.poses,
                                target_pose_detector_index=(
                                    target_track.pose.detector_index
                                ),
                                config=attribution_config,
                            )
                            gesture_evidence = hand_attribution.temporal_evidence(
                                frame.captured_at_ns
                            )
                            last_hand_attribution_confidence = (
                                hand_attribution.attribution_confidence
                            )
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
                                attribution_state=(
                                    hand_attribution.state.value
                                    if hand_attribution is not None
                                    else HandAttributionState.AMBIGUOUS.value
                                ),
                                target_hand_count=(
                                    hand_attribution.target_hand_count
                                    if hand_attribution is not None
                                    else 0
                                ),
                                rejected_hand_count=(
                                    hand_attribution.rejected_hand_count
                                    if hand_attribution is not None
                                    else len(raw_hands)
                                ),
                                actor_binding_id=(
                                    actor_lease.binding.binding_id
                                    if actor_lease.binding is not None
                                    else None
                                ),
                                focus_seat=context.focus_seat.value,
                                state_version=context.expected_state_version,
                            )
                            last_gesture_log_key = gesture_log_key
                        fused = None
                        if speech_confirmation.expire(now_ns):
                            multimodal.cancel_pending_speech()
                            verified_speech_similarity = None
                            _emit(
                                "speech_confirmation_state",
                                status=SpeechConfirmationStatus.EXPIRED.value,
                                reason="pending_speech_action_expired",
                                focus_seat=context.focus_seat.value,
                                state_version=context.expected_state_version,
                            )
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
                            speech_intent = classify_speech_intent(
                                speech_evidence, speech_config
                            )
                            _emit(
                                "speech_observation",
                                transcript=speech_evidence.canonical_transcript,
                                confidence=speech_evidence.confidence,
                                intent=speech_intent.kind.value,
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
                            speaker_result = (
                                speaker_gallery.match(
                                    speech_evidence.speaker_embedding,
                                    speaker_frames=speech_evidence.speaker_frames,
                                )
                                if speech_evidence.speaker_embedding is not None
                                else None
                            )
                            speaker_state = (
                                speaker_result.state.value
                                if speaker_result is not None
                                else SpeakerVerificationState.INSUFFICIENT_AUDIO.value
                            )
                            speaker_player_id = (
                                speaker_result.player_id
                                if speaker_result is not None
                                and speaker_result.state
                                is SpeakerVerificationState.MATCHED
                                else None
                            )
                            _emit(
                                "speaker_verification",
                                state=speaker_state,
                                player_id=speaker_player_id,
                                similarity=(
                                    speaker_result.similarity
                                    if speaker_result is not None
                                    else None
                                ),
                                second_best_similarity=(
                                    speaker_result.second_best_similarity
                                    if speaker_result is not None
                                    else None
                                ),
                                speaker_frames=speech_evidence.speaker_frames,
                                focus_seat=context.focus_seat.value,
                                state_version=context.expected_state_version,
                                embedding_logged=False,
                            )
                            binding = actor_lease.binding
                            if binding is None:
                                _emit(
                                    "speech_intent_rejected",
                                    intent=speech_intent.kind.value,
                                    reason="actor_binding_required",
                                    focus_seat=context.focus_seat.value,
                                    state_version=context.expected_state_version,
                                )
                                continue
                            if speaker_player_id != binding.player_id:
                                _emit(
                                    "speech_intent_rejected",
                                    intent=speech_intent.kind.value,
                                    reason="speaker_does_not_match_bound_player",
                                    speaker_state=speaker_state,
                                    expected_player_id=binding.player_id,
                                    observed_player_id=speaker_player_id,
                                    focus_seat=context.focus_seat.value,
                                    state_version=context.expected_state_version,
                                )
                                continue
                            verified_speech_similarity = (
                                speaker_result.similarity
                                if speaker_result is not None
                                else None
                            )
                            if speech_intent.kind is SpeechIntentKind.ACTION:
                                confirmation = speech_confirmation.offer_action(
                                    speech_observation,
                                    binding,
                                    speaker_player_id=speaker_player_id,
                                )
                                _emit(
                                    "speech_confirmation_state",
                                    status=confirmation.status.value,
                                    reason=confirmation.reason,
                                    candidate_action=(
                                        speech_intent.action.value
                                        if speech_intent.action
                                        else None
                                    ),
                                    player_id=speaker_player_id,
                                )
                                if confirmation.status is SpeechConfirmationStatus.PENDING:
                                    speech_fused = multimodal.add(speech_observation)
                                    if speech_fused is not None:
                                        fused = speech_fused
                                    status_text = (
                                        "voice command pending: say confirm/cancel "
                                        "or show the matching gesture"
                                    )
                            elif speech_intent.kind in {
                                SpeechIntentKind.CONFIRM,
                                SpeechIntentKind.CANCEL,
                            }:
                                confirmation = speech_confirmation.handle_control(
                                    speech_intent,
                                    binding,
                                    speaker_player_id=speaker_player_id,
                                )
                                _emit(
                                    "speech_confirmation_state",
                                    status=confirmation.status.value,
                                    reason=confirmation.reason,
                                    player_id=speaker_player_id,
                                )
                                if confirmation.status is SpeechConfirmationStatus.CONFIRMED:
                                    fused = confirmation.observation
                                    multimodal.cancel_pending_speech()
                                elif confirmation.status is SpeechConfirmationStatus.CANCELLED:
                                    multimodal.cancel_pending_speech()
                                    status_text = "voice command cancelled"
                        if not face_guard_current:
                            _clear_queue(audio_queue)
                        if fused is None and face_guard_current:
                            fused = multimodal.poll(now_ns)
                        if (
                            args.player_mode == "four_player_core"
                            and multimodal.pending_sources == ("speech",)
                        ):
                            status_text = (
                                "speech pending: same speaker confirm, matching gesture, or C UI override"
                            )
                        if fused is not None:
                            before_version = coordinator.engine.state.state_version
                            acted_seat = fused.focus_seat
                            before_stack_units = coordinator.engine.state.players[
                                acted_seat
                            ].stack_units
                            fusion_source_flag = next(
                                (
                                    flag
                                    for flag in fused.quality_flags
                                    if flag.startswith("fusion_sources:")
                                ),
                                "fusion_sources:unknown",
                            )
                            binding = actor_lease.binding
                            outcome = None
                            rejection_reason = None
                            operator_speech_override = (
                                fusion_source_flag == "fusion_sources:speech"
                                and "speech_ui_confirmed" in fused.quality_flags
                            )
                            verified_speech = (
                                fusion_source_flag == "fusion_sources:speech_verified"
                                and "speaker_verified_same_actor" in fused.quality_flags
                            )
                            if (
                                fusion_source_flag == "fusion_sources:speech"
                                and not operator_speech_override
                            ):
                                rejection_reason = "speaker_verification_required"
                            elif binding is None:
                                rejection_reason = "actor_binding_required"
                            elif verified_speech and verified_speech_similarity is None:
                                rejection_reason = "speaker_similarity_required"
                            elif (
                                not operator_speech_override
                                and not verified_speech
                                and last_hand_attribution_confidence is None
                            ):
                                rejection_reason = "target_hand_attribution_required"
                            else:
                                attributed = AttributedActionCandidate(
                                    observation=fused,
                                    binding=binding,
                                    attribution_source=(
                                        "operator_ui_speech_override"
                                        if operator_speech_override
                                        else (
                                            "session_speaker_verification"
                                            if verified_speech
                                            else "face_pose_wrist"
                                        )
                                    ),
                                    attribution_confidence=(
                                        binding.identity_confidence
                                        if operator_speech_override
                                        else (
                                            min(
                                                verified_speech_similarity or 0.0,
                                                fused.confidence or 0.0,
                                            )
                                            if verified_speech
                                            else last_hand_attribution_confidence
                                        )
                                    ),
                                    quality_flags=(fusion_source_flag,),
                                )
                                outcome = coordinator.accept_attributed_action(
                                    attributed
                                )
                            _emit(
                                "multimodal_action_decision",
                                **observation_to_dict(fused),
                                actor_binding_id=(
                                    binding.binding_id if binding else None
                                ),
                                attribution_source=(
                                    (
                                        "operator_ui_speech_override"
                                        if operator_speech_override
                                        else (
                                            "session_speaker_verification"
                                            if verified_speech
                                            else "face_pose_wrist"
                                        )
                                    )
                                    if outcome is not None
                                    else None
                                ),
                                accepted=(outcome.accepted if outcome else False),
                                reason=(
                                    outcome.reason if outcome else rejection_reason
                                ),
                                next_seat=(
                                    outcome.next_seat.value
                                    if outcome is not None and outcome.next_seat
                                    else None
                                ),
                            )
                            if outcome is not None and outcome.accepted:
                                accepted_actions += 1
                                status_text = coordinator.last_reason
                                if event_announcer is not None:
                                    after_stack_units = coordinator.engine.state.players[
                                        acted_seat
                                    ].stack_units
                                    contribution = max(
                                        0, before_stack_units - after_stack_units
                                    )
                                    event_announcer.publish(
                                        "action_committed",
                                        role=role_for_seat(
                                            coordinator.engine.state.button,
                                            acted_seat,
                                        ).value,
                                        action=(
                                            fused.candidate_action.value
                                            if fused.candidate_action
                                            else "action"
                                        ),
                                        amount_units=(contribution or None),
                                    )
                                if args.player_mode in {
                                    "single_player_pilot",
                                    "two_player_pilot",
                                }:
                                    completion_reason = _consume_pilot_action(
                                        args.player_mode,
                                        pilot_remaining,
                                        acted_seat,
                                    )
                                    if completion_reason is not None:
                                        coordinator.complete_pilot(completion_reason)
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
                                speech_confirmation.clear()
                                verified_speech_similarity = None
                                actor_lease.clear()
                                person_tracker.clear()
                                last_hand_attribution_confidence = None
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
                                    if event_announcer is not None:
                                        event_announcer.publish(
                                            "turn_started",
                                            role=role_for_seat(
                                                coordinator.engine.state.button,
                                                coordinator.focus_seat,  # type: ignore[arg-type]
                                            ).value,
                                        )
                            else:
                                rejected_actions += 1
                                status_text = (
                                    coordinator.last_reason
                                    if outcome is not None
                                    else str(rejection_reason)
                                )
                    else:
                        if speaker_enrollment_active and speech_recognizer is not None:
                            if now_ns < speaker_enrollment_listen_after_ns:
                                _clear_queue(audio_queue)
                            while now_ns >= speaker_enrollment_listen_after_ns:
                                try:
                                    pcm = audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                                evidence = speech_recognizer.accept_audio(
                                    pcm, time.monotonic_ns()
                                )
                                if evidence is None:
                                    continue
                                accepted_sample = (
                                    evidence.speaker_embedding is not None
                                    and evidence.speaker_frames
                                    >= speaker_config.minimum_speaker_frames
                                )
                                if accepted_sample:
                                    assert evidence.speaker_embedding is not None
                                    speaker_enrollment_samples.append(
                                        evidence.speaker_embedding.copy()
                                    )
                                accepted_count = len(speaker_enrollment_samples)
                                _emit(
                                    "speaker_enrollment_sample",
                                    player_id=speaker_enrollment_player_id,
                                    accepted=accepted_sample,
                                    sample_count=accepted_count,
                                    required_samples=speaker_config.minimum_samples,
                                    speaker_frames=evidence.speaker_frames,
                                    transcript=evidence.canonical_transcript,
                                )
                                if not accepted_sample:
                                    if evidence.canonical_transcript:
                                        phrase_number = min(
                                            accepted_count + 1,
                                            speaker_config.minimum_samples,
                                        )
                                        status_text = (
                                            f"VOICE NOT ACCEPTED | {accepted_count}/"
                                            f"{speaker_config.minimum_samples} | too short "
                                            f"({evidence.speaker_frames}/"
                                            f"{speaker_config.minimum_speaker_frames} frames) | "
                                            f"repeat phrase {phrase_number} slowly"
                                        )
                                        if event_announcer is not None:
                                            event_announcer.publish(
                                                "voice_enrollment_retry",
                                                role=speaker_enrollment_role,
                                                phrase_number=phrase_number,
                                            )
                                        speech_recognizer.reset_window()
                                        _clear_queue(audio_queue)
                                        speaker_enrollment_listen_after_ns = (
                                            time.monotonic_ns() + VOICE_PROMPT_GUARD_NS
                                        )
                                        break
                                    continue
                                if event_announcer is not None:
                                    event_announcer.publish(
                                        "voice_enrollment_sample_accepted",
                                        role=speaker_enrollment_role,
                                        sample_number=accepted_count,
                                        total_samples=speaker_config.minimum_samples,
                                    )
                                if (
                                    accepted_count >= speaker_config.minimum_samples
                                ):
                                    assert speaker_enrollment_player_id is not None
                                    speaker_gallery.enroll(
                                        speaker_enrollment_player_id,
                                        speaker_enrollment_samples,
                                    )
                                    if registration is not None:
                                        registered_participant = next(
                                            item
                                            for item in registration.participants
                                            if item.participant_id
                                            == speaker_enrollment_player_id
                                        )
                                        registration.mark_voice_enrolled(
                                            registered_participant.seat
                                        )
                                        voice_label = ROLE_LABELS[
                                            registered_participant.initial_role
                                        ]
                                    else:
                                        voice_label = speaker_enrollment_player_id
                                    for sample in speaker_enrollment_samples:
                                        sample.fill(0.0)
                                    speaker_enrollment_samples = []
                                    speaker_enrollment_active = False
                                    status_text = (
                                        f"VOICE COMPLETE | {voice_label} | "
                                        f"{speaker_config.minimum_samples}/"
                                        f"{speaker_config.minimum_samples} accepted"
                                    )
                                    _emit(
                                        "speaker_enrollment_completed",
                                        player_id=speaker_enrollment_player_id,
                                        gallery_size=speaker_gallery.size,
                                        embeddings_persisted=False,
                                        audio_saved=False,
                                    )
                                    if event_announcer is not None:
                                        event_announcer.publish(
                                            "voice_enrollment_completed",
                                            role=speaker_enrollment_role,
                                        )
                                    speaker_enrollment_player_id = None
                                    speaker_enrollment_role = None
                                    speaker_enrollment_listen_after_ns = 0
                                    speech_recognizer.reset_window()
                                    break
                                next_phrase_number = accepted_count + 1
                                next_phrase = VOICE_ENROLLMENT_PHRASES[
                                    (next_phrase_number - 1)
                                    % len(VOICE_ENROLLMENT_PHRASES)
                                ]
                                status_text = (
                                    f"VOICE ACCEPTED {accepted_count}/"
                                    f"{speaker_config.minimum_samples} | next "
                                    f"{next_phrase_number}/"
                                    f"{speaker_config.minimum_samples}: {next_phrase}"
                                )
                                speech_recognizer.reset_window()
                                _clear_queue(audio_queue)
                                speaker_enrollment_listen_after_ns = (
                                    time.monotonic_ns() + VOICE_PROMPT_GUARD_NS
                                )
                                break
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
                        phase_text = (
                            registration.phase.value if registration else "SETUP"
                        )
                        focus_text = (
                            ROLE_LABELS[registration.focus_role]
                            if registration
                            else registration_target.value
                        )
                        expected_player = focus_text
                        legal_text = "press S after enrollment"
                    else:
                        phase_text = coordinator.phase.value
                        focus_text = _role_label(
                            coordinator.engine.state.button,
                            coordinator.focus_seat,
                        )
                        expected_player = focus_text
                        legal_text = ",".join(
                            action.value for action in coordinator.engine.state.legal_actions
                        ) or "Part A boundary reached"
                    cv2.putText(
                        display,
                        f"{args.player_mode} | {phase_text} | role {focus_text} | gallery {gallery.size}/{required_player_count}",
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
                    if speaker_enrollment_active:
                        accepted_voice_samples = len(speaker_enrollment_samples)
                        phrase_number = min(
                            accepted_voice_samples + 1,
                            speaker_config.minimum_samples,
                        )
                        phrase = VOICE_ENROLLMENT_PHRASES[
                            (phrase_number - 1) % len(VOICE_ENROLLMENT_PHRASES)
                        ]
                        listening_state = (
                            "PROMPT - WAIT"
                            if now_ns < speaker_enrollment_listen_after_ns
                            else "LISTENING"
                        )
                        voice_progress_text = (
                            f"VOICE {listening_state} | accepted "
                            f"{accepted_voice_samples}/{speaker_config.minimum_samples} | "
                            f"phrase {phrase_number} in one breath: {phrase}"
                        )
                    else:
                        voice_progress_text = "VOICE idle | V starts enrollment after face registration"
                    cv2.putText(
                        display,
                        voice_progress_text,
                        (18, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (80, 220, 255),
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
                        (
                            "1 Button | 2 Small Blind | 3 Big Blind | 4 UTG | E face | V voice start/cancel | S start | X clear"
                            if registration is not None and coordinator is None
                            else "E/robot confirm | V voice | S start | C UI confirm | X clear | Q quit"
                        ),
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
                        if registration is not None:
                            if registration.phase is RegistrationPhase.CAPTURING_FACE:
                                registration.reject_face_enrollment()
                            registration.select_role(ROLE_KEYS[key])
                            registration_target = registration.focus_seat
                        else:
                            registration_target = SEAT_KEYS[key]
                        enrollment_active = False
                        enrollment_samples = []
                        cancelled_voice_role = (
                            speaker_enrollment_role
                            if speaker_enrollment_active
                            else None
                        )
                        speaker_enrollment_active = False
                        for sample in speaker_enrollment_samples:
                            sample.fill(0.0)
                        speaker_enrollment_samples = []
                        speaker_enrollment_player_id = None
                        speaker_enrollment_role = None
                        speaker_enrollment_listen_after_ns = 0
                        if speech_recognizer is not None:
                            speech_recognizer.reset_window()
                        if cancelled_voice_role is not None and event_announcer is not None:
                            event_announcer.publish(
                                "voice_enrollment_cancelled",
                                role=cancelled_voice_role,
                            )
                        status_text = (
                            f"registration role {ROLE_LABELS[registration.focus_role]}"
                            if registration is not None
                            else f"registration target {registration_target.value}"
                        )
                        if event_announcer is not None and registration is not None:
                            event_announcer.publish(
                                "registration_focus_changed",
                                role=registration.focus_role.value,
                            )
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
                            if registration is not None:
                                control = laptop_controls.process_key(key, now_ns)
                                assert control is not None
                                registration_outcome = registration.accept_control(control)
                                if not registration_outcome.accepted:
                                    status_text = registration_outcome.reason
                                    continue
                            enrollment_active = True
                            enrollment_samples = []
                            last_enrollment_sample_ns = None
                            registration_player_id = _registration_player_id(
                                registration, target
                            )
                            status_text = (
                                f"ENROLL {ROLE_LABELS[registration.focus_role]}: one face only"
                                if registration is not None
                                else f"ENROLL {target.value}: one face only"
                            )
                            _emit(
                                "enrollment_started",
                                player_id=registration_player_id,
                                seat=target.value,
                                role=(
                                    registration.focus_role.value
                                    if registration is not None
                                    else target.value
                                ),
                                player_mode=args.player_mode,
                            )
                    elif key == ord("v"):
                        target = (
                            registration_target
                            if coordinator is None
                            else coordinator.focus_seat
                        )
                        player_id = (
                            _registered_player(gallery, target)
                            if target is not None
                            else None
                        )
                        if player_id is None and target is not None and registration is None:
                            player_id = PLAYER_BY_SEAT[target]
                        face_registered = (
                            target is not None
                            and any(
                                item["seat"] == target.value
                                for item in gallery.metadata()
                            )
                        )
                        voice_label = (
                            ROLE_LABELS[registration.focus_role]
                            if registration is not None
                            else player_id
                        )
                        if speaker_enrollment_active:
                            cancelled_sample_count = len(speaker_enrollment_samples)
                            for sample in speaker_enrollment_samples:
                                sample.fill(0.0)
                            speaker_enrollment_samples = []
                            speaker_enrollment_active = False
                            cancelled_role = speaker_enrollment_role
                            cancelled_player_id = speaker_enrollment_player_id
                            speaker_enrollment_player_id = None
                            speaker_enrollment_role = None
                            speaker_enrollment_listen_after_ns = 0
                            _clear_queue(audio_queue)
                            speech_recognizer.reset_window()
                            status_text = f"VOICE CANCELLED | {voice_label}"
                            _emit(
                                "speaker_enrollment_cancelled",
                                player_id=cancelled_player_id,
                                accepted_samples=cancelled_sample_count,
                            )
                            if event_announcer is not None:
                                event_announcer.publish(
                                    "voice_enrollment_cancelled",
                                    role=cancelled_role,
                                )
                        elif args.disable_speech or speech_recognizer is None:
                            status_text = "voice enrollment blocked: speech is disabled"
                        elif target is None:
                            status_text = "voice enrollment blocked: no focus seat"
                        elif not args.consent_confirmed:
                            status_text = "voice enrollment blocked: consent required"
                        elif not face_registered:
                            status_text = "enroll the player's face before voice"
                        elif speaker_gallery.is_enrolled(player_id):
                            status_text = f"voice already enrolled for {voice_label}"
                        elif coordinator is not None:
                            status_text = "voice enrollment is setup-only"
                        else:
                            speaker_enrollment_active = True
                            speaker_enrollment_player_id = player_id
                            speaker_enrollment_role = (
                                registration.focus_role.value
                                if registration is not None
                                else "player"
                            )
                            for sample in speaker_enrollment_samples:
                                sample.fill(0.0)
                            speaker_enrollment_samples = []
                            _clear_queue(audio_queue)
                            speech_recognizer.reset_window()
                            speaker_enrollment_listen_after_ns = (
                                now_ns + VOICE_PROMPT_GUARD_NS
                            )
                            first_phrase = VOICE_ENROLLMENT_PHRASES[0]
                            status_text = (
                                f"VOICE STARTED | {voice_label} | phrase 1/"
                                f"{speaker_config.minimum_samples}: {first_phrase}"
                            )
                            _emit(
                                "speaker_enrollment_started",
                                player_id=player_id,
                                seat=target.value,
                                consent_confirmed=True,
                                embeddings_persisted=False,
                                audio_saved=False,
                                required_samples=speaker_config.minimum_samples,
                                minimum_speaker_frames=(
                                    speaker_config.minimum_speaker_frames
                                ),
                                phrase=first_phrase,
                            )
                            if event_announcer is not None:
                                event_announcer.publish(
                                    "voice_enrollment_started",
                                    role=speaker_enrollment_role,
                                )
                    elif key == ord("x"):
                        if (
                            coordinator is not None
                            and coordinator.phase is not PartAPhase.VERIFYING_IDENTITY
                        ):
                            status_text = "gallery clear blocked while an action window is open"
                        else:
                            if registration is not None:
                                control = laptop_controls.process_key(key, now_ns)
                                assert control is not None
                                registration_outcome = registration.accept_control(control)
                                if not registration_outcome.accepted:
                                    status_text = registration_outcome.reason
                                    continue
                            gallery.clear()
                            speaker_gallery.clear()
                            enrollment_active = False
                            enrollment_samples = []
                            speaker_enrollment_active = False
                            for sample in speaker_enrollment_samples:
                                sample.fill(0.0)
                            speaker_enrollment_samples = []
                            speaker_enrollment_player_id = None
                            speaker_enrollment_role = None
                            speaker_enrollment_listen_after_ns = 0
                            if speech_recognizer is not None:
                                speech_recognizer.reset_window()
                            speech_confirmation.clear()
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
                                registration.button if registration is not None else Seat.A,
                            )
                        )
                        frozen_roster = None
                        if start_block_reason is None and registration is not None:
                            control = laptop_controls.process_key(key, now_ns)
                            assert control is not None
                            registration_outcome = registration.accept_control(control)
                            if not registration_outcome.accepted:
                                start_block_reason = registration_outcome.reason
                            else:
                                frozen_roster = registration_outcome.roster
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
                                engine,
                                args.session_id,
                                require_actor_binding=True,
                                require_visual_settle=True,
                            )
                            if args.player_mode in {
                                "single_player_pilot",
                                "two_player_pilot",
                            }:
                                pilot_remaining = set(enrolled_seats)
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
                            if event_announcer is not None:
                                event_announcer.publish(
                                    "blind_posted",
                                    role=TableRole.SMALL_BLIND.value,
                                    amount_units=engine.rules.small_blind_units,
                                )
                                event_announcer.publish(
                                    "blind_posted",
                                    role=TableRole.BIG_BLIND.value,
                                    amount_units=engine.rules.big_blind_units,
                                )
                                event_announcer.publish(
                                    "turn_started",
                                    role=role_for_seat(
                                        engine.state.button,
                                        coordinator.focus_seat,  # type: ignore[arg-type]
                                    ).value,
                                )
                            enrollment_active = False
                            speaker_enrollment_active = False
                            for sample in speaker_enrollment_samples:
                                sample.fill(0.0)
                            speaker_enrollment_samples = []
                            speaker_enrollment_player_id = None
                            speaker_enrollment_role = None
                            speaker_enrollment_listen_after_ns = 0
                            if speech_recognizer is not None:
                                speech_recognizer.reset_window()
                            status_text = (
                                f"{start_mode}: first "
                                f"{_role_label(button, coordinator.focus_seat)}"
                            )
                            _emit(
                                "hand_started",
                                mode=start_mode,
                                button=button.value,
                                button_role=TableRole.BUTTON.value,
                                first_acting_seat=coordinator.focus_seat.value,
                                first_acting_role=role_for_seat(
                                    button, coordinator.focus_seat  # type: ignore[arg-type]
                                ).value,
                                roster_version=(
                                    frozen_roster.roster_version
                                    if frozen_roster is not None
                                    else None
                                ),
                                enrolled_seats=sorted(
                                    seat.value for seat in enrolled_seats
                                ),
                            )
    except (CameraError, OSError, RuntimeError, ValueError) as exc:
        _emit("error", error=f"{type(exc).__name__}: {exc}")
        return 2
    finally:
        for sample in speaker_enrollment_samples:
            sample.fill(0.0)
        speaker_enrollment_samples = []
        close_announcer = getattr(announcer_port, "close", None)
        if close_announcer is not None:
            close_announcer()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    _emit(
        "summary",
        status="completed" if frames else "no_readable_frames",
        frames=frames,
        missing_reads=missing_reads,
        dropped_camera_frames=dropped_camera_frames,
        camera_reconnects=camera_reconnects,
        elapsed_seconds=elapsed_s,
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        identity_matches=identity_matches,
        simulated_rotation_acks=simulated_rotation_acks,
        dropped_audio_blocks=dropped_audio_blocks,
        final_phase=coordinator.phase.value if coordinator else "setup",
        player_mode=args.player_mode,
        frames_saved=0,
        audio_saved=False,
        embeddings_persisted=False,
        physical_robot_connected=False,
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
