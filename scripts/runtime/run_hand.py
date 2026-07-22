"""Run one profile through preflight, camera smoke, replay or live mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from poker_dealer.runtime.live_hand_app import LiveHandApplication
from poker_dealer.runtime.profile import RuntimeProfile
from poker_dealer.domain import Seat
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.runtime import (
    HandRuntime,
    HandRuntimeLoop,
    RecordedReplaySources,
    RuntimeEventLog,
    RuntimeEventWriter,
    ScriptedReplaySources,
    StepClock,
    check_runtime_hand_log,
    default_replay_roster,
)
from poker_dealer.runtime.live_perception import (
    InteractiveOpenCVFrameSource,
    LiveKeyboardControlSource,
    LivePerceptionConfig,
    LivePerceptionSession,
    validate_live_perception_assets,
)


ROOT = Path(__file__).resolve().parents[2]
NAMED_PROFILES = {name: ROOT / "configs" / "runtime" / f"{name}.json" for name in (
    "laptop",
    "robot_camera",
    "robot_hardware",
)}


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="laptop, robot_camera, robot_hardware, or a runtime-profile JSON path",
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "live-preflight", "camera-smoke", "replay", "live"),
        help="runtime mode; legacy preflight/smoke flags remain accepted",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-config",
        action="store_true",
        help="validate dependency selection without opening live devices (default)",
    )
    mode.add_argument(
        "--camera-smoke-frames",
        type=int,
        metavar="N",
        help="open the selected camera and read at most N frames",
    )
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument("--camera-index", type=int)
    camera.add_argument("--stream-url")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--session-id")
    parser.add_argument("--hand-id")
    parser.add_argument("--button", choices=tuple(seat.value for seat in Seat))
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument("--speech-device", type=_device)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--development-operator-face-down",
        action="store_true",
        help=(
            "allow F-key confirmation for hole-card back/occupancy; development "
            "only and cannot pass the card-perception gate"
        ),
    )
    parser.add_argument(
        "--identity-config",
        type=Path,
    )
    parser.add_argument(
        "--gesture-config",
        type=Path,
    )
    parser.add_argument(
        "--speech-config",
        type=Path,
    )
    parser.add_argument("--speaker-config", type=Path)
    parser.add_argument(
        "--attribution-config",
        type=Path,
    )
    parser.add_argument(
        "--card-config",
        type=Path,
    )
    parser.add_argument(
        "--replay-log",
        type=Path,
        help="exact prior runtime JSONL evidence to replay; omit for built-in fixture",
    )
    parser.add_argument(
        "--disable-speech",
        action="store_true",
        help="do not reserve a microphone for this invocation",
    )
    return parser.parse_args(argv)


def _profile_path(value: str) -> Path:
    if value in NAMED_PROFILES:
        return NAMED_PROFILES[value]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_profile(args: argparse.Namespace) -> RuntimeProfile:
    profile = RuntimeProfile.from_json(_profile_path(args.profile))
    profile = profile.with_camera_override(
        device_index=args.camera_index,
        stream_url=args.stream_url,
    )
    if args.disable_speech:
        profile = profile.with_speech_override(enabled=False)
    return profile


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        profile = _load_profile(args)
        app = LiveHandApplication(ROOT, profile)
        preflight = preflight_dict = app.preflight().to_dict()
        mode = args.mode
        if mode is None:
            mode = "camera-smoke" if args.camera_smoke_frames is not None else "preflight"
        if args.check_config and mode != "preflight":
            raise ValueError("--check-config conflicts with the selected --mode")
        if args.camera_smoke_frames is not None and mode != "camera-smoke":
            raise ValueError("--camera-smoke-frames requires camera-smoke mode")
        if mode == "preflight":
            print(json.dumps({"type": "runtime_preflight", **preflight_dict}))
            return 0 if bool(preflight["ready"]) else 2
        if not bool(preflight["ready"]):
            print(json.dumps({"type": "runtime_preflight", **preflight_dict}))
            return 2
        if mode == "replay":
            return _run_replay(args, profile, app)
        if mode == "live-preflight":
            return _run_live_preflight(args, profile)
        if mode == "live":
            return _run_live(args, profile, app)
        try:
            app.open(open_camera=True)
            result = app.camera_smoke(
                requested_frames=args.camera_smoke_frames or 30,
                max_seconds=args.max_seconds,
            )
            print(json.dumps({"type": "camera_smoke", **result.to_dict()}))
            return 0 if result.passed else 3
        finally:
            app.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"type": "runtime_error", "error": type(exc).__name__, "reason": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1


def _run_replay(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    app: LiveHandApplication,
) -> int:
    if profile.dealer.physical_motion:
        raise RuntimeError("replay never opens a physical dealer")
    source_log = (
        RuntimeEventLog.from_path(args.replay_log) if args.replay_log else None
    )
    if source_log is None:
        session_id = args.session_id or "replay-session"
        hand_id = args.hand_id or "replay-hand"
        button = Seat(args.button or Seat.A.value)
        sources = ScriptedReplaySources()
    else:
        source_state = source_log.engine_log().recover_state()
        identity_records = source_log.evidence("face_identity_observation")
        if not identity_records:
            raise ValueError("replay log has no identity observations")
        source_session_id = str(identity_records[0].payload["session_id"])
        if args.session_id is not None and args.session_id != source_session_id:
            raise ValueError(
                "exact replay session_id must match the source evidence log"
            )
        if args.hand_id is not None and args.hand_id != source_state.hand_id:
            raise ValueError("exact replay hand_id must match the source evidence log")
        if args.button is not None and Seat(args.button) is not source_state.button:
            raise ValueError("exact replay button must match the source evidence log")
        session_id = source_session_id
        hand_id = source_state.hand_id
        button = source_state.button
        sources = RecordedReplaySources(source_log)
    roster = default_replay_roster(session_id, button)
    runtime = HandRuntime.from_roster(
        hand_id=hand_id,
        roster=roster,
        require_actor_binding=True,
        require_visual_settle=True,
    )
    output_path = args.log_jsonl or app.event_log_path(
        session_id=session_id, hand_id=hand_id
    )
    dealer = SimulatedDealerAdapter(f"replay:{profile.profile_id.value}")
    dealer.open()
    try:
        with RuntimeEventWriter(output_path) as writer:
            loop = HandRuntimeLoop(
                runtime,
                dealer,
                identity_source=sources,
                action_source=sources,
                card_source=sources,
                visual_settle_source=sources,
                event_writer=writer,
                clock_ns=StepClock(),
            )
            result = loop.run(max_steps=args.max_steps)
    finally:
        dealer.close()
    checked = check_runtime_hand_log(RuntimeEventLog.from_path(output_path))
    print(
        json.dumps(
            {
                "type": "hand_replay",
                "profile_id": profile.profile_id.value,
                "completed": result.completed,
                "reason": result.reason,
                "phase": result.hand_phase.value,
                "steps": result.steps,
                "state_version": result.state_version,
                "log_path": str(output_path),
                "log_check_passed": checked.passed,
                "log_check_issues": list(checked.issues),
                "physical_motion": False,
            }
        )
    )
    return 0 if result.completed and checked.passed else 4


def _run_live(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    app: LiveHandApplication,
) -> int:
    if profile.dealer.physical_motion:
        raise RuntimeError("live hardware mode remains Robotics-gated and unavailable")
    if args.headless:
        raise ValueError("interactive four-player registration requires the live UI")
    if not args.consent_confirmed:
        raise PermissionError("--consent-confirmed is required for face enrollment")
    if not args.development_operator_face_down:
        raise RuntimeError(
            "the face-down occupancy/orientation model is not admitted; pass "
            "--development-operator-face-down only for an explicitly non-Gate run"
        )
    if args.button is None:
        raise ValueError("--button is required for four-player live mode")
    session_id = args.session_id or f"live-{profile.profile_id.value}"
    hand_id = args.hand_id or "hand-001"
    output_path = args.log_jsonl or app.event_log_path(
        session_id=session_id, hand_id=hand_id
    )
    live_config = _live_config(args, profile, consent_confirmed=True)
    app.open(open_camera=True)
    frame_source = InteractiveOpenCVFrameSource(app.camera, display=True)
    controls = LiveKeyboardControlSource(frame_source)
    session = LivePerceptionSession(live_config, frame_source)
    try:
        session.open(session_id)
        with RuntimeEventWriter(output_path) as writer:
            roster = session.acquire_roster(
                frame_source=frame_source,
                control_source=controls,
                event_sink=writer,
                session_id=session_id,
                button=Seat(args.button),
                deadline_ns=time.monotonic_ns() + int(args.max_seconds * 1_000_000_000),
            )
            runtime = app.create_hand(hand_id=hand_id, roster=roster)
            loop = HandRuntimeLoop(
                runtime,
                app.dealer,
                identity_source=session,
                action_source=session,
                card_source=session,
                visual_settle_source=session,
                frame_source=frame_source,
                event_writer=writer,
            )
            result = loop.run(max_steps=args.max_steps)
    finally:
        session.close()
        frame_source.close()
        app.close()
    checked = check_runtime_hand_log(RuntimeEventLog.from_path(output_path))
    print(
        json.dumps(
            {
                "type": "live_hand",
                "profile_id": profile.profile_id.value,
                "completed": result.completed,
                "reason": result.reason,
                "phase": result.hand_phase.value,
                "steps": result.steps,
                "log_path": str(output_path),
                "log_check_passed": checked.passed,
                "physical_motion": False,
                "dealer_adapter": "simulated",
                "face_down_evidence": "development_operator_confirmation",
                "card_gate_2b_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.completed and checked.passed else 4


def _live_config(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    *,
    consent_confirmed: bool,
) -> LivePerceptionConfig:
    perception_paths = profile.perception.resolved(ROOT)
    return LivePerceptionConfig(
        identity_config=args.identity_config or perception_paths["identity_config"],
        gesture_config=args.gesture_config or perception_paths["gesture_config"],
        speech_config=args.speech_config or perception_paths["speech_config"],
        speaker_config=args.speaker_config or perception_paths["speaker_config"],
        attribution_config=(
            args.attribution_config or perception_paths["attribution_config"]
        ),
        card_config=args.card_config or perception_paths["card_config"],
        consent_confirmed=consent_confirmed,
        speech_enabled=profile.speech_enabled,
        speech_device=(
            args.speech_device
            if args.speech_device is not None
            else profile.speech_device
        ),
        runtime_calibration_id=profile.perception.calibration_id,
        operator_face_down_confirmation=args.development_operator_face_down,
    )


def _run_live_preflight(
    args: argparse.Namespace,
    profile: RuntimeProfile,
) -> int:
    report = validate_live_perception_assets(
        _live_config(args, profile, consent_confirmed=False)
    )
    print(
        json.dumps(
            {
                "type": "live_perception_preflight",
                "profile_id": profile.profile_id.value,
                "assets_valid": True,
                "target_geometry_validated": (
                    profile.perception.target_geometry_validated
                ),
                "full_live_hand_integrated": False,
                "development_live_available": True,
                **report,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
