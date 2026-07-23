"""Run one profile through preflight, camera smoke, replay or live mode."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from poker_dealer.runtime.live_hand_app import LiveHandApplication
from poker_dealer.runtime.profile import RuntimeProfile
from poker_dealer.domain import Seat
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.runtime import (
    DiagnosticRun,
    HandRuntimeLoop,
    LiveSessionOperatorUI,
    RecordedReplaySources,
    RuntimeEventLog,
    RuntimeEventWriter,
    ScriptedReplaySources,
    StepClock,
    SessionRuntime,
    SessionEventLog,
    SessionEventWriter,
    SessionOperatorController,
    SessionOperatorSignal,
    check_runtime_hand_log,
    check_session_log,
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
    parser.add_argument(
        "--registration-timeout-seconds",
        type=float,
        default=900.0,
        help="registration-only deadline; independent of smoke/live step duration",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--session-id")
    parser.add_argument("--hand-id")
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="maximum hands in this session; replay is suitable for 20-hand qualification",
    )
    parser.add_argument("--button", choices=tuple(seat.value for seat in Seat))
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--session-log-jsonl", type=Path)
    parser.add_argument("--operator-id", default="laptop-operator")
    parser.add_argument("--rebuy-to-units", type=int)
    parser.add_argument(
        "--session-decision-timeout-seconds", type=float, default=900.0
    )
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
    parser.add_argument("--card-geometry-config", type=Path)
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
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="write one bounded startup-to-shutdown diagnostics bundle under runs/",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="new diagnostics directory under runs/; implies --diagnostics",
    )
    parser.add_argument(
        "--diagnostics-max-records",
        type=int,
        default=100_000,
        help="maximum records in each diagnostics JSONL stream",
    )
    parser.add_argument(
        "--diagnostics-max-mib",
        type=int,
        default=32,
        help="maximum MiB in each diagnostics JSONL stream",
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
    if args.disable_speech or args.speech_device is not None:
        profile = profile.with_speech_override(
            enabled=not args.disable_speech,
            device=args.speech_device,
        )
    return profile


def _selected_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return str(args.mode)
    return "camera-smoke" if args.camera_smoke_frames is not None else "preflight"


def _create_diagnostics(
    args: argparse.Namespace, mode: str
) -> DiagnosticRun | None:
    if not args.diagnostics and args.diagnostics_dir is None:
        return None
    if args.diagnostics_max_records <= 0 or args.diagnostics_max_mib <= 0:
        raise ValueError("diagnostics bounds must be positive")
    profile_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(args.profile).stem)
    base_id = args.session_id or mode
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{base_id}-{generated}")
    runs_root = (ROOT / "runs").resolve()
    if args.diagnostics_dir is None:
        root = runs_root / "diagnostics" / profile_label / run_id
    else:
        requested = args.diagnostics_dir
        root = (requested if requested.is_absolute() else ROOT / requested).resolve()
        try:
            root.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError("--diagnostics-dir must resolve inside runs/") from exc
        run_id = root.name
    diagnostics = DiagnosticRun(
        root,
        run_id=run_id,
        profile_id=profile_label,
        mode=mode,
        invocation=vars(args),
        max_records=args.diagnostics_max_records,
        max_bytes_per_stream=args.diagnostics_max_mib * 1024 * 1024,
    )
    diagnostics.add_config("runtime_profile", _profile_path(args.profile))
    diagnostics.add_config("core_game", ROOT / "configs" / "game" / "core_v1.json")
    return diagnostics


def _run_with_error_boundary(
    args: argparse.Namespace,
    mode: str,
    diagnostics: DiagnosticRun | None,
) -> int:
    try:
        if args.max_hands <= 0:
            raise ValueError("--max-hands must be positive")
        if args.session_decision_timeout_seconds <= 0:
            raise ValueError("--session-decision-timeout-seconds must be positive")
        if args.max_hands > 1 and args.log_jsonl is not None:
            raise ValueError("--log-jsonl is single-hand only; use the profile log root")
        if args.max_hands > 1 and args.replay_log is not None:
            raise ValueError("exact --replay-log supports one hand only")
        if diagnostics is not None and (
            args.log_jsonl is not None or args.session_log_jsonl is not None
        ):
            raise ValueError(
                "diagnostics owns the bundled hand/session paths; omit explicit log paths"
            )
        operation = (
            diagnostics.operation("profile_load") if diagnostics else nullcontext()
        )
        with operation:
            profile = _load_profile(args)
        app = LiveHandApplication(ROOT, profile)
        preflight = preflight_dict = app.preflight().to_dict()
        if diagnostics is not None:
            diagnostics.emit("preflight_completed", preflight_dict)
            _register_perception_configs(diagnostics, args, profile)
        if args.check_config and mode != "preflight":
            raise ValueError("--check-config conflicts with the selected --mode")
        if args.camera_smoke_frames is not None and mode != "camera-smoke":
            raise ValueError("--camera-smoke-frames requires camera-smoke mode")
        if mode == "preflight":
            output = {
                "type": "runtime_preflight",
                **preflight_dict,
                **_diagnostics_output(diagnostics),
            }
            print(json.dumps(output))
            if diagnostics is not None:
                diagnostics.emit("runtime_result", output)
            return 0 if bool(preflight["ready"]) else 2
        if not bool(preflight["ready"]):
            output = {
                "type": "runtime_preflight",
                **preflight_dict,
                **_diagnostics_output(diagnostics),
            }
            print(json.dumps(output))
            if diagnostics is not None:
                diagnostics.emit("runtime_result", output, level="error")
            return 2
        if mode == "replay":
            return _run_replay(args, profile, app, diagnostics)
        if mode == "live-preflight":
            return _run_live_preflight(args, profile, diagnostics)
        if mode == "live":
            return _run_live(args, profile, app, diagnostics)
        try:
            if diagnostics is not None:
                diagnostics.emit("device_open_started", {"camera": True})
            app.open(open_camera=True)
            result = app.camera_smoke(
                requested_frames=args.camera_smoke_frames or 30,
                max_seconds=args.max_seconds,
            )
            output = {
                "type": "camera_smoke",
                **result.to_dict(),
                **_diagnostics_output(diagnostics),
            }
            print(json.dumps(output))
            if diagnostics is not None:
                diagnostics.emit(
                    "camera_smoke_result",
                    output,
                    level="info" if result.passed else "error",
                )
            return 0 if result.passed else 3
        finally:
            app.close()
    except (OSError, ValueError, RuntimeError) as exc:
        if diagnostics is not None:
            diagnostics.record_exception(exc, context={"mode": mode})
        print(
            json.dumps(
                {
                    "type": "runtime_error",
                    "error": type(exc).__name__,
                    "reason": str(exc),
                    **_diagnostics_output(diagnostics),
                }
            ),
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        mode = _selected_mode(args)
        diagnostics = _create_diagnostics(args, mode)
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"type": "runtime_error", "error": type(exc).__name__, "reason": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1
    if diagnostics is None:
        return _run_with_error_boundary(args, mode, None)
    exit_code = 1
    try:
        with diagnostics.capture_stdio():
            try:
                exit_code = _run_with_error_boundary(args, mode, diagnostics)
            except BaseException as exc:
                diagnostics.record_exception(exc, context={"mode": mode})
                raise
    finally:
        diagnostics.finish(exit_code)
    return exit_code


def _diagnostics_output(diagnostics: DiagnosticRun | None) -> dict[str, object]:
    return (
        {"diagnostics_path": str(diagnostics.root)}
        if diagnostics is not None
        else {}
    )


def _register_perception_configs(
    diagnostics: DiagnosticRun,
    args: argparse.Namespace,
    profile: RuntimeProfile,
) -> None:
    resolved = profile.perception.resolved(ROOT)
    overrides = {
        "identity_config": args.identity_config,
        "gesture_config": args.gesture_config,
        "speech_config": args.speech_config,
        "speaker_config": args.speaker_config,
        "attribution_config": args.attribution_config,
        "card_config": args.card_config,
        "card_geometry_config": args.card_geometry_config,
    }
    for label, default_path in resolved.items():
        diagnostics.add_config(label, overrides.get(label) or default_path)


def _run_replay(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    app: LiveHandApplication,
    diagnostics: DiagnosticRun | None = None,
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
    roster = default_replay_roster(session_id, button)
    game_session = SessionRuntime(roster, app.game_config)
    session_path = _session_log_path(args, app, session_id, diagnostics)
    if diagnostics is not None:
        diagnostics.register_artifact("session_log", session_path)
        diagnostics.emit(
            "replay_session_started",
            {"session_id": session_id, "max_hands": args.max_hands},
        )
    dealer = SimulatedDealerAdapter(f"replay:{profile.profile_id.value}")
    dealer.open()
    hand_results: list[dict[str, object]] = []
    try:
        with SessionEventWriter(session_path) as session_writer:
            session_writer.sync(game_session.log)
            for index in range(1, args.max_hands + 1):
                current_hand_id = _indexed_hand_id(
                    hand_id, index=index, total=args.max_hands
                )
                runtime = game_session.start_hand(current_hand_id)
                session_writer.sync(game_session.log)
                sources = (
                    RecordedReplaySources(source_log)
                    if source_log is not None
                    else ScriptedReplaySources()
                )
                output_path = _hand_log_path(
                    args,
                    app,
                    session_id=session_id,
                    hand_id=current_hand_id,
                    diagnostics=diagnostics,
                )
                if diagnostics is not None:
                    diagnostics.register_artifact("hand_log", output_path)
                    diagnostics.emit(
                        "hand_started",
                        {
                            "session_id": session_id,
                            "hand_id": current_hand_id,
                            "index": index,
                        },
                    )
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
                        diagnostic_sink=diagnostics,
                    )
                    result = loop.run(max_steps=args.max_steps)
                checked = check_runtime_hand_log(
                    RuntimeEventLog.from_path(output_path)
                )
                hand_results.append(
                    {
                        "hand_id": current_hand_id,
                        "completed": result.completed,
                        "reason": result.reason,
                        "phase": result.hand_phase.value,
                        "steps": result.steps,
                        "state_version": result.state_version,
                        "log_path": str(output_path),
                        "log_check_passed": checked.passed,
                        "log_check_issues": list(checked.issues),
                    }
                )
                if diagnostics is not None:
                    diagnostics.emit(
                        "hand_finished",
                        hand_results[-1],
                        level=(
                            "info"
                            if result.completed and checked.passed
                            else "error"
                        ),
                    )
                if runtime.phase.value not in {"settled", "voided"}:
                    break
                game_session.close_terminal_hand(
                    hand_log_path=str(output_path.resolve()),
                    hand_log_sha256=_sha256_file(output_path),
                    hand_log_check_passed=checked.passed,
                )
                session_writer.sync(game_session.log)
                game_session.confirm_table_cleared(operator_id="simulator")
                session_writer.sync(game_session.log)
                if not result.completed or not checked.passed:
                    break
            if game_session.active_hand is None and game_session.table_cleared:
                game_session.end_session(
                    operator_id="simulator", reason="replay_run_completed"
                )
                session_writer.sync(game_session.log)
    finally:
        dealer.close()
    session_checked = check_session_log(
        SessionEventLog.from_path(session_path), verify_hand_logs=True
    )
    all_hands_passed = (
        len(hand_results) == args.max_hands
        and all(
            bool(item["completed"]) and bool(item["log_check_passed"])
            for item in hand_results
        )
    )
    first = hand_results[0]
    output = {
        "type": "hand_replay" if args.max_hands == 1 else "session_replay",
        "profile_id": profile.profile_id.value,
        "completed": all_hands_passed and session_checked.passed,
        "reason": first["reason"],
        "phase": first["phase"],
        "steps": first["steps"],
        "state_version": first["state_version"],
        "log_path": first["log_path"],
        "log_check_passed": first["log_check_passed"],
        "log_check_issues": first["log_check_issues"],
        "hands": hand_results,
        "hands_completed": len(hand_results),
        "session_log_path": str(session_path),
        "session_log_check_passed": session_checked.passed,
        "session_log_check_issues": list(session_checked.issues),
        "final_button": game_session.button.value,
        "final_stacks": {
            seat.value: game_session.stacks[seat] for seat in Seat
        },
        "physical_motion": False,
        **_diagnostics_output(diagnostics),
    }
    print(json.dumps(output))
    passed = all_hands_passed and session_checked.passed
    if diagnostics is not None:
        diagnostics.emit(
            "runtime_result", output, level="info" if passed else "error"
        )
    return 0 if passed else 4


def _indexed_hand_id(base: str, *, index: int, total: int) -> str:
    if total == 1:
        return base
    return f"{base}-{index:03d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_log_path(
    args: argparse.Namespace,
    app: LiveHandApplication,
    session_id: str,
    diagnostics: DiagnosticRun | None = None,
) -> Path:
    if args.session_log_jsonl is not None:
        return args.session_log_jsonl
    if args.log_jsonl is not None:
        return args.log_jsonl.with_name(f"{args.log_jsonl.stem}.session.jsonl")
    if diagnostics is not None:
        return diagnostics.session_log_path
    return app.session_log_path(session_id=session_id)


def _hand_log_path(
    args: argparse.Namespace,
    app: LiveHandApplication,
    *,
    session_id: str,
    hand_id: str,
    diagnostics: DiagnosticRun | None = None,
) -> Path:
    if args.log_jsonl is not None:
        return args.log_jsonl
    if args.session_log_jsonl is not None:
        return args.session_log_jsonl.parent / f"{hand_id}.jsonl"
    if diagnostics is not None:
        return diagnostics.hand_log_path(hand_id)
    return app.event_log_path(session_id=session_id, hand_id=hand_id)


def _run_live(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    app: LiveHandApplication,
    diagnostics: DiagnosticRun | None = None,
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
    hand_id = args.hand_id or ("hand-001" if args.max_hands == 1 else "hand")
    first_hand_id = _indexed_hand_id(hand_id, index=1, total=args.max_hands)
    first_output_path = _hand_log_path(
        args,
        app,
        session_id=session_id,
        hand_id=first_hand_id,
        diagnostics=diagnostics,
    )
    session_path = _session_log_path(args, app, session_id, diagnostics)
    if diagnostics is not None:
        diagnostics.register_artifact("session_log", session_path)
        diagnostics.register_artifact("hand_log", first_output_path)
        diagnostics.emit(
            "live_session_started",
            {"session_id": session_id, "max_hands": args.max_hands},
        )
    live_config = _live_config(args, profile, consent_confirmed=True)
    app.open(open_camera=True)
    frame_source = InteractiveOpenCVFrameSource(app.camera, display=True)
    controls = LiveKeyboardControlSource(frame_source)
    session = LivePerceptionSession(live_config, frame_source)
    hand_results: list[dict[str, object]] = []
    game_session: SessionRuntime | None = None
    try:
        session.open(session_id)
        writer = RuntimeEventWriter(first_output_path)
        try:
            roster = session.acquire_roster(
                frame_source=frame_source,
                control_source=controls,
                event_sink=writer,
                session_id=session_id,
                button=Seat(args.button),
                deadline_ns=time.monotonic_ns()
                + int(args.registration_timeout_seconds * 1_000_000_000),
            )
            game_session = app.create_session(roster=roster)
            controller = SessionOperatorController(
                game_session,
                operator_id=args.operator_id,
                rebuy_to_units=args.rebuy_to_units,
            )
            boundary_ui = LiveSessionOperatorUI(frame_source, controls)
            with SessionEventWriter(session_path) as session_writer:
                session_writer.sync(game_session.log)
                for index in range(1, args.max_hands + 1):
                    current_hand_id = _indexed_hand_id(
                        hand_id, index=index, total=args.max_hands
                    )
                    output_path = (
                        first_output_path
                        if index == 1
                        else _hand_log_path(
                            args,
                            app,
                            session_id=session_id,
                            hand_id=current_hand_id,
                            diagnostics=diagnostics,
                        )
                    )
                    if diagnostics is not None:
                        diagnostics.register_artifact("hand_log", output_path)
                        diagnostics.emit(
                            "hand_started",
                            {
                                "session_id": session_id,
                                "hand_id": current_hand_id,
                                "index": index,
                            },
                        )
                    if index > 1:
                        writer = RuntimeEventWriter(output_path)
                    runtime = game_session.start_hand(current_hand_id)
                    session_writer.sync(game_session.log)
                    while True:
                        loop = HandRuntimeLoop(
                            runtime,
                            app.dealer,
                            identity_source=session,
                            action_source=session,
                            card_source=session,
                            visual_settle_source=session,
                            control_source=controls,
                            frame_source=frame_source,
                            event_writer=writer,
                            diagnostic_sink=diagnostics,
                        )
                        result = loop.run(max_steps=args.max_steps)
                        if runtime.phase.value != "paused_recovery":
                            break
                        recovery = boundary_ui.wait_for_decision(
                            game_session,
                            controller,
                            timeout_seconds=args.session_decision_timeout_seconds,
                        )
                        session_writer.sync(game_session.log)
                        if recovery.signal is SessionOperatorSignal.RETRY_HAND:
                            continue
                        if recovery.signal is SessionOperatorSignal.HAND_VOIDED:
                            break
                    writer.sync_engine(runtime.engine.log)
                    writer.close()
                    checked = check_runtime_hand_log(
                        RuntimeEventLog.from_path(output_path),
                        allow_voided=runtime.phase.value == "voided",
                    )
                    game_session.close_terminal_hand(
                        hand_log_path=str(output_path.resolve()),
                        hand_log_sha256=_sha256_file(output_path),
                        hand_log_check_passed=checked.passed,
                    )
                    session_writer.sync(game_session.log)
                    hand_results.append(
                        {
                            "hand_id": current_hand_id,
                            "completed": runtime.phase.value == "settled",
                            "reason": result.reason,
                            "phase": runtime.phase.value,
                            "steps": result.steps,
                            "log_path": str(output_path),
                            "log_check_passed": checked.passed,
                            "log_check_issues": list(checked.issues),
                        }
                    )
                    if diagnostics is not None:
                        diagnostics.emit(
                            "hand_finished",
                            hand_results[-1],
                            level=(
                                "info"
                                if hand_results[-1]["completed"] and checked.passed
                                else "error"
                            ),
                        )
                    boundary = boundary_ui.wait_for_decision(
                        game_session,
                        controller,
                        timeout_seconds=args.session_decision_timeout_seconds,
                        stop_after_clear=index >= args.max_hands,
                    )
                    session_writer.sync(game_session.log)
                    if boundary.signal is SessionOperatorSignal.SESSION_ENDED:
                        break
                    if boundary.signal is not SessionOperatorSignal.START_NEXT_HAND:
                        raise RuntimeError(
                            f"unexpected session boundary signal: {boundary.signal.value}"
                        )
                if not game_session.ended:
                    game_session.end_session(
                        operator_id=args.operator_id,
                        reason="configured_hand_limit_reached",
                    )
                    session_writer.sync(game_session.log)
        finally:
            writer.close()
    finally:
        session.close()
        frame_source.close()
        app.close()
    assert game_session is not None
    session_checked = check_session_log(
        SessionEventLog.from_path(session_path), verify_hand_logs=True
    )
    all_hands_checked = bool(hand_results) and all(
        bool(item["log_check_passed"]) for item in hand_results
    )
    first = hand_results[0]
    output = {
        "type": "live_hand" if args.max_hands == 1 else "live_session",
        "profile_id": profile.profile_id.value,
        "completed": all_hands_checked and session_checked.passed,
        "reason": first["reason"],
        "phase": first["phase"],
        "steps": first["steps"],
        "log_path": first["log_path"],
        "log_check_passed": first["log_check_passed"],
        "hands": hand_results,
        "hands_completed": len(hand_results),
        "session_log_path": str(session_path),
        "session_log_check_passed": session_checked.passed,
        "session_log_check_issues": list(session_checked.issues),
        "final_button": game_session.button.value,
        "final_stacks": {
            seat.value: game_session.stacks[seat] for seat in Seat
        },
        "physical_motion": False,
        "dealer_adapter": "simulated",
        "face_down_evidence": "development_operator_confirmation",
        "card_gate_2b_passed": False,
        **_diagnostics_output(diagnostics),
    }
    print(json.dumps(output, ensure_ascii=False))
    passed = all_hands_checked and session_checked.passed
    if diagnostics is not None:
        diagnostics.emit(
            "runtime_result", output, level="info" if passed else "error"
        )
    return 0 if passed else 4


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
        card_geometry_config=(
            args.card_geometry_config
            or perception_paths["card_geometry_config"]
        ),
        consent_confirmed=consent_confirmed,
        speech_enabled=profile.speech_enabled,
        speech_device=(
            args.speech_device
            if args.speech_device is not None
            else profile.speech_device
        ),
        runtime_calibration_id=profile.perception.calibration_id,
        target_geometry_validated=(
            profile.perception.target_geometry_validated
        ),
        operator_face_down_confirmation=args.development_operator_face_down,
    )


def _run_live_preflight(
    args: argparse.Namespace,
    profile: RuntimeProfile,
    diagnostics: DiagnosticRun | None = None,
) -> int:
    report = validate_live_perception_assets(
        _live_config(args, profile, consent_confirmed=False)
    )
    output = {
        "type": "live_perception_preflight",
        "profile_id": profile.profile_id.value,
        "assets_valid": True,
        "target_geometry_validated": profile.perception.target_geometry_validated,
        "full_live_hand_integrated": False,
        "development_live_available": True,
        **report,
        **_diagnostics_output(diagnostics),
    }
    print(json.dumps(output, ensure_ascii=False))
    if diagnostics is not None:
        diagnostics.emit("live_perception_preflight_completed", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
