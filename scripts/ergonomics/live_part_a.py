"""Run calibrated, non-recording Part A states on a laptop camera."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2

from deskmate_advance.features.ergonomics import (
    AudioLevelPoller,
    AudioLevelSnapshot,
    FaceFeatureExtractor,
    LiveScheduleConfig,
    LiveSnapshot,
    PartALiveEngine,
    PoseFeatureConfig,
    PoseFeatureExtractor,
)
from deskmate_advance.features.ergonomics.benchmark import sha256_file
from deskmate_advance.perception.audio import MicrophoneConfig
from deskmate_advance.perception.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)
from deskmate_advance.perception.ergonomics import (
    FaceLandmarkerAdapter,
    FaceLandmarkerConfig,
    ObservationState,
    PoseLandmarkerAdapter,
    PoseLandmarkerConfig,
)
from deskmate_advance.temporal.ergonomics import (
    CalibrationCollector,
    CalibrationProgress,
    CalibrationState,
    ErgonomicsEventConfig,
    ErgonomicsRuleEngine,
    ErgonomicsRuleSnapshot,
    SemanticState,
)


_POSE_CONNECTIONS = (
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ergonomics/perception.json"),
    )
    parser.add_argument(
        "--event-config",
        type=Path,
        default=Path("configs/ergonomics/events.json"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pose-model", choices=("full", "lite"), default="full")
    parser.add_argument("--pose-hz", type=float)
    parser.add_argument("--face-hz", type=float)
    parser.add_argument("--enable-audio", action="store_true")
    parser.add_argument("--audio-device-index", type=int, default=1)
    parser.add_argument("--audio-sample-rate", type=int, default=16_000)
    parser.add_argument("--audio-block-ms", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.camera_index < 0:
        parser.error("--camera-index must be non-negative")
    if args.audio_device_index < 0:
        parser.error("--audio-device-index must be non-negative")
    if args.audio_sample_rate <= 0 or args.audio_block_ms <= 0:
        parser.error("audio sample rate and block duration must be positive")
    if args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("camera width, height and FPS must be positive")
    if args.pose_hz is not None and (
        not math.isfinite(args.pose_hz) or not 0 < args.pose_hz <= 1000
    ):
        parser.error("--pose-hz must be positive")
    if args.face_hz is not None and (
        not math.isfinite(args.face_hz) or not 0 < args.face_hz <= 1000
    ):
        parser.error("--face-hz must be positive")
    if (
        args.max_frames < 0
        or not math.isfinite(args.duration_seconds)
        or args.duration_seconds < 0
    ):
        parser.error("bounds must be non-negative")
    if args.headless and args.max_frames == 0 and args.duration_seconds == 0:
        parser.error("headless mode requires --max-frames or --duration-seconds")
    return args


def _verified_asset(
    project_root: Path,
    relative: str,
    expected_sha256: str,
) -> Path:
    path = (project_root / relative).resolve()
    if project_root not in path.parents or not path.is_file():
        raise ValueError(f"invalid model asset: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"model asset SHA-256 mismatch: {path}")
    return path


def _value(value: float | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _state_text(state: ObservationState | None, stale: bool) -> str:
    if state is None:
        return "not-run"
    return f"{state.value}{'/stale' if stale else ''}"


def _semantic_text(rules: ErgonomicsRuleSnapshot, event_name: str) -> str:
    item = rules.evaluation(event_name)
    return f"{item.semantic_state.value}/{item.phase.value}"


def _calibration_line(progress: CalibrationProgress) -> str:
    counts = progress.counts
    if progress.state is CalibrationState.COLLECTING:
        return (
            f"CALIBRATING {progress.duration_fraction * 100:.0f}% | "
            "sit normally: both shoulders visible, neutral head/distance, eyes open | "
            f"pose={counts.pose} face={counts.face} light={counts.luminance}"
        )
    if progress.state is CalibrationState.READY:
        return (
            "calibration=ready | "
            f"pose={counts.pose} face={counts.face} light={counts.luminance}"
        )
    reasons = ",".join(progress.not_ready_reasons)
    hint = (
        " | reframe: keep both shoulders visible"
        if any(reason.startswith("pose_samples:0/") for reason in progress.not_ready_reasons)
        else ""
    )
    return f"calibration=NOT_READY ({reasons}){hint} | C: retry"


def _audio_line(
    audio: AudioLevelSnapshot | None,
    rules: ErgonomicsRuleSnapshot,
) -> str:
    if audio is None:
        return "audio=disabled (use --enable-audio); noise state remains unknown"
    dbfs = audio.observation.dbfs if audio.observation is not None else None
    noise = rules.evaluation("noise_too_high")
    usable = noise.reason is None
    return (
        f"audio_device={audio.status.value} evidence={'usable' if usable else 'unusable'} "
        f"dBFS={_value(dbfs, digits=1)} "
        f"reason={audio.error or '-'}"
    )


def _draw_pose(display, snapshot: LiveSnapshot) -> None:
    observation = snapshot.pose_observation
    features = snapshot.pose_features
    if (
        observation is None
        or features is None
        or observation.state is not ObservationState.VALID
        or snapshot.pose_stale
    ):
        return
    points: dict[int, tuple[int, int]] = {}
    for index, landmark in enumerate(observation.landmarks):
        if features.missing_mask[index]:
            continue
        x = round(landmark.x * snapshot.frame.width)
        y = round(landmark.y * snapshot.frame.height)
        if 0 <= x < snapshot.frame.width and 0 <= y < snapshot.frame.height:
            points[index] = (x, y)
            cv2.circle(display, (x, y), 3, (40, 220, 40), -1, cv2.LINE_AA)
    for start, end in _POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(display, points[start], points[end], (40, 220, 40), 2, cv2.LINE_AA)


def _draw_face_box(display, snapshot: LiveSnapshot) -> None:
    features = snapshot.face_features
    if (
        features is None
        or features.geometry_state is not ObservationState.VALID
        or features.face_center_xy is None
        or features.face_bbox_width_ratio is None
        or features.face_bbox_height_ratio is None
        or snapshot.face_stale
    ):
        return
    center_x, center_y = features.face_center_xy
    half_width = features.face_bbox_width_ratio / 2
    half_height = features.face_bbox_height_ratio / 2
    left = round((center_x - half_width) * snapshot.frame.width)
    right = round((center_x + half_width) * snapshot.frame.width)
    top = round((center_y - half_height) * snapshot.frame.height)
    bottom = round((center_y + half_height) * snapshot.frame.height)
    left = min(max(left, 0), snapshot.frame.width - 1)
    right = min(max(right, 0), snapshot.frame.width - 1)
    top = min(max(top, 0), snapshot.frame.height - 1)
    bottom = min(max(bottom, 0), snapshot.frame.height - 1)
    cv2.rectangle(display, (left, top), (right, bottom), (255, 180, 30), 2)


def _overlay_lines(
    snapshot: LiveSnapshot,
    summary: dict[str, Any],
    pose_model: str,
    calibration: CalibrationProgress,
    rules: ErgonomicsRuleSnapshot,
    audio: AudioLevelSnapshot | None,
) -> list[str]:
    pose = snapshot.pose_observation
    pose_features = snapshot.pose_features
    face = snapshot.face_observation
    face_features = snapshot.face_features
    pose_display_features = None if snapshot.pose_stale else pose_features
    face_display_features = None if snapshot.face_stale else face_features
    pose_p95 = summary["pose"]["latency_ms"]["p95"]
    face_p95 = summary["face"]["latency_ms"]["p95"]
    pose_missing_rate = summary["pose"]["state_rates"]["missing"]
    face_missing_rate = summary["face"]["state_rates"]["missing"]
    lines = [
        "DeskMate Part A A3 | DEVELOPMENT THRESHOLDS | NO RECORDING",
        _calibration_line(calibration),
        (
            f"camera seq={snapshot.frame.sequence_id} "
            f"{snapshot.frame.width}x{snapshot.frame.height} "
            f"capture={_value(summary['effective_capture_fps'], digits=1)} fps"
        ),
        (
            f"pose[{pose_model}]={_state_text(pose.state if pose else None, snapshot.pose_stale)} "
            f"age={_value(snapshot.pose_age_ms, digits=0)}ms "
            f"lat={_value(pose.context.inference_ms if pose else None)}ms "
            f"p95={_value(pose_p95)}ms miss={_value(pose_missing_rate)} "
            f"reason={pose.reason if pose and pose.reason else '-'}"
        ),
        (
            "  shoulder="
            f"{_value(pose_display_features.shoulder_tilt_deg if pose_display_features else None)}deg "
            "torso="
            f"{_value(pose_display_features.torso_lean_from_vertical_deg if pose_display_features else None)}deg "
            "motion="
            f"{_value(pose_display_features.upper_body_motion_per_second if pose_display_features else None)} body/s"
        ),
        (
            f"face={_state_text(face.state if face else None, snapshot.face_stale)} "
            f"age={_value(snapshot.face_age_ms, digits=0)}ms "
            f"lat={_value(face.context.inference_ms if face else None)}ms "
            f"p95={_value(face_p95)}ms miss={_value(face_missing_rate)} "
            f"reason={face.reason if face and face.reason else '-'}"
        ),
        (
            "  area_proxy="
            f"{_value(face_display_features.face_bbox_area_ratio if face_display_features else None, digits=4)} "
            "raw_rotXYZ_uncal="
            f"{tuple(round(v, 1) for v in face_display_features.raw_rotation_xyz_deg) if face_display_features and face_display_features.raw_rotation_xyz_deg else 'n/a'}"
        ),
        (
            "  blink L/R="
            f"{_value(face_display_features.eye_blink_left if face_display_features else None)}/"
            f"{_value(face_display_features.eye_blink_right if face_display_features else None)} "
            f"blink_state={face_display_features.blink_state.value if face_display_features else 'n/a'}"
        ),
        (
            f"luminance mean={_value(snapshot.luminance.mean, digits=1)} "
            f"p10={_value(snapshot.luminance.p10, digits=1)} "
            f"p90={_value(snapshot.luminance.p90, digits=1)} "
            f"age={_value(snapshot.luminance_age_ms, digits=0)}ms"
        ),
        _audio_line(audio, rules),
        (
            "states: "
            f"static={_semantic_text(rules, 'static_too_long')}  "
            f"posture={_semantic_text(rules, 'bad_posture')}  "
            f"distance={_semantic_text(rules, 'screen_too_close')}  "
            f"head={_semantic_text(rules, 'head_off_center')}"
        ),
        (
            "states: "
            f"blink={_semantic_text(rules, 'low_blink_rate')}  "
            f"dark={_semantic_text(rules, 'environment_too_dark')}  "
            f"bright={_semantic_text(rules, 'environment_too_bright')}  "
            f"noise={_semantic_text(rules, 'noise_too_high')}"
        ),
        "Q / Esc: exit | C: restart neutral calibration",
    ]
    return lines


def _render(
    snapshot: LiveSnapshot,
    summary: dict[str, Any],
    pose_model: str,
    calibration: CalibrationProgress,
    rules: ErgonomicsRuleSnapshot,
    audio: AudioLevelSnapshot | None,
):
    display = snapshot.frame.image.copy()
    _draw_pose(display, snapshot)
    _draw_face_box(display, snapshot)
    lines = _overlay_lines(
        snapshot,
        summary,
        pose_model,
        calibration,
        rules,
        audio,
    )
    line_height = 25
    panel_height = line_height * len(lines) + 12
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(display.shape[1], 1260), panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, display, 0.32, 0, display)
    for index, line in enumerate(lines):
        if "warning" in line:
            color = (80, 100, 255)
        elif "CALIBRATING" in line or "NOT_READY" in line:
            color = (80, 220, 255)
        else:
            color = (100, 255, 100) if index == 0 else (235, 235, 235)
        cv2.putText(
            display,
            line,
            (12, 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            1,
            cv2.LINE_AA,
        )
    return display


def _window_closed(title: str) -> bool:
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _calibration_summary(progress: CalibrationProgress | None) -> dict[str, Any]:
    if progress is None:
        return {"state": "not_started", "ready": False}
    return {
        "state": progress.state.value,
        "ready": progress.ready,
        "elapsed_ms": progress.elapsed_ms,
        "duration_ms": progress.duration_ms,
        "counts": {
            "pose": progress.counts.pose,
            "torso_lean": progress.counts.torso_lean,
            "face": progress.counts.face,
            "eye_open": progress.counts.eye_open,
            "luminance": progress.counts.luminance,
        },
        "not_ready_reasons": list(progress.not_ready_reasons),
    }


def _rules_summary(rules: ErgonomicsRuleSnapshot | None) -> dict[str, Any]:
    if rules is None:
        return {}
    return {
        item.event_name: {
            "condition": item.condition.value,
            "semantic_state": item.semantic_state.value,
            "phase": item.phase.value,
            "evidence_elapsed_ms": item.evidence_elapsed_ms,
            "active_duration_ms": item.active_duration_ms,
            "cooldown_remaining_ms": item.cooldown_remaining_ms,
            "reason": item.reason,
            "evidence": item.evidence_dict(),
        }
        for item in rules.evaluations
    }


def _audio_summary(
    enabled: bool,
    audio: AudioLevelSnapshot | None,
) -> dict[str, Any]:
    if audio is None:
        return {
            "enabled": enabled,
            "status": "not_started" if enabled else "disabled",
            "records_audio": False,
        }
    observation = audio.observation
    return {
        "enabled": enabled,
        "status": audio.status.value,
        "observed_at_ns": audio.observed_at_ns,
        "state": observation.state.value if observation is not None else "unknown",
        "rms": observation.rms if observation is not None else None,
        "dbfs": observation.dbfs if observation is not None else None,
        "reason": audio.error,
        "records_audio": False,
    }


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    event_config_path = args.event_config.resolve()
    project_root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    event_config = ErgonomicsEventConfig.load(event_config_path)
    event_config_sha256 = hashlib.sha256(event_config_path.read_bytes()).hexdigest()
    pose_config = config["pose"]
    face_config = config["face"]
    live_config = config["live_probe"]
    pose_prefix = "primary" if args.pose_model == "full" else "fallback"
    pose_asset = _verified_asset(
        project_root,
        pose_config[f"{pose_prefix}_asset"],
        pose_config[f"{pose_prefix}_asset_sha256"],
    )
    face_asset = _verified_asset(
        project_root,
        face_config["asset"],
        face_config["asset_sha256"],
    )
    camera_config = CameraConfig(
        device_index=args.camera_index,
        source_id="laptop_camera_exploratory",
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    schedule = LiveScheduleConfig(
        pose_hz=args.pose_hz or live_config["pose_hz"],
        face_hz=args.face_hz or live_config["face_hz"],
        luminance_hz=live_config["luminance_hz"],
        stale_after_ms=live_config["stale_after_ms"],
        metric_reservoir_size=live_config["metric_reservoir_size"],
    )
    feature_config = config["features"]["pose"]
    calibration_collector = CalibrationCollector(event_config.calibration)
    rule_engine = ErgonomicsRuleEngine(event_config)
    title = "DeskMate Advance - Part A A3 Live (Q/Esc close, C calibrate)"
    camera_statuses: Counter[str] = Counter()
    termination = "unknown"
    wall_started = time.monotonic()
    loop_started: float | None = None
    engine: PartALiveEngine | None = None
    negotiated: dict[str, Any] | None = None
    audio_poller: AudioLevelPoller | None = None
    latest_audio: AudioLevelSnapshot | None = None
    calibration_progress: CalibrationProgress = calibration_collector.progress
    latest_rules: ErgonomicsRuleSnapshot | None = None
    audio_cleanup: dict[str, Any] = {"enabled": args.enable_audio}
    window_cleanup_error: str | None = None
    runtime_error: str | None = None
    try:
        with ExitStack() as stack:
            camera = stack.enter_context(OpenCVCamera(camera_config))
            negotiated = camera.negotiated_properties()
            pose = stack.enter_context(
                PoseLandmarkerAdapter(
                    PoseLandmarkerConfig(
                        asset_path=pose_asset,
                        model_id=pose_config[f"{pose_prefix}_model_id"],
                        model_version=pose_config[f"{pose_prefix}_model_version"],
                        asset_sha256=pose_config[f"{pose_prefix}_asset_sha256"],
                        config_sha256=config_sha256,
                        num_poses=pose_config["num_poses"],
                        min_pose_detection_confidence=pose_config[
                            "min_pose_detection_confidence"
                        ],
                        min_pose_presence_confidence=pose_config[
                            "min_pose_presence_confidence"
                        ],
                        min_tracking_confidence=pose_config[
                            "min_tracking_confidence"
                        ],
                    )
                )
            )
            face = stack.enter_context(
                FaceLandmarkerAdapter(
                    FaceLandmarkerConfig(
                        asset_path=face_asset,
                        model_id=face_config["model_id"],
                        model_version=face_config["model_version"],
                        asset_sha256=face_config["asset_sha256"],
                        config_sha256=config_sha256,
                        num_faces=face_config["num_faces"],
                        output_face_blendshapes=face_config[
                            "output_face_blendshapes"
                        ],
                        output_facial_transformation_matrixes=face_config[
                            "output_facial_transformation_matrixes"
                        ],
                        min_face_detection_confidence=face_config[
                            "min_face_detection_confidence"
                        ],
                        min_face_presence_confidence=face_config[
                            "min_face_presence_confidence"
                        ],
                        min_tracking_confidence=face_config[
                            "min_tracking_confidence"
                        ],
                    )
                )
            )
            engine = PartALiveEngine(
                pose=pose,
                face=face,
                schedule=schedule,
                pose_features=PoseFeatureExtractor(
                    PoseFeatureConfig(
                        min_visibility=feature_config["min_visibility"],
                        min_presence=feature_config["min_presence"],
                        max_motion_gap_ms=feature_config["max_motion_gap_ms"],
                    )
                ),
                face_features=FaceFeatureExtractor(),
            )
            if args.enable_audio:
                audio_poller = stack.enter_context(
                    AudioLevelPoller(
                        MicrophoneConfig(
                            device_index=args.audio_device_index,
                            source_id=(
                                f"laptop_microphone_{args.audio_device_index}_exploratory"
                            ),
                            sample_rate_hz=args.audio_sample_rate,
                            channel_count=1,
                            block_duration_ms=args.audio_block_ms,
                        )
                    )
                )
            if not args.headless:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            loop_started = time.monotonic()
            while True:
                result = camera.read()
                camera_statuses[result.status.value] += 1
                if (
                    args.duration_seconds
                    and time.monotonic() - loop_started >= args.duration_seconds
                ):
                    termination = "duration"
                    break
                if result.status is not CameraReadStatus.OK or result.frame is None:
                    gap_reason = result.reason or f"camera_read_{result.status.value}"
                    calibration_progress = calibration_collector.mark_evidence_gap(
                        result.observed_at_ns,
                        reason=gap_reason,
                    )
                    if latest_rules is not None:
                        latest_rules = rule_engine.mark_evidence_gap(
                            result.observed_at_ns,
                            reason=gap_reason,
                        )
                    latest_audio = (
                        audio_poller.snapshot() if audio_poller is not None else None
                    )
                    if result.status is CameraReadStatus.DISCONNECTED:
                        termination = "camera_disconnected"
                        break
                    if not args.headless:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            termination = "user_exit"
                            break
                        if _window_closed(title):
                            termination = "window_closed"
                            break
                    continue
                snapshot = engine.process(result.frame)
                calibration_progress = calibration_collector.update(snapshot)
                if (
                    calibration_collector.profile is not None
                    and rule_engine.profile is None
                ):
                    rule_engine.set_profile(calibration_collector.profile)
                latest_audio = (
                    audio_poller.snapshot() if audio_poller is not None else None
                )
                latest_rules = rule_engine.update(
                    snapshot,
                    audio_level=(
                        latest_audio.observation if latest_audio is not None else None
                    ),
                )
                current_summary = engine.summary()
                if not args.headless:
                    cv2.imshow(
                        title,
                        _render(
                            snapshot,
                            current_summary,
                            args.pose_model,
                            calibration_progress,
                            latest_rules,
                            latest_audio,
                        ),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        termination = "user_exit"
                        break
                    if key in (ord("c"), ord("C")):
                        calibration_collector.reset()
                        rule_engine.reset(keep_profile=False)
                        calibration_progress = calibration_collector.progress
                        latest_rules = None
                    if _window_closed(title):
                        termination = "window_closed"
                        break
                if args.max_frames and current_summary["frames"] >= args.max_frames:
                    termination = "max_frames"
                    break
        if audio_poller is not None:
            latest_audio = audio_poller.snapshot()
            audio_cleanup = {
                "enabled": True,
                "poller_stopped": not audio_poller.is_running,
                "error": latest_audio.error,
            }
    except (CameraError, ValueError, cv2.error) as error:
        runtime_error = f"{type(error).__name__}:{error}"
        termination = "runtime_error"
    finally:
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error as error:
                window_cleanup_error = f"cv2_destroy_failed:{type(error).__name__}"

    if audio_poller is not None and "poller_stopped" not in audio_cleanup:
        latest_audio = audio_poller.snapshot()
        audio_cleanup = {
            "enabled": True,
            "poller_stopped": not audio_poller.is_running,
            "error": latest_audio.error,
        }

    summary = engine.summary() if engine is not None else {}
    summary.update(
        {
            "status": (
                "error"
                if runtime_error is not None
                else (
                    "camera_disconnected"
                    if termination == "camera_disconnected"
                    else "completed"
                )
            ),
            "termination": termination,
            "error": runtime_error,
            "wall_seconds": time.monotonic() - wall_started,
            "capture_loop_wall_seconds": (
                time.monotonic() - loop_started if loop_started is not None else None
            ),
            "camera": negotiated,
            "camera_read_counts": dict(camera_statuses),
            "pose_model_choice": args.pose_model,
            "event_config": {
                "path": str(event_config_path),
                "sha256": event_config_sha256,
                "schema_version": event_config.schema_version,
                "status": event_config.status,
            },
            "calibration": _calibration_summary(calibration_progress),
            "semantic_states": _rules_summary(latest_rules),
            "audio_level": _audio_summary(args.enable_audio, latest_audio),
            "cleanup": {
                "audio": audio_cleanup,
                "window_error": window_cleanup_error,
            },
            "exploratory_only": True,
            "records_media": False,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if runtime_error is not None:
        return 1
    return 2 if termination == "camera_disconnected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
