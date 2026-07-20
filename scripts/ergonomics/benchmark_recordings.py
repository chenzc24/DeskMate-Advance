"""Compare Part A candidates on identical hash-verified recordings.

This command never records or copies media. Detailed output is restricted to
the ignored artifacts directory because it may contain pseudonymous session
metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import mediapipe
import numpy as np

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.features.ergonomics import (
    FaceFeatureConfig,
    FaceFeatureExtractor,
    PoseFeatureConfig,
    PoseFeatureExtractor,
)
from deskmate_advance.features.ergonomics.benchmark import (
    ComponentMetrics,
    RecordingRecord,
    iter_timestamp_sidecar,
    load_recording_manifest,
    sha256_file,
)
from deskmate_advance.perception.ergonomics import (
    FaceLandmarkerAdapter,
    FaceLandmarkerConfig,
    ObservationState,
    PoseLandmarkerAdapter,
    PoseLandmarkerConfig,
)

try:
    import psutil
except ImportError:  # The benchmark extra is recommended but not mandatory.
    psutil = None


POSE_FEATURES = (
    "normalized_landmarks",
    "shoulder_tilt",
    "torso_lean",
    "nose_offset",
    "motion",
    "world_landmarks",
)
FACE_FEATURES = (
    "face_bbox",
    "raw_rotation",
    "raw_translation",
    "blink_scores",
)


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _asset_path_and_verify(
    project_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    asset_path = (project_root / relative_path).resolve()
    if project_root not in asset_path.parents or not asset_path.is_file():
        raise ValueError(f"invalid local asset path: {asset_path}")
    actual = sha256_file(asset_path)
    if actual != expected_sha256:
        raise ValueError(f"asset SHA-256 mismatch: {asset_path}")
    return asset_path


def _new_metric_set() -> dict[str, ComponentMetrics]:
    return {
        "pose_full": ComponentMetrics(POSE_FEATURES),
        "pose_lite": ComponentMetrics(POSE_FEATURES),
        "face": ComponentMetrics(FACE_FEATURES),
    }


def _record_pose(
    metrics: ComponentMetrics,
    observation,
    features,
) -> None:
    metrics.add(
        state=observation.state,
        timestamp_ns=observation.context.captured_at_ns,
        inference_ms=observation.context.inference_ms,
        dropped_before=observation.context.dropped_before,
        available={
            "normalized_landmarks": features.normalization_scale is not None,
            "shoulder_tilt": features.shoulder_tilt_deg is not None,
            "torso_lean": features.torso_lean_from_vertical_deg is not None,
            "nose_offset": features.nose_offset_from_shoulders is not None,
            "motion": features.upper_body_motion_per_second is not None,
            "world_landmarks": bool(observation.world_landmarks),
        },
    )


def _record_face(metrics: ComponentMetrics, observation, features) -> None:
    metrics.add(
        state=observation.state,
        timestamp_ns=observation.context.captured_at_ns,
        inference_ms=observation.context.inference_ms,
        dropped_before=observation.context.dropped_before,
        available={
            "face_bbox": features.geometry_state is ObservationState.VALID,
            "raw_rotation": features.rotation_state is ObservationState.VALID,
            "raw_translation": features.raw_translation_xyz is not None,
            "blink_scores": features.blink_state is ObservationState.VALID,
        },
    )


def _fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))


def _process_recording(
    record: RecordingRecord,
    *,
    config: dict[str, Any],
    config_sha256: str,
    assets: dict[str, Path],
    max_frames: int,
    aggregate: dict[str, ComponentMetrics],
) -> dict[str, Any]:
    local = _new_metric_set()
    for metrics in (*aggregate.values(), *local.values()):
        metrics.begin_recording()

    pose_config = config["pose"]
    face_config = config["face"]
    feature_config = config["features"]
    init_started = time.perf_counter_ns()
    pose_full = PoseLandmarkerAdapter(
        PoseLandmarkerConfig(
            asset_path=assets["pose_full"],
            model_id=pose_config["primary_model_id"],
            model_version=pose_config["primary_model_version"],
            asset_sha256=pose_config["primary_asset_sha256"],
            config_sha256=config_sha256,
            num_poses=pose_config["num_poses"],
            min_pose_detection_confidence=pose_config[
                "min_pose_detection_confidence"
            ],
            min_pose_presence_confidence=pose_config[
                "min_pose_presence_confidence"
            ],
            min_tracking_confidence=pose_config["min_tracking_confidence"],
        )
    )
    full_initialized = time.perf_counter_ns()
    pose_lite = PoseLandmarkerAdapter(
        PoseLandmarkerConfig(
            asset_path=assets["pose_lite"],
            model_id=pose_config["fallback_model_id"],
            model_version=pose_config["fallback_model_version"],
            asset_sha256=pose_config["fallback_asset_sha256"],
            config_sha256=config_sha256,
            num_poses=pose_config["num_poses"],
            min_pose_detection_confidence=pose_config[
                "min_pose_detection_confidence"
            ],
            min_pose_presence_confidence=pose_config[
                "min_pose_presence_confidence"
            ],
            min_tracking_confidence=pose_config["min_tracking_confidence"],
        )
    )
    lite_initialized = time.perf_counter_ns()
    face = FaceLandmarkerAdapter(
        FaceLandmarkerConfig(
            asset_path=assets["face"],
            model_id=face_config["model_id"],
            model_version=face_config["model_version"],
            asset_sha256=face_config["asset_sha256"],
            config_sha256=config_sha256,
            num_faces=face_config["num_faces"],
            output_face_blendshapes=face_config["output_face_blendshapes"],
            output_facial_transformation_matrixes=face_config[
                "output_facial_transformation_matrixes"
            ],
            min_face_detection_confidence=face_config[
                "min_face_detection_confidence"
            ],
            min_face_presence_confidence=face_config[
                "min_face_presence_confidence"
            ],
            min_tracking_confidence=face_config["min_tracking_confidence"],
        )
    )
    face_initialized = time.perf_counter_ns()
    init_ms = {
        "pose_full": (full_initialized - init_started) / 1_000_000,
        "pose_lite": (lite_initialized - full_initialized) / 1_000_000,
        "face": (face_initialized - lite_initialized) / 1_000_000,
    }

    pose_features_config = PoseFeatureConfig(
        min_visibility=feature_config["pose"]["min_visibility"],
        min_presence=feature_config["pose"]["min_presence"],
        max_motion_gap_ms=feature_config["pose"]["max_motion_gap_ms"],
    )
    full_features = PoseFeatureExtractor(pose_features_config)
    lite_features = PoseFeatureExtractor(pose_features_config)
    face_features = FaceFeatureExtractor(FaceFeatureConfig())
    capture = cv2.VideoCapture(str(record.media_path))
    if not capture.isOpened():
        pose_full.close()
        pose_lite.close()
        face.close()
        raise RuntimeError(f"cannot open recording: {record.media_path}")

    metadata = {
        "reported_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "reported_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec_fourcc": _fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
        "color_space": "bgr",
    }
    process = psutil.Process() if psutil is not None else None
    peak_rss = process.memory_info().rss if process is not None else None
    frames = 0
    loop_started = time.perf_counter_ns()
    length_validation = "complete"
    timestamp_iter = iter_timestamp_sidecar(record.timestamp_path)
    try:
        while frames < max_frames:
            try:
                timestamp = next(timestamp_iter)
            except StopIteration:
                ok, extra_frame = capture.read()
                if ok and extra_frame is not None:
                    raise ValueError("video contains more frames than timestamp sidecar")
                break
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                raise ValueError("timestamp sidecar contains more frames than video")
            owned = np.ascontiguousarray(raw_frame).copy()
            owned.setflags(write=False)
            height, width = owned.shape[:2]
            packet = FramePacket(
                sequence_id=timestamp.frame_index,
                captured_at_ns=timestamp.captured_at_ns,
                source_id=record.sample_id,
                device_index=0,
                width=width,
                height=height,
                color_space=ColorSpace.BGR,
                nominal_fps=metadata["reported_fps"],
                dropped_before=timestamp.dropped_before,
                image=owned,
            )
            full_observation = pose_full.observe(packet)
            lite_observation = pose_lite.observe(packet)
            face_observation = face.observe(packet)
            full_feature = full_features.extract(full_observation)
            lite_feature = lite_features.extract(lite_observation)
            face_feature = face_features.extract(face_observation)
            for metrics in (aggregate["pose_full"], local["pose_full"]):
                _record_pose(metrics, full_observation, full_feature)
            for metrics in (aggregate["pose_lite"], local["pose_lite"]):
                _record_pose(metrics, lite_observation, lite_feature)
            for metrics in (aggregate["face"], local["face"]):
                _record_face(metrics, face_observation, face_feature)
            frames += 1
            if process is not None:
                peak_rss = max(peak_rss or 0, process.memory_info().rss)
        if frames == max_frames:
            length_validation = "not_completed_frame_limit_reached"
    finally:
        capture.release()
        pose_full.close()
        pose_lite.close()
        face.close()
    elapsed_seconds = (time.perf_counter_ns() - loop_started) / 1_000_000_000
    return {
        "sample_id": record.sample_id,
        "media_sha256": record.media_sha256,
        "timestamp_sha256": record.timestamp_sha256,
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "device_id": record.device_id,
        "scenario": record.scenario,
        "scenario_tags": list(record.scenario_tags),
        "split": record.split,
        "media_metadata": metadata,
        "processed_frames": frames,
        "length_validation": length_validation,
        "wall_seconds": elapsed_seconds,
        "full_loop_fps": frames / elapsed_seconds if elapsed_seconds else None,
        "peak_rss_mb": peak_rss / (1024 * 1024) if peak_rss is not None else None,
        "initialization_ms": init_ms,
        "components": {name: metrics.summary() for name, metrics in local.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ergonomics/perception.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ergonomics/a2-benchmark.json"),
    )
    parser.add_argument("--max-frames-per-recording", type=int, default=1800)
    args = parser.parse_args()
    if args.max_frames_per_recording <= 0:
        parser.error("--max-frames-per-recording must be positive")

    config_path = args.config.resolve()
    project_root = config_path.parents[2]
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    manifest_root = (project_root / "data" / "manifests").resolve()
    artifacts_root = (project_root / "artifacts").resolve()
    if manifest_root not in manifest_path.parents:
        parser.error("--manifest must resolve under data/manifests")
    if artifacts_root not in output_path.parents:
        parser.error("--output must resolve under the ignored artifacts directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = _config_hash(config_path)
    pose_config = config["pose"]
    face_config = config["face"]
    assets = {
        "pose_full": _asset_path_and_verify(
            project_root,
            pose_config["primary_asset"],
            pose_config["primary_asset_sha256"],
        ),
        "pose_lite": _asset_path_and_verify(
            project_root,
            pose_config["fallback_asset"],
            pose_config["fallback_asset_sha256"],
        ),
        "face": _asset_path_and_verify(
            project_root,
            face_config["asset"],
            face_config["asset_sha256"],
        ),
    }
    records = load_recording_manifest(
        manifest_path,
        project_root=project_root,
        verify_files=True,
    )
    aggregate = _new_metric_set()
    sessions = [
        _process_recording(
            record,
            config=config,
            config_sha256=config_sha256,
            assets=assets,
            max_frames=args.max_frames_per_recording,
            aggregate=aggregate,
        )
        for record in records
    ]
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workstream": "A",
        "evidence_status": "target" if all(
            "target_camera" in record.scenario_tags for record in records
        ) else "exploratory",
        "config": {
            "path": str(config_path.relative_to(project_root)),
            "sha256": config_sha256,
        },
        "manifest": {
            "path": str(manifest_path.relative_to(project_root)),
            "sha256": sha256_file(manifest_path),
        },
        "git": _git_provenance(project_root),
        "environment": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "mediapipe": mediapipe.__version__,
            "psutil_available": psutil is not None,
        },
        "assets": {
            name: {"path": str(path.relative_to(project_root)), "sha256": sha256_file(path)}
            for name, path in assets.items()
        },
        "max_frames_per_recording": args.max_frames_per_recording,
        "sessions": sessions,
        "aggregate_components": {
            name: metrics.summary() for name, metrics in aggregate.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
