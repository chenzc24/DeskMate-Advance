"""Verify local MediaPipe assets and run bounded synthetic-image inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import mediapipe as mp
import numpy as np


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_asset(project_root: Path, record: dict[str, Any]) -> Path:
    asset_path = (project_root / record["local_path"]).resolve()
    if project_root not in asset_path.parents:
        raise ValueError(f"asset escapes project root: {asset_path}")
    payload = asset_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != record["bytes"]:
        raise ValueError(f"size mismatch for {asset_path}")
    if actual_hash != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {asset_path}")
    return asset_path


def _timed_detect(
    task_factory: Callable[[], Any],
    image: mp.Image,
) -> tuple[float, float, Any]:
    started = time.perf_counter_ns()
    with task_factory() as task:
        initialized = time.perf_counter_ns()
        result = task.detect(image)
        completed = time.perf_counter_ns()
    return (
        (initialized - started) / 1_000_000,
        (completed - initialized) / 1_000_000,
        result,
    )


def _factories(asset_by_role: dict[str, Path]) -> list[tuple[str, Callable[[], Any], Callable[[Any], int]]]:
    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions
    return [
        (
            "pose_landmarker_full",
            lambda: vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=str(asset_by_role["pose_primary"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                )
            ),
            lambda result: len(result.pose_landmarks),
        ),
        (
            "pose_landmarker_lite",
            lambda: vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=str(asset_by_role["pose_fallback"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                )
            ),
            lambda result: len(result.pose_landmarks),
        ),
        (
            "hand_landmarker",
            lambda: vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=str(asset_by_role["hand_primary"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                )
            ),
            lambda result: len(result.hand_landmarks),
        ),
        (
            "face_landmarker",
            lambda: vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=str(asset_by_role["face_primary"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            ),
            lambda result: len(result.face_landmarks),
        ),
        (
            "phone_detector",
            lambda: vision.ObjectDetector.create_from_options(
                vision.ObjectDetectorOptions(
                    base_options=base_options(
                        model_asset_path=str(asset_by_role["phone_primary"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    category_allowlist=["cell phone"],
                    score_threshold=0.3,
                    max_results=3,
                )
            ),
            lambda result: len(result.detections),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/manifest.yaml"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    project_root = manifest_path.parent.parent
    manifest = _load_manifest(manifest_path)
    records = manifest["models"]

    verified: dict[tuple[str, str], Path] = {}
    for record in records:
        verified[(record["model_id"], record["role"])] = _verify_asset(
            project_root, record
        )
    for record in manifest.get("supporting_assets", []):
        _verify_asset(project_root, record)

    asset_by_role = {
        "pose_primary": verified[("pose_landmarker", "primary_gate1_evaluation")],
        "pose_fallback": verified[("pose_landmarker", "fallback_gate1_evaluation")],
        "hand_primary": verified[("hand_landmarker", "primary_gate1_evaluation")],
        "face_primary": verified[("face_landmarker", "primary_gate1_evaluation")],
        "phone_primary": verified[("phone_detector", "primary_gate1_evaluation")],
    }
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    for task_id, factory, count_observations in _factories(asset_by_role):
        init_ms, inference_ms, result = _timed_detect(factory, image)
        print(
            json.dumps(
                {
                    "task": task_id,
                    "asset_verified": True,
                    "synthetic_observations": count_observations(result),
                    "initialization_ms": round(init_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
