"""Derive ignored normalized hand-landmark sequences from immutable action videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from poker_dealer.evaluation import canonical_sha256, validate_action_manifest
from poker_dealer.training import ActionTcnConfig, normalize_hand_landmarks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/training/action_tcn_v1.json"
DEFAULT_ASSET = ROOT / "models/assets/hand_landmarker.task"
EXPECTED_ASSET_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
NEGATIVE_LABELS = {"cancelled", "ambiguous", "occluded"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(source_id: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", source_id).strip("_")[:48] or "source"
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{suffix}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resolved_source_manifest", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/work/action_landmarks_v1",
    )
    parser.add_argument("--view-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.resolved_source_manifest.read_text(encoding="utf-8"))
        errors = validate_action_manifest(manifest, root=ROOT, verify_files=True)
        if errors:
            raise ValueError("; ".join(errors))
        if manifest.get("status") != "resolved":
            raise ValueError("landmark extraction requires a resolved source manifest")
        config = ActionTcnConfig.from_json(args.config)
        if _sha256(args.asset) != EXPECTED_ASSET_SHA256:
            raise ValueError("hand landmarker asset SHA-256 mismatch")
        output_root = args.output_root.resolve()
        work_root = (ROOT / "data/work").resolve()
        if work_root not in output_root.parents and output_root != work_root:
            raise ValueError("derived landmarks must stay under ignored data/work")
        view_manifest_path = args.view_manifest or output_root / "view_manifest.json"
        if view_manifest_path.exists():
            raise FileExistsError(f"view manifest already exists: {view_manifest_path}")

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.asset)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        derived_records: list[dict[str, object]] = []
        with HandLandmarker.create_from_options(options) as landmarker:
            for record in manifest["records"]:
                video_path = ROOT / Path(record["capture_path"])
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise OSError(f"cannot open action video: {video_path}")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if fps <= 0:
                    fps = 30.0
                features: list[np.ndarray] = []
                valid: list[bool] = []
                timestamps_ms: list[int] = []
                frame_index = 0
                try:
                    while True:
                        ok, bgr = capture.read()
                        if not ok:
                            break
                        timestamp_ms = int(round(frame_index * 1000.0 / fps))
                        if timestamps_ms and timestamp_ms <= timestamps_ms[-1]:
                            timestamp_ms = timestamps_ms[-1] + 1
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        result = landmarker.detect_for_video(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                            timestamp_ms,
                        )
                        if len(result.hand_landmarks) == 1:
                            values = np.asarray(
                                [
                                    (landmark.x, landmark.y, landmark.z)
                                    for landmark in result.hand_landmarks[0]
                                ],
                                dtype=np.float32,
                            )
                            try:
                                feature = normalize_hand_landmarks(values)
                                is_valid = True
                            except ValueError:
                                feature = np.zeros((63,), dtype=np.float32)
                                is_valid = False
                        else:
                            feature = np.zeros((63,), dtype=np.float32)
                            is_valid = False
                        features.append(feature)
                        valid.append(is_valid)
                        timestamps_ms.append(timestamp_ms)
                        frame_index += 1
                finally:
                    capture.release()
                if not features:
                    raise ValueError(f"video contains no readable frames: {video_path}")
                output_path = output_root / _safe_name(str(record["source_id"]))
                if output_path.exists():
                    raise FileExistsError(f"derived view already exists: {output_path}")
                np.savez_compressed(
                    output_path,
                    features=np.stack(features),
                    valid_mask=np.asarray(valid, dtype=np.bool_),
                    timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
                )
                source_label = str(record["label"])
                training_label = (
                    "no_action" if source_label in NEGATIVE_LABELS else source_label
                )
                relative_output = output_path.relative_to(ROOT).as_posix()
                derived_records.append(
                    {
                        "source_id": record["source_id"],
                        "source_sha256": record["sha256"],
                        "participant_code": record["participant_code"],
                        "session_id": record["session_id"],
                        "seat": record["seat"],
                        "split": record["split"],
                        "source_label": source_label,
                        "label": training_label,
                        "view_path": relative_output,
                        "view_sha256": _sha256(output_path),
                        "frames": len(features),
                        "valid_frames": int(sum(valid)),
                        "valid_ratio": round(sum(valid) / len(valid), 6),
                    }
                )
        view_manifest = {
            "schema_version": "1.0",
            "status": "derived",
            "model_id": config.model_id,
            "config_sha256": _sha256(args.config),
            "landmarker_asset_sha256": EXPECTED_ASSET_SHA256,
            "source_manifest_sha256": canonical_sha256(manifest),
            "records": derived_records,
        }
        view_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with view_manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(view_manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "result": "PASS",
                "records": len(derived_records),
                "view_manifest": str(view_manifest_path),
                "frames_saved": 0,
                "audio_saved": False,
                "derived_landmarks_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
