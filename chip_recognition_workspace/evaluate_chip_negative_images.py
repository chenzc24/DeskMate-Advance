"""Measure chip-detector responses on an all-negative image directory.

The report is a fit diagnostic when the images were used for training. It must
not be presented as held-out false-positive performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Mark that these images contributed to training.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    image_dir = args.images.resolve()
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"no images found in {image_dir}")

    minimum_threshold = min(args.thresholds)
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=minimum_threshold,
        iou=0.5,
        device=args.device,
        verbose=False,
        save=False,
        stream=False,
    )

    records = []
    for image_path, result in zip(image_paths, results, strict=True):
        confidences = (
            [float(value) for value in result.boxes.conf.cpu().tolist()]
            if result.boxes is not None
            else []
        )
        records.append(
            {
                "image": image_path.name,
                "image_sha256": sha256(image_path),
                "detection_confidences": confidences,
            }
        )

    summaries = {}
    for threshold in args.thresholds:
        summaries[str(threshold)] = {
            "images_with_detections": sum(
                any(confidence >= threshold for confidence in record["detection_confidences"])
                for record in records
            ),
            "total_detections": sum(
                sum(confidence >= threshold for confidence in record["detection_confidences"])
                for record in records
            ),
            "maximum_confidence": max(
                (
                    confidence
                    for record in records
                    for confidence in record["detection_confidences"]
                    if confidence >= threshold
                ),
                default=None,
            ),
        }

    report = {
        "schema_version": "1.0",
        "status": (
            "development_fit_only_no_independent_validation"
            if args.fit_only
            else "development_negative_evaluation"
        ),
        "warning": (
            "These negative images contributed to training; results measure fit only."
            if args.fit_only
            else "Results apply only to this image directory and capture session."
        ),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "images": str(image_dir),
        "image_count": len(records),
        "imgsz": args.imgsz,
        "summaries": summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
