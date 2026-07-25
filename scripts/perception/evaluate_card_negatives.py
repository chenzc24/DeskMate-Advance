"""Evaluate card-detector false positives on expected-empty image sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_confidences(
    confidences_by_image: Sequence[Sequence[float]],
    threshold: float,
) -> dict[str, float | int]:
    counts = [
        sum(confidence >= threshold for confidence in confidences)
        for confidences in confidences_by_image
    ]
    images = len(counts)
    false_boxes = sum(counts)
    return {
        "images": images,
        "images_with_false_detection": sum(count > 0 for count in counts),
        "false_boxes": false_boxes,
        "false_boxes_per_image": false_boxes / images if images else 0.0,
    }


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--model must be NAME=PATH")
    return name, Path(raw_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    args.threshold = args.threshold or [0.15, 0.25, 0.5]
    if any(not 0.0 < value <= 1.0 for value in args.threshold):
        parser.error("thresholds must be in (0, 1]")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    images_root = args.images.resolve()
    images = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no images found under {images_root}")
    thresholds = sorted(set(args.threshold))
    result: dict[str, object] = {
        "schema_version": "poker_dealer.card_negative_evaluation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_output": "zero card detections",
        "images": {
            "root": str(images_root),
            "count": len(images),
            "records": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in images
            ],
        },
        "settings": {
            "imgsz": args.imgsz,
            "device": args.device,
            "thresholds": thresholds,
        },
        "models": {},
    }
    for name, raw_model_path in args.model:
        model_path = raw_model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = YOLO(str(model_path))
        predictions = model.predict(
            source=[str(path) for path in images],
            imgsz=args.imgsz,
            conf=min(thresholds),
            device=args.device,
            verbose=False,
        )
        confidence_rows = [
            [float(value) for value in prediction.boxes.conf.cpu().tolist()]
            for prediction in predictions
        ]
        model_records = []
        for image_path, prediction in zip(images, predictions, strict=True):
            boxes = []
            for class_id, confidence in zip(
                prediction.boxes.cls.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                strict=True,
            ):
                boxes.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(model.names[int(class_id)]),
                        "confidence": float(confidence),
                    }
                )
            model_records.append({"image": str(image_path), "detections": boxes})
        result["models"][name] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "metrics": {
                str(threshold): aggregate_confidences(confidence_rows, threshold)
                for threshold in thresholds
            },
            "records": model_records,
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "models": result["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
