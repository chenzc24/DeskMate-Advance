"""Measure chip-detector false positives on an image-only negative replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.25, 0.50),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve(args.model).resolve()
    image_root = resolve(args.images).resolve()
    output_path = resolve(args.output).resolve()
    files = image_paths(image_root)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not files:
        raise SystemExit(f"no negative images found: {image_root}")
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    if not thresholds or thresholds[0] <= 0 or thresholds[-1] >= 1:
        raise SystemExit("thresholds must be inside (0, 1)")

    results = YOLO(str(model_path)).predict(
        [str(path) for path in files],
        imgsz=args.imgsz,
        conf=min(0.01, thresholds[0]),
        device=args.device,
        batch=args.batch,
        verbose=False,
        save=False,
    )
    rows: list[dict[str, object]] = []
    summaries = {
        f"{threshold:.2f}": {"detections": 0, "images_with_detections": 0}
        for threshold in thresholds
    }
    group_summaries: dict[str, dict[str, dict[str, int]]] = {}
    for path, result in zip(files, results):
        confidences = [float(value) for value in result.boxes.conf.cpu().tolist()]
        group = path.relative_to(image_root).parts[0] if path != image_root else "."
        group_summary = group_summaries.setdefault(
            group,
            {
                f"{threshold:.2f}": {
                    "detections": 0,
                    "images_with_detections": 0,
                }
                for threshold in thresholds
            },
        )
        threshold_counts: dict[str, int] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            count = sum(confidence >= threshold for confidence in confidences)
            threshold_counts[key] = count
            summaries[key]["detections"] += count
            summaries[key]["images_with_detections"] += int(count > 0)
            group_summary[key]["detections"] += count
            group_summary[key]["images_with_detections"] += int(count > 0)
        rows.append(
            {
                "relative_path": path.relative_to(image_root).as_posix(),
                "maximum_confidence": max(confidences, default=0.0),
                "threshold_counts": threshold_counts,
                "confidences": confidences,
            }
        )

    report = {
        "schema_version": "1.0",
        "task": "poker-chip false-positive negative replay",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "image_root": str(image_root),
        "image_count": len(files),
        "imgsz": args.imgsz,
        "thresholds": thresholds,
        "summary": summaries,
        "group_summary": group_summaries,
        "maximum_false_positive_confidence": max(
            (float(row["maximum_confidence"]) for row in rows),
            default=0.0,
        ),
        "images": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
