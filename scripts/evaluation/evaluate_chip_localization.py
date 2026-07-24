"""Evaluate a local single-class chip detector and write a compact JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    model_path = resolve(args.model).resolve()
    data_path = resolve(args.data).resolve()
    output_path = resolve(args.output).resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not data_path.is_file():
        raise SystemExit(f"dataset yaml not found: {data_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = YOLO(str(model_path)).val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        plots=False,
        save_json=False,
        project=str(output_path.parent),
        name=f"{output_path.stem}_artifacts",
        exist_ok=True,
        verbose=False,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "schema_version": "1.0",
        "task": "single-class poker-chip localization",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "fitness": float(metrics.fitness),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
