"""Evaluate a local card detector and persist aggregate and grouped metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_macro_metrics(
    per_class: Sequence[dict[str, object]],
    key: str,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in per_class:
        groups.setdefault(str(record[key]), []).append(record)
    metric_names = ("precision", "recall", "f1", "map50", "map50_95")
    return {
        group: {
            metric: float(
                np.mean([float(record[metric]) for record in records])
            )
            for metric in metric_names
        }
        for group, records in sorted(groups.items())
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        parser.error("imgsz/batch must be positive and workers non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    project = args.project.resolve()
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        raise ValueError(f"model must be an existing local .pt file: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=False,
    )
    names = [str(model.names[index]) for index in sorted(model.names)]
    arrays = {
        "precision": np.asarray(metrics.box.p, dtype=float),
        "recall": np.asarray(metrics.box.r, dtype=float),
        "f1": np.asarray(metrics.box.f1, dtype=float),
        "map50": np.asarray(metrics.box.ap50, dtype=float),
        "map50_95": np.asarray(metrics.box.maps, dtype=float),
    }
    if any(len(values) != len(names) for values in arrays.values()):
        raise ValueError("evaluation did not return one metric row per class")
    per_class = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "rank": class_name[:-1],
            "suit": class_name[-1],
            **{
                metric_name: float(values[class_id])
                for metric_name, values in arrays.items()
            },
        }
        for class_id, class_name in enumerate(names)
    ]
    result = {
        "schema_version": "poker_dealer.card_detector_evaluation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "dataset": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "split": "val",
        },
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
        },
        "overall": {
            key: float(value)
            for key, value in metrics.results_dict.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "per_class": per_class,
        "macro_by_rank": grouped_macro_metrics(per_class, "rank"),
        "macro_by_suit": grouped_macro_metrics(per_class, "suit"),
        "confusion_matrix": np.asarray(
            metrics.confusion_matrix.matrix,
            dtype=float,
        ).tolist(),
        "confusion_matrix_axis": {
            "rows": "predicted class plus background",
            "columns": "true class plus background",
            "labels": [*names, "background"],
        },
        "artifacts": str(Path(metrics.save_dir).resolve()),
    }
    output = Path(metrics.save_dir) / "poker_dealer_evaluation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "overall": result["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
