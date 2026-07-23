"""Evaluate fixed-design denomination templates on labelled chip captures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from build_chip_templates import ROOT, _relative
from chip_template_matcher import (
    ChipTemplateMatcher,
    DENOMINATIONS,
    center_number_view,
    digit_mask,
)
from rectify_chip_images import (
    DEFAULT_MODEL,
    _derive_top_ellipse_from_inlay,
    _ellipse_to_circle,
    _expand_bbox,
    _fit_top_ellipse,
    _grabcut_chip_mask,
)


DEFAULT_INPUT = ROOT / "data" / "chips"
DEFAULT_LIBRARY = (
    ROOT
    / "data"
    / "work"
    / "chips"
    / "2026-07-23-template-matching"
    / "library"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "work"
    / "chips"
    / "2026-07-23-template-matching"
    / "evaluation.json"
)
DEFAULT_DEBUG_DIR = DEFAULT_OUTPUT.parent / "evaluation_views"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--minimum-score", type=float, default=0.58)
    parser.add_argument("--minimum-margin", type=float, default=0.035)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--padding", type=float, default=0.14)
    parser.add_argument("--normalized-size", type=int, default=384)
    return parser.parse_args()


def _sources(root: Path) -> list[tuple[int, Path]]:
    sources: list[tuple[int, Path]] = []
    for denomination in DENOMINATIONS:
        directory = root / str(denomination)
        if not directory.is_dir():
            raise SystemExit(f"denomination directory is missing: {directory}")
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        sources.extend((denomination, path) for path in images)
    if not sources:
        raise SystemExit(f"no evaluation images found in: {root}")
    return sources


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), percentile)), 3)


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model is missing: {args.model}")
    sources = _sources(args.input)
    matcher = ChipTemplateMatcher(
        args.library,
        minimum_score=args.minimum_score,
        minimum_margin=args.minimum_margin,
    )
    detector = YOLO(str(args.model.resolve()))
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    started_ns = time.perf_counter_ns()
    predictions = detector.predict(
        [str(path) for _, path in sources],
        conf=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    detector_elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

    records: list[dict[str, object]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    failure_counts: Counter[str] = Counter()
    for (expected, source), prediction in zip(sources, predictions):
        record: dict[str, object] = {
            "source": _relative(source),
            "expected": expected,
        }
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            record["failure"] = "read_failed"
        elif len(prediction.boxes) == 0:
            record["failure"] = "no_detection"
        else:
            confidences = prediction.boxes.conf.detach().cpu().numpy()
            selected = int(np.argmax(confidences))
            raw_box = prediction.boxes.xyxy[selected].detach().cpu().numpy()
            x1, y1, x2, y2 = _expand_bbox(raw_box, image.shape, args.padding)
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            outer_mask = _grabcut_chip_mask(crop)
            preliminary, _ = _fit_top_ellipse(outer_mask)
            if preliminary is None:
                record["failure"] = "no_outer_ellipse"
            else:
                ellipse, _ = _derive_top_ellipse_from_inlay(crop, preliminary)
                if ellipse is None:
                    record["failure"] = "no_center_inlay_ellipse"
                else:
                    _, normalized = _ellipse_to_circle(
                        crop, ellipse, args.normalized_size
                    )
                    sample_id = f"expected_{expected}_{source.stem}"
                    normalized_path = args.debug_dir / f"{sample_id}_chip.png"
                    mask_path = args.debug_dir / f"{sample_id}_mask.png"
                    if not cv2.imwrite(str(normalized_path), normalized):
                        raise RuntimeError(f"failed to write: {normalized_path}")
                    if not cv2.imwrite(
                        str(mask_path),
                        digit_mask(center_number_view(normalized)),
                    ):
                        raise RuntimeError(f"failed to write: {mask_path}")
                    match = matcher.match_normalized_chip(normalized)
                    latencies.append(match.latency_ms)
                    record.update(
                        {
                            "detector_confidence": round(
                                float(confidences[selected]), 6
                            ),
                            "detections_in_source": len(prediction.boxes),
                            "ellipse_quality": ellipse.quality,
                            "predicted": match.denomination,
                            "accepted": match.accepted,
                            "best_score": match.best_score,
                            "margin": match.margin,
                            "scores": {str(k): v for k, v in match.scores.items()},
                            "matched_template": match.source_id,
                            "rotation_degrees": match.rotation_degrees,
                            "matching_latency_ms": match.latency_ms,
                            "normalized_file": _relative(normalized_path),
                            "mask_file": _relative(mask_path),
                        }
                    )
        if "failure" in record:
            failure_counts[str(record["failure"])] += 1
            confusion[str(expected)]["pipeline_failure"] += 1
        else:
            predicted_label = (
                str(record["predicted"])
                if record.get("predicted") is not None
                else "unknown"
            )
            confusion[str(expected)][predicted_label] += 1
        records.append(record)

    accepted = [record for record in records if record.get("accepted") is True]
    correct = [
        record
        for record in accepted
        if record.get("predicted") == record.get("expected")
    ]
    processed = [record for record in records if "failure" not in record]
    per_class: dict[str, dict[str, object]] = {}
    for denomination in DENOMINATIONS:
        class_records = [r for r in records if r["expected"] == denomination]
        class_accepted = [r for r in class_records if r.get("accepted") is True]
        class_correct = [
            r for r in class_accepted if r.get("predicted") == denomination
        ]
        per_class[str(denomination)] = {
            "total": len(class_records),
            "accepted": len(class_accepted),
            "correct": len(class_correct),
            "coverage": round(len(class_accepted) / len(class_records), 6),
            "accepted_accuracy": (
                round(len(class_correct) / len(class_accepted), 6)
                if class_accepted
                else None
            ),
        }

    total = len(records)
    report = {
        "schema_version": "1.0",
        "evaluation_kind": "development_fit_not_independent_held_out",
        "input": _relative(args.input),
        "library": _relative(args.library),
        "model": _relative(args.model),
        "thresholds": {
            "detector_confidence": args.confidence,
            "minimum_score": args.minimum_score,
            "minimum_margin": args.minimum_margin,
        },
        "summary": {
            "total": total,
            "pipeline_processed": len(processed),
            "pipeline_failures": total - len(processed),
            "accepted": len(accepted),
            "correct": len(correct),
            "coverage": round(len(accepted) / total, 6),
            "accepted_accuracy": (
                round(len(correct) / len(accepted), 6) if accepted else None
            ),
            "overall_correct_rate": round(len(correct) / total, 6),
        },
        "per_class": per_class,
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "pipeline_failure_counts": dict(failure_counts),
        "latency_ms": {
            "detector_batch_total": round(detector_elapsed_ms, 3),
            "detector_batch_per_image": round(detector_elapsed_ms / total, 3),
            "template_match_p50": _percentile(latencies, 50),
            "template_match_p95": _percentile(latencies, 95),
            "template_match_max": round(max(latencies), 3) if latencies else None,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("summary", "per_class", "confusion", "pipeline_failure_counts", "latency_ms")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
