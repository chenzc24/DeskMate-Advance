"""Check a chip detector against the reviewed foreground box in source frames.

This is a resubstitution/fit diagnostic. The source frames belong to the same
capture session as training, and background chip stacks were not exhaustively
annotated. Unmatched detections are therefore reported, not called false
positives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from ultralytics import YOLO


EXPECTED_NAMES = {0: "chip_1", 1: "chip_5", 2: "chip_10", 3: "chip_20"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.5)
    return parser.parse_args()


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def summarize(records: list[dict[str, object]], threshold: float, match_iou: float) -> dict[str, object]:
    selected = []
    for record in records:
        detections = [
            detection
            for detection in record["detections"]
            if detection["confidence"] >= threshold
        ]
        matches = [
            detection for detection in detections if detection["iou"] >= match_iou
        ]
        best_match = max(matches, key=lambda item: item["iou"], default=None)
        selected.append((record, detections, best_match))

    class_rows: dict[str, dict[str, object]] = {}
    for class_name in EXPECTED_NAMES.values():
        class_selected = [row for row in selected if row[0]["class_name"] == class_name]
        correct = sum(
            row[2] is not None and row[2]["class_name"] == class_name
            for row in class_selected
        )
        any_match = sum(row[2] is not None for row in class_selected)
        confidences = [
            row[2]["confidence"]
            for row in class_selected
            if row[2] is not None and row[2]["class_name"] == class_name
        ]
        class_rows[class_name] = {
            "images": len(class_selected),
            "correct_class_matches": correct,
            "correct_class_recall": correct / len(class_selected),
            "any_class_matches": any_match,
            "median_correct_match_confidence": median(confidences) if confidences else None,
        }

    correct = sum(
        best is not None and best["class_name"] == record["class_name"]
        for record, _, best in selected
    )
    any_match = sum(best is not None for _, _, best in selected)
    correct_confidences = [
        best["confidence"]
        for record, _, best in selected
        if best is not None and best["class_name"] == record["class_name"]
    ]
    unmatched = sum(
        sum(detection["iou"] < match_iou for detection in detections)
        for _, detections, _ in selected
    )
    return {
        "confidence_threshold": threshold,
        "reviewed_source_images": len(selected),
        "foreground_correct_class_matches": correct,
        "foreground_correct_class_recall": correct / len(selected),
        "foreground_any_class_matches": any_match,
        "foreground_any_class_recall": any_match / len(selected),
        "median_correct_match_confidence": median(correct_confidences) if correct_confidences else None,
        "total_detections": sum(len(detections) for _, detections, _ in selected),
        "unmatched_detections_not_scored_as_false_positives": unmatched,
        "per_class": class_rows,
    }


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model.resolve()))
    names = {int(key): value for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise SystemExit(f"unexpected class map: {names}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for source in manifest["labeled_records"]:
        source_path = (args.repo_root / source["source_path"]).resolve()
        x, y, width, height = source["source_bbox_xywh"]
        expected_box = [x, y, x + width, y + height]
        result = model.predict(
            str(source_path),
            conf=args.min_confidence,
            iou=0.45,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": names[int(class_id)],
                        "confidence": float(confidence),
                        "bbox_xyxy": box,
                        "iou": iou_xyxy(box, expected_box),
                    }
                )
        records.append(
            {
                "record_id": source["record_id"],
                "source_path": source["source_path"],
                "class_id": source["class_id"],
                "class_name": source["class_name"],
                "reviewed_foreground_bbox_xyxy": expected_box,
                "detections": detections,
            }
        )

    report = {
        "schema_version": "1.0",
        "status": "development_fit_only_no_independent_validation",
        "warning": (
            "Same capture-session sources contributed crops to training. "
            "Metrics are resubstitution diagnostics, not held-out accuracy."
        ),
        "background_annotation_policy": (
            "Only the deliberately presented foreground chip was reviewed. "
            "Unmatched detections may be real background chips and are not scored as false positives."
        ),
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "match_iou": args.match_iou,
        "threshold_summaries": [
            summarize(records, threshold, args.match_iou)
            for threshold in (0.10, 0.25, 0.50, 0.75)
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["threshold_summaries"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
