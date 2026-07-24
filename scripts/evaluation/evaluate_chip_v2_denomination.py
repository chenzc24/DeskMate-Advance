"""Evaluate 10/20 denomination matching on reviewed oblique chip-v2 crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "chip_recognition_workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chip_live_value import recognize_chip_value  # noqa: E402
from chip_template_matcher import ChipTemplateMatcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/chips/2026-07-24-chip-v2-source",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/reviewed_annotations_candidate.json",
    )
    parser.add_argument(
        "--baseline-library",
        type=Path,
        default=ROOT
        / "models/assets/chip_recognition/las-vegas-denomination-templates-v1",
    )
    parser.add_argument(
        "--candidate-library",
        type=Path,
        default=ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/denomination_library/library",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/chip_evaluation/chip-v2-optimization-v1/denomination",
    )
    return parser.parse_args()


def evaluate(library: Path, raw_root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    matcher = ChipTemplateMatcher(library, allowed_denominations=(10, 20))
    details: list[dict[str, object]] = []
    for record in records:
        if str(record["capture_group"]).startswith("straight:"):
            continue
        image = cv2.imread(str(raw_root / record["relative_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable source: {record['relative_path']}")
        for instance in record["instances"]:
            expected = int(instance["denomination"])
            box = tuple(round(float(value)) for value in instance["box_xyxy"])
            seed_material = f"{record['relative_path']}:{box}".encode()
            cv2.setRNGSeed(
                int(hashlib.sha256(seed_material).hexdigest()[:8], 16) & 0x7FFFFFFF
            )
            observation = recognize_chip_value(matcher, image, box)
            details.append(
                {
                    "relative_path": record["relative_path"],
                    "capture_group": record["capture_group"],
                    "expected": expected,
                    "predicted": observation.denomination,
                    "accepted": observation.accepted,
                    "correct": observation.accepted and observation.denomination == expected,
                    "score": observation.score,
                    "margin": observation.margin,
                    "rejection_reason": observation.rejection_reason,
                    "decision_reason": observation.decision_reason,
                    "ellipse_aspect_ratio": observation.ellipse_aspect_ratio,
                    "ellipse_minor_axis_px": observation.ellipse_minor_axis_px,
                    "raw_color_denomination": observation.raw_color_denomination,
                    "digit_denomination": observation.digit_denomination,
                    "latency_ms": observation.latency_ms,
                }
            )
    total = len(details)
    accepted = [item for item in details if item["accepted"]]
    correct = [item for item in details if item["correct"]]
    wrong = [item for item in accepted if not item["correct"]]
    per_value = {}
    for denomination in (10, 20):
        subset = [item for item in details if item["expected"] == denomination]
        per_value[str(denomination)] = {
            "total": len(subset),
            "accepted": sum(bool(item["accepted"]) for item in subset),
            "correct": sum(bool(item["correct"]) for item in subset),
            "wrong": sum(bool(item["accepted"] and not item["correct"]) for item in subset),
        }
    reasons: dict[str, int] = {}
    for item in details:
        reason = str(item["rejection_reason"] or "accepted")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "library": str(library.resolve()),
        "active_denominations": [10, 20],
        "total": total,
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong": len(wrong),
        "rejected": total - len(accepted),
        "accepted_accuracy": len(correct) / len(accepted) if accepted else 0.0,
        "overall_correct_rate": len(correct) / total if total else 0.0,
        "mean_latency_ms": statistics.fmean(item["latency_ms"] for item in details),
        "per_value": per_value,
        "rejection_reasons": reasons,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    records = annotations["records"]
    baseline = evaluate(args.baseline_library, args.raw_root, records)
    candidate = evaluate(args.candidate_library, args.raw_root, records)
    report = {
        "schema_version": "1.0",
        "evaluation_id": "chip-v2-denomination-oblique-20260724",
        "policy": "straight captures build candidate templates; only oblique captures are evaluated",
        "baseline": baseline,
        "candidate": candidate,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": {
                    key: baseline[key]
                    for key in ("total", "accepted", "correct", "wrong", "rejected", "overall_correct_rate")
                },
                "candidate": {
                    key: candidate[key]
                    for key in ("total", "accepted", "correct", "wrong", "rejected", "overall_correct_rate")
                },
                "report": str(path.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
