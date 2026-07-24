"""Evaluate the separate 10/20 outer-rim-colour classifier on reviewed boxes."""

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

from chip_rim_color_value import (  # noqa: E402
    RimColourBinaryClassifier,
    recognize_chip_rim_colour,
)
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
        "--binary-model",
        type=Path,
        default=ROOT
        / "models/assets/chip_recognition/rim-colour-binary-10-20-v1/model.json",
    )
    parser.add_argument(
        "--library",
        action="append",
        type=Path,
        dest="libraries",
        help="Repeat to compare multiple template colour-signature libraries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/chip_evaluation/chip-v2-rim-color-binary-v1",
    )
    return parser.parse_args()


def evaluate(
    matcher,
    source: str,
    raw_root: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
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
            seed = int(
                hashlib.sha256(f"{record['relative_path']}:{box}".encode()).hexdigest()[:8],
                16,
            )
            cv2.setRNGSeed(seed & 0x7FFFFFFF)
            observation = recognize_chip_rim_colour(matcher, image, box)
            details.append(
                {
                    "relative_path": record["relative_path"],
                    "expected": expected,
                    "predicted": observation.denomination,
                    "accepted": observation.accepted,
                    "correct": bool(
                        observation.accepted
                        and observation.denomination == expected
                    ),
                    "score": observation.score,
                    "margin": observation.margin,
                    "rejection_reason": observation.rejection_reason,
                    "ellipse_aspect_ratio": observation.ellipse_aspect_ratio,
                    "ellipse_minor_axis_px": observation.ellipse_minor_axis_px,
                    "latency_ms": observation.latency_ms,
                }
            )
    accepted = [item for item in details if item["accepted"]]
    correct = [item for item in details if item["correct"]]
    wrong = [item for item in accepted if not item["correct"]]
    per_value = {}
    for value in (10, 20):
        subset = [item for item in details if item["expected"] == value]
        per_value[str(value)] = {
            "total": len(subset),
            "accepted": sum(bool(item["accepted"]) for item in subset),
            "correct": sum(bool(item["correct"]) for item in subset),
            "wrong": sum(
                bool(item["accepted"] and not item["correct"]) for item in subset
            ),
        }
    return {
        "source": source,
        "total": len(details),
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong": len(wrong),
        "rejected": len(details) - len(accepted),
        "accepted_accuracy": len(correct) / len(accepted) if accepted else 0.0,
        "overall_correct_rate": len(correct) / len(details) if details else 0.0,
        "mean_latency_ms": statistics.fmean(item["latency_ms"] for item in details),
        "per_value": per_value,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    libraries = args.libraries or [
        ROOT / "models/assets/chip_recognition/las-vegas-denomination-templates-v1",
        ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/selected_denomination_library",
        ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/denomination_library/library",
    ]
    records = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    results = [
        evaluate(
            RimColourBinaryClassifier(
                libraries[0],
                allowed_denominations=(10, 20),
                model_path=args.binary_model,
            ),
            str(args.binary_model.resolve()),
            args.raw_root,
            records,
        )
    ]
    results.extend(
        evaluate(
            ChipTemplateMatcher(library, allowed_denominations=(10, 20)),
            str(library.resolve()),
            args.raw_root,
            records,
        )
        for library in libraries
    )
    report = {
        "schema_version": "1.0",
        "evaluation_id": "chip-v2-rim-colour-binary-20260724",
        "active_denominations": [10, 20],
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    key: result[key]
                    for key in (
                        "source",
                        "accepted",
                        "correct",
                        "wrong",
                        "rejected",
                        "accepted_accuracy",
                        "overall_correct_rate",
                        "mean_latency_ms",
                    )
                }
                for result in results
            ],
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
