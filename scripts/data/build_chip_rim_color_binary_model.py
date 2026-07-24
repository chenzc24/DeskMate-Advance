"""Fit the fixed-design 10/20 rim-colour classifier from reviewed chip-v2 boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "chip_recognition_workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chip_rim_color_value import extract_rim_colour_evidence  # noqa: E402


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
        "--output",
        type=Path,
        default=ROOT
        / "models/assets/chip_recognition/rim-colour-binary-10-20-v1/model.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    records = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    rows: list[dict[str, object]] = []
    for record in records:
        image = cv2.imread(str(args.raw_root / record["relative_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable source: {record['relative_path']}")
        for instance in record["instances"]:
            box = tuple(round(float(value)) for value in instance["box_xyxy"])
            seed = int(
                hashlib.sha256(f"{record['relative_path']}:{box}".encode()).hexdigest()[:8],
                16,
            )
            cv2.setRNGSeed(seed & 0x7FFFFFFF)
            evidence, rejection = extract_rim_colour_evidence(
                image,
                box,
                minimum_minor_axis_px=24.0,
                minimum_aspect_ratio=0.25,
            )
            if evidence is None:
                rows.append(
                    {
                        "relative_path": record["relative_path"],
                        "capture_group": record["capture_group"],
                        "denomination": int(instance["denomination"]),
                        "rejection": rejection,
                    }
                )
                continue
            rows.append(
                {
                    "relative_path": record["relative_path"],
                    "capture_group": record["capture_group"],
                    "denomination": int(instance["denomination"]),
                    "feature": evidence.pattern_feature.tolist(),
                    "rejection": None,
                }
            )

    usable = [row for row in rows if row.get("feature") is not None]
    training = [row for row in usable if row["capture_group"] != "chip_v2:20"]
    holdout = [row for row in usable if row["capture_group"] == "chip_v2:20"]
    x_train = np.asarray([row["feature"] for row in training], dtype=np.float32)
    y_train = np.asarray(
        [1 if row["denomination"] == 20 else 0 for row in training], dtype=np.int64
    )
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        random_state=20260724,
        max_iter=2000,
    ).fit(scaler.transform(x_train), y_train)

    def metrics(rows_to_score: list[dict[str, object]]) -> dict[str, object]:
        if not rows_to_score:
            return {"count": 0}
        x = np.asarray([row["feature"] for row in rows_to_score], dtype=np.float32)
        expected = np.asarray(
            [1 if row["denomination"] == 20 else 0 for row in rows_to_score],
            dtype=np.int64,
        )
        probabilities = classifier.predict_proba(scaler.transform(x))[:, 1]
        predicted = (probabilities >= 0.5).astype(np.int64)
        matrix = confusion_matrix(expected, predicted, labels=(0, 1))
        return {
            "count": len(rows_to_score),
            "correct": int(np.count_nonzero(predicted == expected)),
            "accuracy": float(np.mean(predicted == expected)),
            "confusion_10_20": matrix.tolist(),
            "minimum_confidence": float(
                np.min(np.maximum(probabilities, 1.0 - probabilities))
            ),
        }

    independent_metrics = {
        "fit_groups": metrics(training),
        "heldout_chip_v2_20": metrics(holdout),
    }

    # The runtime development model consumes all available fixed-design colour
    # examples. Independent evidence above is retained from the pre-refit model;
    # a new capture session is still required for promotion.
    x_all = np.asarray([row["feature"] for row in usable], dtype=np.float32)
    y_all = np.asarray(
        [1 if row["denomination"] == 20 else 0 for row in usable], dtype=np.int64
    )
    scaler = StandardScaler().fit(x_all)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=20260724,
        max_iter=2000,
    ).fit(scaler.transform(x_all), y_all)

    payload = {
        "schema_version": "1.0",
        "model_id": "chip-rim-colour-binary-fixed-design",
        "version": "v1-20260724",
        "state": "development",
        "active_denominations": [10, 20],
        "class_contract": {
            "10": "alternating blue and flesh-coloured outer rim",
            "20": "alternating green and dark-green outer rim",
        },
        "feature_contract": (
            "40D saturation-weighted hue histogram, HSV/Lab distributions and "
            "blue/green/flesh/dark-green coverage over the fitted outer annulus"
        ),
        "feature_count": int(x_train.shape[1]),
        "source_annotation_sha256": sha256(args.annotations),
        "training_policy": (
            "record independent metrics by holding out complete chip_v2:20, then "
            "refit the development runtime classifier on all reviewed fixed-design "
            "10/20 colour examples"
        ),
        "feature_standardization": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "logistic_regression": {
            "positive_class": 20,
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "C": 1.0,
            "class_weight": "balanced",
        },
        "metrics": {
            "pre_refit_independent": independent_metrics,
            "runtime_refit_in_sample": metrics(usable),
            "extraction_rejections": len(rows) - len(usable),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
