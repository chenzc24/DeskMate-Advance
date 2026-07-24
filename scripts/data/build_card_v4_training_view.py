"""Build the V4 replay-plus-rotation source-isolated training view."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.augment_poker_cards import load_boxes, sha256_file
from scripts.data.build_card_retrain_v2 import (
    _load_classes,
    _write_list,
    _write_yaml,
)


DEFAULT_BASE = ROOT / "data" / "work" / "poker_big_data_v3" / "dataset"
DEFAULT_ROTATION = (
    ROOT / "data" / "work" / "poker_big_data_v3" / "rotation_only_520"
)
DEFAULT_OUTPUT = ROOT / "data" / "work" / "poker_big_data_v4" / "dataset"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "poker-dealer-v3"
    / "model.classes.json"
)


def split_rotation_records(
    records: Sequence[dict[str, object]],
    validation_class_ids: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [
        record
        for record in records
        if int(record["class_id"]) not in validation_class_ids
    ]
    validation = [
        record
        for record in records
        if int(record["class_id"]) in validation_class_ids
    ]
    train_sources = {str(record["source_image_sha256"]) for record in train}
    validation_sources = {
        str(record["source_image_sha256"]) for record in validation
    }
    if train_sources & validation_sources:
        raise ValueError("rotation siblings cross train/validation")
    return train, validation


def _read_paths(path: Path) -> list[Path]:
    values = [
        Path(value).resolve()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate image path in {path}")
    return values


def build_v4_view(
    base_root: Path,
    rotation_root: Path,
    output_root: Path,
    class_names: Sequence[str],
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    base_manifest_path = base_root / "manifest.json"
    rotation_manifest_path = rotation_root / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    rotation_manifest = json.loads(
        rotation_manifest_path.read_text(encoding="utf-8")
    )
    if base_manifest["counts"]["combined_train"] != 2125:
        raise ValueError("unexpected V3 base training count")
    if base_manifest["counts"]["combined_validation"] != 375:
        raise ValueError("unexpected V3 base validation count")
    if rotation_manifest["summary"]["images"] != 520:
        raise ValueError("rotation pool must contain exactly 520 images")
    if list(rotation_manifest["classes"]) != list(class_names):
        raise ValueError("rotation class order differs from V3")

    validation_ids = {
        int(value)
        for value in base_manifest["local_source_split"][
            "validation_class_ids"
        ]
    }
    rotation_train, rotation_validation = split_rotation_records(
        rotation_manifest["records"],
        validation_ids,
    )
    if len(rotation_train) != 440 or len(rotation_validation) != 80:
        raise ValueError("rotation source split must be 440 train and 80 validation")
    per_class = Counter(int(record["class_id"]) for record in rotation_manifest["records"])
    if per_class != Counter({class_id: 10 for class_id in range(52)}):
        raise ValueError("rotation pool must contain ten variants per class")

    base_source_split = {
        str(record["image_sha256"]): str(record["split"])
        for record in base_manifest["local_source_split"]["records"]
    }
    for record in rotation_train:
        if base_source_split.get(str(record["source_image_sha256"])) != "train":
            raise ValueError("rotation train source differs from V3 source split")
    for record in rotation_validation:
        if (
            base_source_split.get(str(record["source_image_sha256"]))
            != "validation"
        ):
            raise ValueError("rotation validation source differs from V3 source split")

    rotation_images = rotation_root / "images" / "train"
    rotation_labels = rotation_root / "labels" / "train"
    for record in rotation_manifest["records"]:
        image_path = rotation_images / str(record["image"])
        label_path = rotation_labels / str(record["label"])
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing rotation pair: {record['image']}")
        boxes = load_boxes(label_path, len(class_names))
        if len(boxes) != 2:
            raise ValueError(f"rotation label must contain two corners: {label_path}")

    base_train = _read_paths(base_root / "train.txt")
    base_validation = _read_paths(base_root / "val.txt")
    train_paths = [
        *base_train,
        *(
            (rotation_images / str(record["image"])).resolve()
            for record in rotation_train
        ),
    ]
    validation_paths = [
        *base_validation,
        *(
            (rotation_images / str(record["image"])).resolve()
            for record in rotation_validation
        ),
    ]
    if len(train_paths) != len(set(train_paths)):
        raise ValueError("duplicate V4 training image path")
    if len(validation_paths) != len(set(validation_paths)):
        raise ValueError("duplicate V4 validation image path")
    overlap = sorted(set(train_paths) & set(validation_paths))
    if overlap:
        raise ValueError(f"V4 train/validation path overlap: {overlap[:5]}")

    output_root.mkdir(parents=True, exist_ok=True)
    train_list = output_root / "train.txt"
    validation_list = output_root / "val.txt"
    _write_list(train_list, train_paths)
    _write_list(validation_list, validation_paths)
    dataset_yaml = output_root / "dataset.yaml"
    _write_yaml(dataset_yaml, train_list, validation_list, class_names)
    counts = {
        "base_train": len(base_train),
        "base_validation": len(base_validation),
        "rotation_train": len(rotation_train),
        "rotation_validation": len(rotation_validation),
        "combined_train": len(train_paths),
        "combined_validation": len(validation_paths),
        "combined_total": len(train_paths) + len(validation_paths),
    }
    expected_counts = {
        "base_train": 2125,
        "base_validation": 375,
        "rotation_train": 440,
        "rotation_validation": 80,
        "combined_train": 2565,
        "combined_validation": 455,
        "combined_total": 3020,
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected V4 counts: {counts}")
    manifest = {
        "schema_version": "poker_dealer.card_training_view.v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_source_isolated_rotation_replay",
        "classes": list(class_names),
        "counts": counts,
        "split": {
            "train_fraction": counts["combined_train"] / counts["combined_total"],
            "validation_fraction": (
                counts["combined_validation"] / counts["combined_total"]
            ),
            "unit": "source image before augmentation",
            "validation_class_ids": sorted(validation_ids),
            "validation_class_names": [
                class_names[class_id] for class_id in sorted(validation_ids)
            ],
            "image_path_overlap": [],
            "rotation_source_hash_overlap": [],
        },
        "base": {
            "root": str(base_root.resolve()),
            "manifest_sha256": sha256_file(base_manifest_path),
            "train_list_sha256": sha256_file(base_root / "train.txt"),
            "validation_list_sha256": sha256_file(base_root / "val.txt"),
        },
        "rotation": {
            "root": str(rotation_root.resolve()),
            "manifest_sha256": sha256_file(rotation_manifest_path),
            "angles_degrees": rotation_manifest["contract"]["angles_degrees"],
            "mirror": False,
            "source_card_count": 1,
            "output_card_count": 1,
        },
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "train_list_sha256": sha256_file(train_list),
        "validation_list_sha256": sha256_file(validation_list),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--rotation", type=Path, default=DEFAULT_ROTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_v4_view(
        args.base.resolve(),
        args.rotation.resolve(),
        args.output.resolve(),
        _load_classes(args.classes.resolve()),
    )
    print(
        json.dumps(
            {"counts": manifest["counts"], "split": manifest["split"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
