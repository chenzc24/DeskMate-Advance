"""Build a compact, source-isolated card retraining dataset.

The resulting view contains exactly 1,000 locally derived images and 1,000
external images. Local source images are split before augmentation, so no
augmentation siblings cross the train/validation boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.augment_poker_cards import (
    AugmentConfig,
    SourceItem,
    discover_sources,
    generate_dataset,
    load_boxes,
    validate_dataset,
)
from scripts.data.build_card_training_view import hardlink_same_volume

DEFAULT_OLD_LOCAL = ROOT / "data" / "raw" / "poker_label" / "dataset"
DEFAULT_NEW_LOCAL = ROOT / "data" / "raw" / "poker_label" / "poker new data"
DEFAULT_EXTERNAL_VIEW = ROOT / "data" / "work" / "card_finetune_v1" / "external"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "card_retrain_v2"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "lgd-cards-gen3"
    / "model.classes.json"
)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _load_classes(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 52:
        raise ValueError("expected the pinned 52-class JSON list")
    return [str(item) for item in value]


def _items_by_class(items: Sequence[SourceItem]) -> dict[int, SourceItem]:
    result: dict[int, SourceItem] = {}
    for item in items:
        class_id = item.boxes[0].class_id
        if class_id in result:
            raise ValueError(f"duplicate local source class {class_id}")
        result[class_id] = item
    return result


def _link_source(item: SourceItem, destination: Path, prefix: str) -> None:
    image_name = f"{prefix}_{item.image_path.name}"
    label_name = f"{Path(image_name).stem}.txt"
    hardlink_same_volume(item.image_path, destination / "images" / image_name)
    hardlink_same_volume(item.label_path, destination / "labels" / label_name)


def split_local_sources(
    old_root: Path,
    new_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
) -> dict[str, object]:
    old_items = _items_by_class(
        discover_sources(old_root / "images", old_root / "labels", len(class_names))
    )
    new_items = _items_by_class(
        discover_sources(new_root, new_root / "labels", len(class_names))
    )
    expected = set(range(len(class_names)))
    if set(old_items) != expected or set(new_items) != expected:
        raise ValueError("both local source sets must cover all 52 classes")

    rng = np.random.default_rng(seed)
    shuffled = [int(value) for value in rng.permutation(len(class_names))]
    old_train_classes = set(shuffled[: len(class_names) // 2])
    records: list[dict[str, object]] = []
    train_hashes: set[str] = set()
    val_hashes: set[str] = set()
    for class_id in range(len(class_names)):
        old_item = old_items[class_id]
        new_item = new_items[class_id]
        if class_id in old_train_classes:
            train_item, train_origin = old_item, "old"
            val_item, val_origin = new_item, "new"
        else:
            train_item, train_origin = new_item, "new"
            val_item, val_origin = old_item, "old"
        _link_source(train_item, output_root / "train", train_origin)
        _link_source(val_item, output_root / "val", val_origin)
        train_hashes.add(train_item.image_sha256)
        val_hashes.add(val_item.image_sha256)
        records.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "train_origin": train_origin,
                "train_image": train_item.image_path.name,
                "train_sha256": train_item.image_sha256,
                "validation_origin": val_origin,
                "validation_image": val_item.image_path.name,
                "validation_sha256": val_item.image_sha256,
            }
        )
    overlap = sorted(train_hashes & val_hashes)
    if overlap:
        raise ValueError(f"local source leakage detected: {overlap}")
    return {
        "seed": seed,
        "strategy": "per-class one-source-each with balanced old/new origins",
        "train_sources": len(train_hashes),
        "validation_sources": len(val_hashes),
        "train_old": sum(record["train_origin"] == "old" for record in records),
        "train_new": sum(record["train_origin"] == "new" for record in records),
        "validation_old": sum(
            record["validation_origin"] == "old" for record in records
        ),
        "validation_new": sum(
            record["validation_origin"] == "new" for record in records
        ),
        "source_sha256_overlap": overlap,
        "records": records,
    }


def _external_records(
    external_root: Path, class_count: int, seed: int
) -> list[dict[str, object]]:
    images_dir = external_root / "images" / "train"
    labels_dir = external_root / "labels" / "train"
    records: list[dict[str, object]] = []
    candidates = sorted(
        images_dir.glob("*.jpg"),
        key=lambda path: _stable_key(seed, path.name),
    )[:3000]
    for image_path in candidates:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        class_ids = sorted({box.class_id for box in load_boxes(label_path, class_count)})
        records.append(
            {
                "image": image_path,
                "label": label_path,
                "class_ids": class_ids,
            }
        )
    if len(records) < 1000:
        raise ValueError("external view must contain at least 1,000 images")
    return records


def _select_with_coverage(
    records: Sequence[dict[str, object]],
    count: int,
    class_count: int,
    minimum_per_class: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_names: set[str] = set()
    counts: Counter[int] = Counter()
    for record in records:
        class_ids = [int(value) for value in record["class_ids"]]
        if any(counts[class_id] < minimum_per_class for class_id in class_ids):
            selected.append(record)
            selected_names.add(str(record["image"]))
            counts.update(class_ids)
            if all(counts[class_id] >= minimum_per_class for class_id in range(class_count)):
                break
    for record in records:
        if len(selected) >= count:
            break
        name = str(record["image"])
        if name not in selected_names:
            selected.append(record)
            selected_names.add(name)
            counts.update(int(value) for value in record["class_ids"])
    if len(selected) != count:
        raise ValueError(f"could not select {count} external images")
    missing = [
        class_id
        for class_id in range(class_count)
        if counts[class_id] < minimum_per_class
    ]
    if missing:
        raise ValueError(f"external selection lacks class coverage: {missing}")
    return selected


def split_external(
    external_root: Path,
    class_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records = _external_records(external_root, class_count, seed)
    validation = _select_with_coverage(records, 200, class_count, 3)
    validation_names = {str(record["image"]) for record in validation}
    remaining = [
        record for record in records if str(record["image"]) not in validation_names
    ]
    train = _select_with_coverage(remaining, 800, class_count, 10)
    train_names = {str(record["image"]) for record in train}
    if train_names & validation_names:
        raise ValueError("external source leakage detected")
    return train, validation


def _write_list(path: Path, image_paths: Iterable[Path]) -> None:
    values = [image.resolve().as_posix() for image in image_paths]
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _write_yaml(
    path: Path,
    train_list: Path,
    validation_list: Path,
    class_names: Sequence[str],
) -> None:
    lines = [
        f"path: {path.parent.resolve().as_posix()}",
        f"train: {train_list.resolve().as_posix()}",
        f"val: {validation_list.resolve().as_posix()}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    old_local: Path,
    new_local: Path,
    external_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
    workers: int,
    *,
    resume: bool = False,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_split = split_local_sources(
        old_local,
        new_local,
        output_root / "local_sources",
        class_names,
        seed,
    )
    local_train_root = output_root / "local_augmented_train"
    local_val_root = output_root / "local_augmented_validation"
    train_manifest_path = local_train_root / "manifest.json"
    validation_manifest_path = local_val_root / "manifest.json"
    if resume and train_manifest_path.is_file():
        local_train_manifest = json.loads(
            train_manifest_path.read_text(encoding="utf-8")
        )
    else:
        local_train_manifest = generate_dataset(
            output_root / "local_sources" / "train" / "images",
            output_root / "local_sources" / "train" / "labels",
            local_train_root,
            class_names,
            AugmentConfig(
                total_variants=800,
                profile="train",
                seed=seed,
                workers=workers,
            ),
        )
    if resume and validation_manifest_path.is_file():
        local_val_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
    else:
        local_val_manifest = generate_dataset(
            output_root / "local_sources" / "val" / "images",
            output_root / "local_sources" / "val" / "labels",
            local_val_root,
            class_names,
            AugmentConfig(
                total_variants=200,
                profile="validation",
                seed=seed + 1,
                workers=workers,
            ),
        )
    local_train_validation = validate_dataset(local_train_root)
    local_val_validation = validate_dataset(local_val_root)
    if not local_train_validation["valid"] or not local_val_validation["valid"]:
        raise ValueError("generated local data failed validation")

    external_train, external_val = split_external(
        external_root, len(class_names), seed
    )
    local_train_images = sorted(
        (local_train_root / "images" / "train").glob("*.jpg")
    )
    local_val_images = sorted(
        (local_val_root / "images" / "train").glob("*.jpg")
    )
    train_list = output_root / "train.txt"
    val_list = output_root / "val.txt"
    _write_list(
        train_list,
        [*local_train_images, *(Path(record["image"]) for record in external_train)],
    )
    _write_list(
        val_list,
        [*local_val_images, *(Path(record["image"]) for record in external_val)],
    )
    dataset_yaml = output_root / "dataset.yaml"
    _write_yaml(dataset_yaml, train_list, val_list, class_names)

    manifest: dict[str, object] = {
        "schema_version": "poker_dealer.card_retrain_view.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_source_isolated_train_validation",
        "seed": seed,
        "counts": {
            "local_total": 1000,
            "local_train": len(local_train_images),
            "local_validation": len(local_val_images),
            "external_total": 1000,
            "external_train": len(external_train),
            "external_validation": len(external_val),
            "combined_train": len(local_train_images) + len(external_train),
            "combined_validation": len(local_val_images) + len(external_val),
        },
        "local_source_split": source_split,
        "local_train_manifest_sha256": hashlib.sha256(
            (local_train_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "local_validation_manifest_sha256": hashlib.sha256(
            (local_val_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "local_train_summary": local_train_manifest["summary"],
        "local_validation_summary": local_val_manifest["summary"],
        "external": {
            "source_root": str(external_root.resolve()),
            "train_images": [Path(record["image"]).name for record in external_train],
            "validation_images": [
                Path(record["image"]).name for record in external_val
            ],
            "overlap": sorted(
                {Path(record["image"]).name for record in external_train}
                & {Path(record["image"]).name for record in external_val}
            ),
        },
        "dataset_yaml": str(dataset_yaml.resolve()),
    }
    if manifest["counts"] != {
        "local_total": 1000,
        "local_train": 800,
        "local_validation": 200,
        "external_total": 1000,
        "external_train": 800,
        "external_validation": 200,
        "combined_train": 1600,
        "combined_validation": 400,
    }:
        raise ValueError(f"unexpected dataset counts: {manifest['counts']}")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-local", type=Path, default=DEFAULT_OLD_LOCAL)
    parser.add_argument("--new-local", type=Path, default=DEFAULT_NEW_LOCAL)
    parser.add_argument("--external-view", type=Path, default=DEFAULT_EXTERNAL_VIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(
        args.old_local.resolve(),
        args.new_local.resolve(),
        args.external_view.resolve(),
        args.output.resolve(),
        _load_classes(args.classes.resolve()),
        args.seed,
        args.workers,
        resume=args.resume,
    )
    print(json.dumps({"counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
