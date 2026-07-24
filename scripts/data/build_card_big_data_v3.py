"""Build the source-isolated 1,500-local plus 1,000-external card view."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.augment_poker_cards import (
    AugmentConfig,
    SourceItem,
    discover_sources,
    generate_dataset,
    load_boxes,
    sha256_file,
    validate_dataset,
)
from scripts.data.build_card_retrain_v2 import (
    _external_records,
    _load_classes,
    _select_with_coverage,
    _write_list,
    _write_yaml,
)
from scripts.data.build_card_training_view import hardlink_same_volume


DEFAULT_LOCAL = ROOT / "data" / "raw" / "poker_label" / "new big poker"
DEFAULT_EXTERNAL = ROOT / "data" / "work" / "card_finetune_v1" / "external"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "poker_big_data_v3" / "dataset"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "poker-dealer-v2"
    / "model.classes.json"
)
DEFAULT_LABEL_REPORT = (
    ROOT / "data" / "work" / "poker_big_data_v3" / "label_review" / "report.json"
)
DEFAULT_LABEL_DECISION = (
    ROOT
    / "data"
    / "work"
    / "poker_big_data_v3"
    / "label_review"
    / "review_decision.json"
)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def select_validation_class_ids(
    class_names: Sequence[str],
    seed: int,
    *,
    per_suit: int = 2,
) -> set[int]:
    by_suit: dict[str, list[tuple[int, str]]] = {
        "C": [],
        "D": [],
        "H": [],
        "S": [],
    }
    for class_id, class_name in enumerate(class_names):
        suit = class_name[-1:]
        if suit not in by_suit:
            raise ValueError(f"unsupported card class: {class_name}")
        by_suit[suit].append((class_id, class_name))
    selected: set[int] = set()
    for suit, values in by_suit.items():
        if len(values) <= per_suit:
            raise ValueError(f"not enough {suit} classes to hold out {per_suit}")
        ordered = sorted(
            values,
            key=lambda item: _stable_key(seed, item[1]),
        )
        selected.update(class_id for class_id, _ in ordered[:per_suit])
    return selected


def _link_source(item: SourceItem, destination: Path) -> None:
    image_name = f"c{item.boxes[0].class_id:02d}_{item.image_path.name}"
    label_name = f"{Path(image_name).stem}.txt"
    hardlink_same_volume(item.image_path, destination / "images" / image_name)
    hardlink_same_volume(item.label_path, destination / "labels" / label_name)


def split_local_sources(
    local_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
) -> dict[str, object]:
    items = discover_sources(local_root, local_root / "labels", len(class_names))
    items_by_class = {item.boxes[0].class_id: item for item in items}
    if len(items_by_class) != len(items):
        raise ValueError("local source must contain exactly one image per class")
    if set(items_by_class) != set(range(len(class_names))):
        raise ValueError("local source must cover all 52 classes")

    validation_ids = select_validation_class_ids(class_names, seed)
    records: list[dict[str, object]] = []
    train_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    for class_id in range(len(class_names)):
        item = items_by_class[class_id]
        split = "validation" if class_id in validation_ids else "train"
        destination = output_root / ("val" if split == "validation" else "train")
        _link_source(item, destination)
        if split == "validation":
            validation_hashes.add(item.image_sha256)
        else:
            train_hashes.add(item.image_sha256)
        records.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "split": split,
                "image": item.image_path.name,
                "image_sha256": item.image_sha256,
                "label_sha256": item.label_sha256,
            }
        )
    overlap = sorted(train_hashes & validation_hashes)
    if overlap:
        raise ValueError(f"local source leakage detected: {overlap}")
    validation_suits = Counter(
        class_names[class_id][-1] for class_id in validation_ids
    )
    if validation_suits != Counter({"C": 2, "D": 2, "H": 2, "S": 2}):
        raise ValueError(f"unbalanced validation suits: {validation_suits}")
    return {
        "seed": seed,
        "strategy": "one source per class; two held-out classes per suit",
        "train_sources": len(train_hashes),
        "validation_sources": len(validation_hashes),
        "train_class_ids": sorted(set(range(len(class_names))) - validation_ids),
        "validation_class_ids": sorted(validation_ids),
        "validation_suit_counts": dict(sorted(validation_suits.items())),
        "source_sha256_overlap": overlap,
        "records": records,
    }


def split_external(
    external_root: Path,
    class_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records = _external_records(external_root, class_count, seed)
    validation = _select_with_coverage(
        records,
        150,
        class_count=class_count,
        minimum_per_class=2,
    )
    validation_names = {str(record["image"]) for record in validation}
    remaining = [
        record
        for record in records
        if str(record["image"]) not in validation_names
    ]
    train = _select_with_coverage(
        remaining,
        850,
        class_count=class_count,
        minimum_per_class=10,
    )
    train_names = {str(record["image"]) for record in train}
    if train_names & validation_names:
        raise ValueError("external source leakage detected")
    return train, validation


def _selection_sha256(records: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: Path(item["image"]).name):
        image = Path(record["image"])
        label = Path(record["label"])
        digest.update(image.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(image)))
        digest.update(bytes.fromhex(sha256_file(label)))
    return digest.hexdigest()


def _class_counts(records: Sequence[dict[str, object]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for record in records:
        for class_id in record["class_ids"]:
            counts[int(class_id)] += 1
    return dict(sorted(counts.items()))


def build_dataset(
    local_root: Path,
    external_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    label_report: Path,
    label_decision: Path,
    seed: int,
    workers: int,
    *,
    resume: bool = False,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    if not label_report.is_file():
        raise FileNotFoundError(f"missing model-label review report: {label_report}")
    if not label_decision.is_file():
        raise FileNotFoundError(f"missing label review decision: {label_decision}")
    label_review = json.loads(label_report.read_text(encoding="utf-8"))
    review_decision = json.loads(label_decision.read_text(encoding="utf-8"))
    if label_review["summary"]["images"] != 52:
        raise ValueError("label review must cover exactly 52 source images")
    if label_review["summary"]["zero_box_images"]:
        raise ValueError("label review contains source images without boxes")
    if (
        review_decision.get("status") != "accepted"
        or review_decision.get("accepted_images") != 52
    ):
        raise ValueError("all 52 model-proposed labels must be reviewed and accepted")
    output_root.mkdir(parents=True, exist_ok=True)

    source_split = split_local_sources(
        local_root,
        output_root / "local_sources",
        class_names,
        seed,
    )
    local_train_root = output_root / "local_augmented_train"
    local_validation_root = output_root / "local_augmented_validation"
    train_manifest_path = local_train_root / "manifest.json"
    validation_manifest_path = local_validation_root / "manifest.json"
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
                total_variants=1275,
                profile="train",
                seed=seed,
                workers=workers,
                require_all_classes=False,
            ),
        )
    if resume and validation_manifest_path.is_file():
        local_validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
    else:
        local_validation_manifest = generate_dataset(
            output_root / "local_sources" / "val" / "images",
            output_root / "local_sources" / "val" / "labels",
            local_validation_root,
            class_names,
            AugmentConfig(
                total_variants=225,
                profile="validation",
                seed=seed + 1,
                workers=workers,
                require_all_classes=False,
            ),
        )

    local_train_validation = validate_dataset(local_train_root)
    local_validation_validation = validate_dataset(local_validation_root)
    if not local_train_validation["valid"]:
        raise ValueError(f"invalid local training data: {local_train_validation}")
    if not local_validation_validation["valid"]:
        raise ValueError(
            f"invalid local validation data: {local_validation_validation}"
        )
    for summary in (
        local_train_manifest["summary"],
        local_validation_manifest["summary"],
    ):
        if set(summary["orientation_counts"]) != {
            "upright",
            "right",
            "inverted",
            "left",
        }:
            raise ValueError("augmentation does not cover all four orientations")

    external_train, external_validation = split_external(
        external_root,
        len(class_names),
        seed,
    )
    external_train_counts = _class_counts(external_train)
    external_validation_counts = _class_counts(external_validation)
    expected_classes = set(range(len(class_names)))
    if set(external_train_counts) != expected_classes:
        raise ValueError("external training selection lacks 52-class coverage")
    if set(external_validation_counts) != expected_classes:
        raise ValueError("external validation selection lacks 52-class coverage")

    local_train_images = sorted(
        (local_train_root / "images" / "train").glob("*.jpg")
    )
    local_validation_images = sorted(
        (local_validation_root / "images" / "train").glob("*.jpg")
    )
    train_list = output_root / "train.txt"
    validation_list = output_root / "val.txt"
    _write_list(
        train_list,
        [
            *local_train_images,
            *(Path(record["image"]) for record in external_train),
        ],
    )
    _write_list(
        validation_list,
        [
            *local_validation_images,
            *(Path(record["image"]) for record in external_validation),
        ],
    )
    train_paths = {
        Path(value).resolve()
        for value in train_list.read_text(encoding="utf-8").splitlines()
        if value
    }
    validation_paths = {
        Path(value).resolve()
        for value in validation_list.read_text(encoding="utf-8").splitlines()
        if value
    }
    if train_paths & validation_paths:
        raise ValueError("combined train/validation image leakage detected")

    dataset_yaml = output_root / "dataset.yaml"
    _write_yaml(dataset_yaml, train_list, validation_list, class_names)
    counts = {
        "local_total": 1500,
        "local_train": len(local_train_images),
        "local_validation": len(local_validation_images),
        "external_total": 1000,
        "external_train": len(external_train),
        "external_validation": len(external_validation),
        "combined_total": len(train_paths) + len(validation_paths),
        "combined_train": len(train_paths),
        "combined_validation": len(validation_paths),
    }
    expected_counts = {
        "local_total": 1500,
        "local_train": 1275,
        "local_validation": 225,
        "external_total": 1000,
        "external_train": 850,
        "external_validation": 150,
        "combined_total": 2500,
        "combined_train": 2125,
        "combined_validation": 375,
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected dataset counts: {counts}")

    manifest: dict[str, object] = {
        "schema_version": "poker_dealer.card_training_view.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_source_isolated_train_validation",
        "seed": seed,
        "split": {
            "train_fraction": counts["combined_train"] / counts["combined_total"],
            "validation_fraction": (
                counts["combined_validation"] / counts["combined_total"]
            ),
        },
        "counts": counts,
        "model_label_review": {
            "path": str(label_report.resolve()),
            "sha256": sha256_file(label_report),
            "summary": label_review["summary"],
            "decision_path": str(label_decision.resolve()),
            "decision_sha256": sha256_file(label_decision),
            "decision": review_decision,
        },
        "local_source_split": source_split,
        "local_train_manifest_sha256": sha256_file(train_manifest_path),
        "local_validation_manifest_sha256": sha256_file(validation_manifest_path),
        "local_train_summary": local_train_manifest["summary"],
        "local_validation_summary": local_validation_manifest["summary"],
        "local_validation": local_validation_validation,
        "local_train_validation": local_train_validation,
        "external": {
            "source_root": str(external_root.resolve()),
            "train_images": [Path(record["image"]).name for record in external_train],
            "validation_images": [
                Path(record["image"]).name for record in external_validation
            ],
            "train_class_counts": {
                class_names[class_id]: count
                for class_id, count in external_train_counts.items()
            },
            "validation_class_counts": {
                class_names[class_id]: count
                for class_id, count in external_validation_counts.items()
            },
            "train_selection_sha256": _selection_sha256(external_train),
            "validation_selection_sha256": _selection_sha256(
                external_validation
            ),
            "overlap": sorted(
                {Path(record["image"]).name for record in external_train}
                & {
                    Path(record["image"]).name
                    for record in external_validation
                }
            ),
        },
        "dataset_yaml": str(dataset_yaml.resolve()),
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
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument(
        "--label-report",
        type=Path,
        default=DEFAULT_LABEL_REPORT,
    )
    parser.add_argument(
        "--label-decision",
        type=Path,
        default=DEFAULT_LABEL_DECISION,
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(
        args.local.resolve(),
        args.external.resolve(),
        args.output.resolve(),
        _load_classes(args.classes.resolve()),
        args.label_report.resolve(),
        args.label_decision.resolve(),
        args.seed,
        args.workers,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "split": manifest["split"],
                "local_validation_class_ids": manifest[
                    "local_source_split"
                ]["validation_class_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
