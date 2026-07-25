"""Build the session-isolated 1,500-local plus 1,500-external V5 card view."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import cv2
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


DEFAULT_LOCAL = ROOT / "data" / "raw" / "poker_label" / "corner_case"
DEFAULT_LABELS = ROOT / "data" / "work" / "card_corner_v5" / "auto_labels"
DEFAULT_REVIEW = ROOT / "data" / "work" / "card_corner_v5" / "label_review"
DEFAULT_EXTERNAL = ROOT / "data" / "work" / "card_finetune_v1" / "external"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "card_corner_v5" / "dataset"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "lgd-cards-gen3"
    / "model.classes.json"
)
V4_SHA256 = "0204fedbe0e83d887ed370dfc7cd17dfb1da57052d7c30583b127cddd43a2fa1"
TRAIN_SESSIONS = ("heng", "slope1")
VALIDATION_SESSIONS = ("slope2",)


def _link_source(item: SourceItem, destination: Path, session: str) -> None:
    image_name = f"{session}_{item.image_path.name}"
    label_name = f"{Path(image_name).stem}.txt"
    hardlink_same_volume(item.image_path, destination / "images" / image_name)
    hardlink_same_volume(item.label_path, destination / "labels" / label_name)


def _load_reviewed_session(
    local_root: Path,
    labels_root: Path,
    review_root: Path,
    session: str,
    class_names: Sequence[str],
) -> tuple[list[SourceItem], dict[str, object]]:
    report_path = review_root / session / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    if report.get("model", {}).get("sha256") != V4_SHA256:
        raise ValueError(f"{session}: labels were not proposed by frozen V4")
    if (
        summary.get("images") != 52
        or summary.get("labels") != 52
        or summary.get("selected_boxes") != 104
        or summary.get("two_box_images") != 52
        or summary.get("zero_box_images") != 0
        or summary.get("class_coverage") != 52
    ):
        raise ValueError(f"{session}: incomplete label review summary")

    items = discover_sources(
        local_root / session,
        labels_root / session / "labels",
        len(class_names),
    )
    records = {str(record["image"]): record for record in report["records"]}
    if len(items) != 52 or len(records) != 52:
        raise ValueError(f"{session}: expected exactly 52 reviewed sources")
    class_counts: Counter[int] = Counter()
    for item in items:
        record = records.get(item.image_path.name)
        if record is None:
            raise ValueError(f"{session}: source missing from report: {item.image_path}")
        if record["image_sha256"] != item.image_sha256:
            raise ValueError(f"{session}: source changed after review: {item.image_path}")
        if record["label_sha256"] != item.label_sha256:
            raise ValueError(f"{session}: label changed after review: {item.label_path}")
        expected_id = int(record["expected_class_id"])
        if len(item.boxes) != 2 or {box.class_id for box in item.boxes} != {expected_id}:
            raise ValueError(f"{session}: reviewed label identity mismatch: {item.label_path}")
        class_counts[expected_id] += 1
    if class_counts != Counter({class_id: 1 for class_id in range(len(class_names))}):
        raise ValueError(f"{session}: expected one source for every class")
    return items, {
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "summary": summary,
    }


def split_local_sessions(
    local_root: Path,
    labels_root: Path,
    review_root: Path,
    output_root: Path,
    class_names: Sequence[str],
) -> dict[str, object]:
    sessions: dict[str, object] = {}
    train_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    records: list[dict[str, object]] = []
    for session in (*TRAIN_SESSIONS, *VALIDATION_SESSIONS):
        items, review = _load_reviewed_session(
            local_root, labels_root, review_root, session, class_names
        )
        split = "train" if session in TRAIN_SESSIONS else "validation"
        destination = output_root / ("train" if split == "train" else "val")
        for item in items:
            _link_source(item, destination, session)
            target = train_hashes if split == "train" else validation_hashes
            target.add(item.image_sha256)
            records.append(
                {
                    "session": session,
                    "split": split,
                    "image": item.image_path.name,
                    "image_sha256": item.image_sha256,
                    "label_sha256": item.label_sha256,
                    "class_id": item.boxes[0].class_id,
                    "box_count": len(item.boxes),
                }
            )
        sessions[session] = review
    overlap = sorted(train_hashes & validation_hashes)
    if overlap:
        raise ValueError(f"local source leakage detected: {overlap}")
    return {
        "strategy": "complete capture-session holdout before augmentation",
        "train_sessions": list(TRAIN_SESSIONS),
        "validation_sessions": list(VALIDATION_SESSIONS),
        "train_sources": len(train_hashes),
        "validation_sources": len(validation_hashes),
        "source_sha256_overlap": overlap,
        "review_reports": sessions,
        "records": records,
    }


def image_dhash(path: Path) -> int:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    reduced = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = reduced[:, 1:] > reduced[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _add_perceptual_hashes(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        copied = dict(record)
        copied["dhash64"] = image_dhash(Path(record["image"]))
        result.append(copied)
    return result


def split_external(
    external_root: Path,
    class_count: int,
    seed: int,
    *,
    near_duplicate_distance: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    records = _add_perceptual_hashes(_external_records(external_root, class_count, seed))
    validation = _select_with_coverage(records, 225, class_count, 3)
    validation_names = {str(record["image"]) for record in validation}
    validation_hashes = [int(record["dhash64"]) for record in validation]
    filtered: list[dict[str, object]] = []
    rejected = 0
    for record in records:
        if str(record["image"]) in validation_names:
            continue
        fingerprint = int(record["dhash64"])
        if any(
            hamming_distance(fingerprint, held_out) <= near_duplicate_distance
            for held_out in validation_hashes
        ):
            rejected += 1
            continue
        filtered.append(record)
    train = _select_with_coverage(filtered, 1275, class_count, 12)
    if {str(record["image"]) for record in train} & validation_names:
        raise ValueError("external path leakage detected")
    if min(
        hamming_distance(int(train_item["dhash64"]), int(val_item["dhash64"]))
        for train_item in train
        for val_item in validation
    ) <= near_duplicate_distance:
        raise ValueError("external perceptual-near-duplicate leakage detected")
    train_sha = {sha256_file(Path(record["image"])) for record in train}
    validation_sha = {sha256_file(Path(record["image"])) for record in validation}
    if train_sha & validation_sha:
        raise ValueError("external exact-byte leakage detected")
    return train, validation, rejected


def _class_counts(records: Sequence[dict[str, object]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(int(value) for value in record["class_ids"])
    return dict(sorted(counts.items()))


def _selection_sha256(records: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: Path(item["image"]).name):
        image = Path(record["image"])
        label = Path(record["label"])
        digest.update(image.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(image)))
        digest.update(bytes.fromhex(sha256_file(label)))
    return digest.hexdigest()


def _verify_augmentation(
    root: Path, manifest: dict[str, object], expected_images: int
) -> dict[str, object]:
    validation = validate_dataset(root)
    summary = manifest["summary"]
    if not validation["valid"] or summary["image_count"] != expected_images:
        raise ValueError(f"invalid augmented dataset: {root}")
    if set(summary["orientation_counts"]) != {
        "upright",
        "right",
        "inverted",
        "left",
    }:
        raise ValueError(f"all four orientation bins are required: {root}")
    if set(summary["scale_counts"]) != {"far", "medium", "near", "very_far"}:
        raise ValueError(f"all four distance bins are required: {root}")
    if summary["annotation_count"] != expected_images * 2:
        raise ValueError(f"augmentation changed per-image box count: {root}")
    return validation


def build_dataset(
    local_root: Path,
    labels_root: Path,
    review_root: Path,
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
    decision_path = review_root / "review_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "accepted" or decision.get("accepted_images") != 156:
        raise ValueError("all 156 proposed labels must be reviewed and accepted")
    output_root.mkdir(parents=True, exist_ok=True)

    source_split = split_local_sessions(
        local_root,
        labels_root,
        review_root,
        output_root / "local_sources",
        class_names,
    )
    local_train_root = output_root / "local_augmented_train"
    local_validation_root = output_root / "local_augmented_validation"
    train_manifest_path = local_train_root / "manifest.json"
    validation_manifest_path = local_validation_root / "manifest.json"
    if resume and train_manifest_path.is_file():
        local_train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
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
                require_all_classes=True,
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
                require_all_classes=True,
            ),
        )
    local_train_validation = _verify_augmentation(
        local_train_root, local_train_manifest, 1275
    )
    local_validation_validation = _verify_augmentation(
        local_validation_root, local_validation_manifest, 225
    )

    external_train, external_validation, near_duplicate_rejections = split_external(
        external_root, len(class_names), seed
    )
    expected_classes = set(range(len(class_names)))
    external_train_counts = _class_counts(external_train)
    external_validation_counts = _class_counts(external_validation)
    if set(external_train_counts) != expected_classes:
        raise ValueError("external training selection lacks 52-class coverage")
    if set(external_validation_counts) != expected_classes:
        raise ValueError("external validation selection lacks 52-class coverage")

    local_train_images = sorted((local_train_root / "images" / "train").glob("*.jpg"))
    local_validation_images = sorted(
        (local_validation_root / "images" / "train").glob("*.jpg")
    )
    train_list = output_root / "train.txt"
    validation_list = output_root / "val.txt"
    _write_list(
        train_list,
        [*local_train_images, *(Path(record["image"]) for record in external_train)],
    )
    _write_list(
        validation_list,
        [
            *local_validation_images,
            *(Path(record["image"]) for record in external_validation),
        ],
    )
    train_paths = set(train_list.read_text(encoding="utf-8").splitlines())
    validation_paths = set(validation_list.read_text(encoding="utf-8").splitlines())
    if train_paths & validation_paths:
        raise ValueError("combined path leakage detected")
    dataset_yaml = output_root / "dataset.yaml"
    _write_yaml(dataset_yaml, train_list, validation_list, class_names)

    counts = {
        "local_total": len(local_train_images) + len(local_validation_images),
        "local_train": len(local_train_images),
        "local_validation": len(local_validation_images),
        "external_total": len(external_train) + len(external_validation),
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
        "external_total": 1500,
        "external_train": 1275,
        "external_validation": 225,
        "combined_total": 3000,
        "combined_train": 2550,
        "combined_validation": 450,
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected dataset counts: {counts}")

    manifest: dict[str, object] = {
        "schema_version": "poker_dealer.card_training_view.v5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_session_isolated_train_validation",
        "seed": seed,
        "counts": counts,
        "split": {"train_fraction": 0.85, "validation_fraction": 0.15},
        "label_review": {
            "decision": str(decision_path.resolve()),
            "decision_sha256": sha256_file(decision_path),
            "summary": decision,
        },
        "local_source_split": source_split,
        "local_train_manifest_sha256": sha256_file(train_manifest_path),
        "local_validation_manifest_sha256": sha256_file(validation_manifest_path),
        "local_train_summary": local_train_manifest["summary"],
        "local_validation_summary": local_validation_manifest["summary"],
        "local_train_validation": local_train_validation,
        "local_validation_validation": local_validation_validation,
        "external": {
            "source_root": str(external_root.resolve()),
            "near_duplicate_metric": "64-bit difference hash",
            "near_duplicate_max_hamming_distance": 4,
            "near_duplicate_candidates_rejected_from_train": near_duplicate_rejections,
            "cross_split_near_duplicate_count": 0,
            "cross_split_exact_sha256_count": 0,
            "train_class_counts": {
                class_names[class_id]: count
                for class_id, count in external_train_counts.items()
            },
            "validation_class_counts": {
                class_names[class_id]: count
                for class_id, count in external_validation_counts.items()
            },
            "train_selection_sha256": _selection_sha256(external_train),
            "validation_selection_sha256": _selection_sha256(external_validation),
            "train_images": [Path(record["image"]).name for record in external_train],
            "validation_images": [
                Path(record["image"]).name for record in external_validation
            ],
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
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=20260725)
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
        args.labels.resolve(),
        args.review.resolve(),
        args.external.resolve(),
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
