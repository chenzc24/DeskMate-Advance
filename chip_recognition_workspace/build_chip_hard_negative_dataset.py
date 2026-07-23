"""Add one target-camera hard-negative session to a YOLO localization snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _split_files(root: Path, split: str) -> tuple[list[Path], list[Path]]:
    images = sorted(
        path
        for path in (root / split / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted((root / split / "labels").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise ValueError(f"{split} image/label stems do not match")
    return images, labels


def build_dataset(
    base: Path,
    negatives: Path,
    output: Path,
    *,
    repeats: int,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    base_yaml = base / "data.yaml"
    base_manifest = base / "dataset_manifest.json"
    if not base_yaml.is_file() or not base_manifest.is_file():
        raise FileNotFoundError("base dataset YAML or manifest is missing")
    config = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    if config.get("names") != {0: "poker_chip"}:
        raise ValueError(f"unexpected base class map: {config.get('names')}")

    negative_images = sorted(
        path
        for path in negatives.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not negative_images:
        raise ValueError(f"no negative images found in: {negatives}")

    output.mkdir(parents=True)
    copy_modes: Counter[str] = Counter()
    split_image_counts: dict[str, int] = {}
    for split in SPLITS:
        output_images = output / split / "images"
        output_labels = output / split / "labels"
        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)
        images, labels = _split_files(base, split)
        for source in images:
            copy_modes[_link_or_copy(source, output_images / source.name)] += 1
        for source in labels:
            copy_modes[_link_or_copy(source, output_labels / source.name)] += 1
        split_image_counts[split] = len(images)

    negative_records: list[dict[str, object]] = []
    train_images = output / "train" / "images"
    train_labels = output / "train" / "labels"
    for index, source in enumerate(negative_images, start=1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read negative image: {source}")
        source_hash = sha256(source)
        output_names: list[str] = []
        for repeat in range(repeats):
            stem = f"target_negative_s01_{index:04d}_r{repeat:02d}_{source.stem}"
            destination = train_images / f"{stem}{source.suffix.lower()}"
            copy_modes[_link_or_copy(source, destination)] += 1
            (train_labels / f"{stem}.txt").write_text("", encoding="utf-8")
            output_names.append(destination.name)
        negative_records.append(
            {
                "source_image": str(source.resolve()),
                "source_image_sha256": source_hash,
                "width": image.shape[1],
                "height": image.shape[0],
                "source_group": "target-camera-negative-20260723-session01",
                "split": "train",
                "instances": 0,
                "output_images": output_names,
            }
        )
    split_image_counts["train"] += len(negative_images) * repeats

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "names:",
                "  0: poker_chip",
                "",
            ]
        ),
        encoding="utf-8",
    )
    capture_manifest = negatives / "capture_manifest.jsonl"
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "poker-chip-localization-hard-negative-20260723-v1",
        "status": "development_single_negative_session_train_only",
        "base_dataset": str(base.resolve()),
        "base_manifest_sha256": sha256(base_manifest),
        "class_map": {"0": "poker_chip"},
        "split_policy": (
            "base split is preserved exactly; the complete target-camera negative "
            "capture session is restricted to train; repeated entries are controlled "
            "training exposure and never independent samples"
        ),
        "negative_source": str(negatives.resolve()),
        "negative_capture_manifest_sha256": (
            sha256(capture_manifest) if capture_manifest.is_file() else None
        ),
        "negative_unique_images": len(negative_images),
        "negative_repeat_factor": repeats,
        "negative_training_entries": len(negative_images) * repeats,
        "split_image_counts": split_image_counts,
        "copy_modes": dict(copy_modes),
        "limitations": [
            "The negative images are one target-camera session and have no held-out split.",
            "Negative repeats are augmented training exposure, not unique observations.",
            "Public validation and test backgrounds do not validate target-camera false positives.",
            "Physical chip observations remain non-authoritative Plus evidence.",
        ],
        "negative_records": negative_records,
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset": str(output.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "negative_unique_images": len(negative_images),
        "negative_training_entries": len(negative_images) * repeats,
        "split_image_counts": split_image_counts,
        "copy_modes": dict(copy_modes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    result = build_dataset(
        args.base.resolve(),
        args.negatives.resolve(),
        args.output.resolve(),
        repeats=args.repeats,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
