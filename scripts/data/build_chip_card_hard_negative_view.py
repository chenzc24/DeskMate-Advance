"""Build a group-safe chip dataset view with new poker-card hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "C:/Users/ASUS/xwechat_files/wxid_08cxqt3rjj2822_8f6a/"
    "msg/file/2026-07/neg_new/neg_new"
)
DEFAULT_RAW = ROOT / "data/raw/chips/2026-07-25-card-hard-negative-source"
DEFAULT_WORK = ROOT / "data/work/chips/2026-07-25-card-hard-negative-v2"
DEFAULT_BASE = ROOT / "data/work/chips/2026-07-24-chip-v2-optimization/dataset"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAIN_GROUP = "chip_neg"
HOLDOUT_GROUP = "cards_neg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def _snapshot_source(source_root: Path, raw_root: Path) -> list[dict[str, object]]:
    sources = image_paths(source_root)
    if not sources:
        raise ValueError(f"no source images found: {source_root}")
    records: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(source_root)
        if relative.parent.name not in {TRAIN_GROUP, HOLDOUT_GROUP}:
            raise ValueError(f"unexpected source group: {relative}")
        destination = raw_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise ValueError(f"immutable raw snapshot differs: {destination}")
        else:
            shutil.copy2(source, destination)
        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable source: {destination}")
        height, width = image.shape[:2]
        records.append(
            {
                "relative_path": relative.as_posix(),
                "capture_group": relative.parent.name,
                "split": (
                    "train_hard_negative"
                    if relative.parent.name == TRAIN_GROUP
                    else "heldout_negative"
                ),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "width": width,
                "height": height,
                "difference_hash": f"{difference_hash(image):016x}",
            }
        )
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("exact duplicate source images are not allowed")
    return records


def _audit_cross_split_near_duplicates(
    records: list[dict[str, object]],
    raw_root: Path,
    maximum_hamming: int = 3,
    maximum_mean_absolute_difference: float = 4.0,
) -> list[dict[str, object]]:
    collisions: list[dict[str, object]] = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first["split"] == second["split"]:
                continue
            distance = hamming(
                int(str(first["difference_hash"]), 16),
                int(str(second["difference_hash"]), 16),
            )
            if distance > maximum_hamming:
                continue
            first_image = cv2.imread(
                str(raw_root / str(first["relative_path"])),
                cv2.IMREAD_COLOR,
            )
            second_image = cv2.imread(
                str(raw_root / str(second["relative_path"])),
                cv2.IMREAD_COLOR,
            )
            if first_image is None or second_image is None:
                raise ValueError("failed to decode snapshot during leakage audit")
            first_thumbnail = cv2.resize(
                first_image,
                (160, 120),
                interpolation=cv2.INTER_AREA,
            )
            second_thumbnail = cv2.resize(
                second_image,
                (160, 120),
                interpolation=cv2.INTER_AREA,
            )
            mean_absolute_difference = float(
                np.mean(cv2.absdiff(first_thumbnail, second_thumbnail))
            )
            if mean_absolute_difference <= maximum_mean_absolute_difference:
                collisions.append(
                    {
                        "first": first["relative_path"],
                        "second": second["relative_path"],
                        "hamming": distance,
                        "mean_absolute_difference": round(
                            mean_absolute_difference,
                            5,
                        ),
                    }
                )
    return collisions


def _augment_negative(
    image: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    angle = float(rng.uniform(-24.0, 24.0))
    scale = float(rng.uniform(0.88, 1.06))
    translate_x = float(rng.uniform(-0.055, 0.055) * width)
    translate_y = float(rng.uniform(-0.055, 0.055) * height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += (translate_x, translate_y)
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    gamma = float(rng.uniform(0.72, 1.32))
    lookup = np.asarray(
        [round(255.0 * ((value / 255.0) ** gamma)) for value in range(256)],
        dtype=np.uint8,
    )
    transformed = cv2.LUT(transformed, lookup)
    effect = str(rng.choice(["none", "blur", "noise", "shadow"]))
    if effect == "blur":
        transformed = cv2.GaussianBlur(
            transformed,
            (0, 0),
            sigmaX=float(rng.uniform(0.5, 1.6)),
        )
    elif effect == "noise":
        noise = rng.normal(0.0, rng.uniform(2.0, 7.0), transformed.shape)
        transformed = np.clip(
            transformed.astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
    elif effect == "shadow":
        overlay = np.zeros((height, width), dtype=np.float32)
        center = (
            int(rng.uniform(0.15, 0.85) * width),
            int(rng.uniform(0.15, 0.85) * height),
        )
        axes = (
            int(rng.uniform(0.18, 0.42) * width),
            int(rng.uniform(0.12, 0.30) * height),
        )
        cv2.ellipse(
            overlay,
            center,
            axes,
            float(rng.uniform(0, 180)),
            0,
            360,
            1.0,
            cv2.FILLED,
        )
        overlay = cv2.GaussianBlur(
            overlay,
            (0, 0),
            sigmaX=max(width, height) * 0.05,
        )
        strength = float(rng.uniform(0.12, 0.28))
        transformed = np.clip(
            transformed.astype(np.float32)
            * (1.0 - overlay[:, :, None] * strength),
            0,
            255,
        ).astype(np.uint8)
    return transformed, {
        "seed": seed,
        "angle_degrees": round(angle, 4),
        "scale": round(scale, 5),
        "translate_fraction": [
            round(translate_x / width, 5),
            round(translate_y / height, 5),
        ],
        "gamma": round(gamma, 5),
        "effect": effect,
    }


def _yaml_path(path: Path) -> str:
    return json.dumps(path.resolve().as_posix())


def build_view(
    *,
    source_root: Path,
    raw_root: Path,
    work_root: Path,
    base_root: Path,
    variants_per_train_source: int,
) -> dict[str, object]:
    if variants_per_train_source < 1:
        raise ValueError("variants_per_train_source must include at least the original")
    required_base = [
        base_root / split / kind
        for split in ("train", "valid", "test")
        for kind in ("images", "labels")
    ] + [base_root / "dataset_manifest.json", base_root / "target_validation.yaml"]
    missing = [str(path) for path in required_base if not path.exists()]
    if missing:
        raise ValueError(f"base chip-v2 dataset is incomplete: {missing}")

    records = _snapshot_source(source_root, raw_root)
    cross_split_near_duplicates = _audit_cross_split_near_duplicates(
        records,
        raw_root,
    )
    if cross_split_near_duplicates:
        raise ValueError(
            "near-duplicate leakage between negative sequences: "
            f"{cross_split_near_duplicates}"
        )

    dataset_root = work_root / "dataset"
    if dataset_root.exists():
        resolved = dataset_root.resolve()
        if resolved.parent != work_root.resolve():
            raise ValueError(f"refusing unexpected dataset cleanup: {resolved}")
        shutil.rmtree(resolved)
    train_images = dataset_root / "new_train" / "images"
    train_labels = dataset_root / "new_train" / "labels"
    holdout_images = dataset_root / "negative_holdout" / "images"
    holdout_labels = dataset_root / "negative_holdout" / "labels"

    derived_records: list[dict[str, object]] = []
    for record in records:
        source = raw_root / str(record["relative_path"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        assert image is not None
        is_train = record["split"] == "train_hard_negative"
        variants = variants_per_train_source if is_train else 1
        for variant_index in range(variants):
            output = image
            augmentation: dict[str, object] | None = None
            if variant_index:
                seed = int(
                    hashlib.sha256(
                        f"{record['sha256']}:{variant_index}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                output, augmentation = _augment_negative(image, seed=seed)
            stem = (
                f"card_negative_{record['capture_group']}_"
                f"{Path(str(record['relative_path'])).stem}_"
                f"{str(record['sha256'])[:10]}_a{variant_index:02d}"
            )
            image_root = train_images if is_train else holdout_images
            label_root = train_labels if is_train else holdout_labels
            output_image = image_root / f"{stem}.jpg"
            output_label = label_root / f"{stem}.txt"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(output_image),
                output,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(f"failed to write: {output_image}")
            output_label.write_text("", encoding="utf-8")
            derived_records.append(
                {
                    "source_relative_path": record["relative_path"],
                    "source_sha256": record["sha256"],
                    "capture_group": record["capture_group"],
                    "split": record["split"],
                    "output_image": output_image.relative_to(dataset_root).as_posix(),
                    "output_image_sha256": sha256(output_image),
                    "output_label": output_label.relative_to(dataset_root).as_posix(),
                    "label_bytes": output_label.stat().st_size,
                    "augmentation": augmentation,
                }
            )

    data_yaml = (
        f"path: {_yaml_path(dataset_root)}\n"
        "train:\n"
        f"  - {_yaml_path(base_root / 'train/images')}\n"
        f"  - {_yaml_path(train_images)}\n"
        f"val: {_yaml_path(base_root / 'valid/images')}\n"
        f"test: {_yaml_path(base_root / 'test/images')}\n"
        "names:\n"
        "  0: poker_chip\n"
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    data_yaml_path = dataset_root / "data.yaml"
    data_yaml_path.write_text(data_yaml, encoding="utf-8")
    holdout_list = dataset_root / "negative_holdout.txt"
    holdout_paths = [
        str((dataset_root / str(record["output_image"])).resolve())
        for record in derived_records
        if record["split"] == "heldout_negative"
    ]
    holdout_list.write_text("\n".join(holdout_paths) + "\n", encoding="utf-8")

    source_manifest = {
        "schema_version": "1.0",
        "dataset_id": "chip-card-hard-negative-source-20260725",
        "external_source": str(source_root.resolve()),
        "raw_snapshot": str(raw_root.resolve()),
        "split_policy": (
            "complete chip_neg face-up sequence train-only; complete cards_neg "
            "card-back sequence held out; adjacent frames and derivatives never cross"
        ),
        "exact_unique": len({record["sha256"] for record in records}),
        "cross_split_near_duplicate_max_hamming": 3,
        "cross_split_near_duplicates": cross_split_near_duplicates,
        "records": records,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = work_root / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    base_counts = {
        split: len(image_paths(base_root / split / "images"))
        for split in ("train", "valid", "test")
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "poker-chip-localization-card-hard-negative-20260725-v2",
        "class_map": {"0": "poker_chip"},
        "base_dataset": str(base_root.resolve()),
        "base_manifest_sha256": sha256(base_root / "dataset_manifest.json"),
        "source_manifest_sha256": sha256(source_manifest_path),
        "data_yaml_sha256": sha256(data_yaml_path),
        "split_policy": source_manifest["split_policy"],
        "previous_dataset_replay": {
            "train_images": base_counts["train"],
            "valid_images": base_counts["valid"],
            "test_images": base_counts["test"],
            "target_validation_yaml": str(
                (base_root / "target_validation.yaml").resolve()
            ),
        },
        "new_negative_sources": {
            "train": sum(record["split"] == "train_hard_negative" for record in records),
            "heldout": sum(record["split"] == "heldout_negative" for record in records),
        },
        "variants_per_train_source_including_original": variants_per_train_source,
        "new_negative_images": {
            "train": sum(
                record["split"] == "train_hard_negative"
                for record in derived_records
            ),
            "heldout": sum(
                record["split"] == "heldout_negative"
                for record in derived_records
            ),
        },
        "negative_holdout_list_sha256": sha256(holdout_list),
        "all_new_labels_empty": all(
            int(record["label_bytes"]) == 0 for record in derived_records
        ),
        "derived_records": derived_records,
    }
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "data_yaml": str(data_yaml_path.resolve()),
        "data_yaml_sha256": sha256(data_yaml_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--variants-per-train-source",
        type=int,
        default=6,
        help="including the unchanged original",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_view(
        source_root=args.source.resolve(),
        raw_root=args.raw_root.resolve(),
        work_root=args.work_root.resolve(),
        base_root=args.base_root.resolve(),
        variants_per_train_source=args.variants_per_train_source,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
