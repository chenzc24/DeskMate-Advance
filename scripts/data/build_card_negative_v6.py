"""Build a V5-replay training view with deterministic target-camera negatives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Iterable

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEGATIVES = ROOT / "data" / "raw" / "poker_label" / "neg_new"
DEFAULT_V5_DATASET = ROOT / "data" / "work" / "card_corner_v5" / "dataset"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "card_negative_v6" / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CARD_NEGATIVE_GROUPS = {"cards_neg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    suffix = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def augment_negative(
    image: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply bounded camera-realistic geometry and photometric degradation."""
    rng = random.Random(seed)
    height, width = image.shape[:2]
    angle = rng.uniform(-25.0, 25.0)
    scale = rng.uniform(0.92, 1.08)
    translate_x = rng.uniform(-0.04, 0.04) * width
    translate_y = rng.uniform(-0.04, 0.04) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    perspective_fraction = rng.uniform(0.0, 0.018)
    jitter_x = perspective_fraction * width
    jitter_y = perspective_fraction * height
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    destination = source + np.float32(
        [
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)]
            for _ in range(4)
        ]
    )
    perspective = cv2.getPerspectiveTransform(source, destination)
    transformed = cv2.warpPerspective(
        transformed,
        perspective,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha = rng.uniform(0.78, 1.22)
    beta = rng.uniform(-18.0, 18.0)
    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
    channel_gains = np.array(
        [rng.uniform(0.94, 1.06), rng.uniform(0.97, 1.03), rng.uniform(0.94, 1.06)],
        dtype=np.float32,
    )
    transformed = np.clip(
        transformed.astype(np.float32) * channel_gains.reshape(1, 1, 3),
        0,
        255,
    ).astype(np.uint8)

    blur = rng.choice(("none", "gaussian3", "gaussian5", "motion5"))
    if blur == "gaussian3":
        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)
    elif blur == "gaussian5":
        transformed = cv2.GaussianBlur(transformed, (5, 5), 0)
    elif blur == "motion5":
        kernel = np.zeros((5, 5), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[2, :] = 0.2
        else:
            kernel[:, 2] = 0.2
        transformed = cv2.filter2D(transformed, -1, kernel)

    noise_sigma = rng.uniform(0.0, 5.0)
    if noise_sigma:
        noise_rng = np.random.default_rng(seed ^ 0x5A17)
        noise = noise_rng.normal(0.0, noise_sigma, transformed.shape)
        transformed = np.clip(
            transformed.astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)

    jpeg_quality = rng.randint(68, 94)
    ok, encoded = cv2.imencode(
        ".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not ok:
        raise ValueError("could not apply JPEG degradation")
    transformed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return transformed, {
        "angle_degrees": round(angle, 4),
        "scale": round(scale, 6),
        "translate_fraction": [
            round(translate_x / width, 6),
            round(translate_y / height, 6),
        ],
        "perspective_fraction": round(perspective_fraction, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "channel_gains_bgr": [round(float(value), 6) for value in channel_gains],
        "blur": blur,
        "noise_sigma": round(noise_sigma, 4),
        "jpeg_quality": jpeg_quality,
        "mirrored": False,
    }


def _write_list(path: Path, rows: list[Path]) -> None:
    path.write_text(
        "".join(f"{item.resolve().as_posix()}\n" for item in rows),
        encoding="utf-8",
    )


def _make_contact_sheet(
    records: list[dict[str, object]],
    output: Path,
) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_sha256"]), []).append(record)
    panels: list[np.ndarray] = []
    for source_records in grouped.values():
        source_records.sort(key=lambda item: int(item["variant"]))
        chosen = [source_records[index] for index in (0, 3, 6, 9)]
        thumbs: list[np.ndarray] = []
        for record in chosen:
            image = _read_image(Path(str(record["output_image"])))
            thumb = cv2.resize(image, (160, 105), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb,
                f"v{int(record['variant']):02d}",
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
        panel = np.vstack((np.hstack(thumbs[:2]), np.hstack(thumbs[2:])))
        cv2.putText(
            panel,
            f"{source_records[0]['group']}/{source_records[0]['source_name']}",
            (4, panel.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    blank = np.zeros_like(panels[0])
    while len(panels) % 6:
        panels.append(blank.copy())
    rows = [
        np.hstack(panels[index : index + 6])
        for index in range(0, len(panels), 6)
    ]
    _write_image(output, np.vstack(rows), quality=92)


def build_dataset(
    negatives_root: Path,
    v5_dataset: Path,
    output_root: Path,
    variants_per_source: int,
    seed: int,
) -> dict[str, object]:
    if variants_per_source < 1:
        raise ValueError("variants_per_source must be positive")
    v5_yaml_path = v5_dataset / "dataset.yaml"
    v5_manifest_path = v5_dataset / "manifest.json"
    v5_train_path = v5_dataset / "train.txt"
    v5_val_path = v5_dataset / "val.txt"
    for required in (v5_yaml_path, v5_manifest_path, v5_train_path, v5_val_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    all_images = sorted(
        path
        for path in negatives_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    source_paths = [
        path for path in all_images if path.parent.name in CARD_NEGATIVE_GROUPS
    ]
    if not source_paths:
        raise ValueError(f"no negative images found under {negatives_root}")
    source_hashes = [sha256_file(path) for path in source_paths]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("exact duplicate negative sources detected")

    staging = output_root.with_name(
        f"{output_root.name}.partial.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_root}")
    staging.mkdir(parents=True)
    negative_images = staging / "negatives" / "images" / "train"
    negative_labels = staging / "negatives" / "labels" / "train"
    negative_images.mkdir(parents=True)
    negative_labels.mkdir(parents=True)
    records: list[dict[str, object]] = []

    try:
        for source_index, source_path in enumerate(source_paths):
            source_image = _read_image(source_path)
            source_sha = source_hashes[source_index]
            group = source_path.parent.name
            for variant in range(variants_per_source):
                stem = f"{group}_{source_index:03d}_{source_sha[:10]}_v{variant:02d}"
                if variant == 0:
                    output_image = negative_images / f"{stem}{source_path.suffix.lower()}"
                    try:
                        os.link(source_path, output_image)
                    except OSError:
                        shutil.copy2(source_path, output_image)
                    augmentation: dict[str, object] = {
                        "identity": True,
                        "mirrored": False,
                    }
                else:
                    output_image = negative_images / f"{stem}.jpg"
                    item_seed = seed + source_index * 1009 + variant * 9176
                    image, augmentation = augment_negative(source_image, item_seed)
                    _write_image(output_image, image, quality=92)
                    augmentation["identity"] = False
                output_label = negative_labels / f"{stem}.txt"
                output_label.write_bytes(b"")
                records.append(
                    {
                        "group": group,
                        "source_name": source_path.name,
                        "source_path": str(source_path.resolve()),
                        "source_sha256": source_sha,
                        "variant": variant,
                        "output_image": str(output_image.resolve()),
                        "output_image_sha256": sha256_file(output_image),
                        "output_label": str(output_label.resolve()),
                        "output_label_sha256": sha256_file(output_label),
                        "augmentation": augmentation,
                    }
                )

        v5_train = [
            Path(line.strip())
            for line in v5_train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        v5_val = [
            Path(line.strip())
            for line in v5_val_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(v5_train) != 2550 or len(v5_val) != 450:
            raise ValueError("unexpected V5 train/validation counts")
        for image_path in (*v5_train, *v5_val):
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        negative_outputs = [Path(str(record["output_image"])) for record in records]
        train_rows = [*v5_train, *negative_outputs]
        train_list = staging / "train.txt"
        val_list = staging / "val.txt"
        _write_list(train_list, train_rows)
        _write_list(val_list, v5_val)

        v5_yaml = yaml.safe_load(v5_yaml_path.read_text(encoding="utf-8"))
        dataset_yaml = staging / "dataset.yaml"
        dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(staging.resolve()).replace("\\", "/"),
                    "train": str(train_list.resolve()).replace("\\", "/"),
                    "val": str(val_list.resolve()).replace("\\", "/"),
                    "nc": int(v5_yaml["nc"]),
                    "names": v5_yaml["names"],
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        review_path = staging / "review" / "negative_augmentation_overview.jpg"
        review_path.parent.mkdir(parents=True)
        _make_contact_sheet(records, review_path)
        counts_by_group = {
            group: sum(1 for record in records if record["group"] == group)
            for group in sorted({str(record["group"]) for record in records})
        }
        manifest = {
            "schema_version": "poker_dealer.card_negative_training_view.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "development",
            "seed": seed,
            "negative_contract": (
                "face-down backs, hands, table/chip context and explicitly "
                "unsupported novelty/deck designs have zero detector labels"
            ),
            "v5": {
                "dataset": str(v5_dataset.resolve()),
                "manifest_sha256": sha256_file(v5_manifest_path),
                "train_list_sha256": sha256_file(v5_train_path),
                "validation_list_sha256": sha256_file(v5_val_path),
                "positive_train_images": len(v5_train),
                "positive_validation_images": len(v5_val),
            },
            "negative": {
                "root": str(negatives_root.resolve()),
                "accepted_groups": sorted(CARD_NEGATIVE_GROUPS),
                "excluded_images": [
                    {
                        "group": path.parent.name,
                        "path": str(path.resolve()),
                        "reason": (
                            "not a card-detector negative; chip_neg contains "
                            "face-up playing cards"
                        ),
                    }
                    for path in all_images
                    if path not in source_paths
                ],
                "source_images": len(source_paths),
                "variants_per_source_including_original": variants_per_source,
                "generated_images": len(records),
                "counts_by_group": counts_by_group,
                "all_labels_empty": True,
                "mirroring": False,
                "records": records,
            },
            "combined": {
                "train_images": len(train_rows),
                "validation_images": len(v5_val),
                "negative_train_fraction": len(records) / len(train_rows),
            },
            "train_list_sha256": sha256_file(train_list),
            "validation_list_sha256": sha256_file(val_list),
            "review_contact_sheet": str(review_path.resolve()),
            "review_contact_sheet_sha256": sha256_file(review_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
        # Absolute staging paths change after the atomic rename; rewrite outputs.
        manifest_path = output_root / "manifest.json"
        text = manifest_path.read_text(encoding="utf-8").replace(
            str(staging.resolve()), str(output_root.resolve())
        )
        manifest_path.write_text(text, encoding="utf-8")
        yaml_path = output_root / "dataset.yaml"
        text = yaml_path.read_text(encoding="utf-8").replace(
            str(staging.resolve()).replace("\\", "/"),
            str(output_root.resolve()).replace("\\", "/"),
        )
        yaml_path.write_text(text, encoding="utf-8")
        train_path = output_root / "train.txt"
        text = train_path.read_text(encoding="utf-8").replace(
            str(staging.resolve()).replace("\\", "/"),
            str(output_root.resolve()).replace("\\", "/"),
        )
        train_path.write_text(text, encoding="utf-8")
        final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_manifest["train_list_sha256"] = sha256_file(train_path)
        final_manifest["validation_list_sha256"] = sha256_file(output_root / "val.txt")
        final_manifest["review_contact_sheet_sha256"] = sha256_file(
            output_root / "review" / "negative_augmentation_overview.jpg"
        )
        manifest_path.write_text(
            json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return final_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--v5-dataset", type=Path, default=DEFAULT_V5_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(
        args.negatives.resolve(),
        args.v5_dataset.resolve(),
        args.output.resolve(),
        args.variants_per_source,
        args.seed,
    )
    print(json.dumps(manifest["combined"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
