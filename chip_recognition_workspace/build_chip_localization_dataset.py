"""Build a single-class poker-chip localization dataset from reviewed sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotation(row: str, *, segment_only: bool) -> tuple[float, float, float, float] | None:
    fields = row.split()
    if len(fields) < 5:
        raise ValueError(f"invalid YOLO row: {row}")
    coordinates = [float(value) for value in fields[1:]]
    if len(coordinates) == 4:
        if segment_only:
            return None
        x_center, y_center, width, height = coordinates
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
    else:
        raise ValueError(f"unsupported YOLO row: {row}")
    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(f"out-of-range YOLO row: {row}")
    return x_center, y_center, width, height


def roboflow_group(stem: str, *, collapse_burst: bool) -> str:
    group = stem.split(".rf.", 1)[0]
    if collapse_burst and "_BURST" in group:
        group = group.split("_BURST", 1)[0]
    return group


def chip_det_session_group(stem: str) -> str:
    """Keep adjacent timestamped chip_det frames in one split."""
    group = stem.split(".rf.", 1)[0]
    timestamp = re.search(r"(20\d{6})[_-]?(\d{6})", group)
    if timestamp:
        return timestamp.group(1)
    return group


def grouped_splits(groups: list[str], *, seed: int) -> dict[str, str]:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    valid_count = max(1, round(len(unique) * 0.15))
    test_count = max(1, round(len(unique) * 0.15))
    train_count = len(unique) - valid_count - test_count
    if train_count < 1:
        raise ValueError("not enough source groups for train/valid/test")
    mapping = {group: "train" for group in unique[:train_count]}
    mapping.update(
        {group: "valid" for group in unique[train_count : train_count + valid_count]}
    )
    mapping.update({group: "test" for group in unique[train_count + valid_count :]})
    return mapping


def discover_roboflow(
    root: Path,
    *,
    dataset_id: str,
    segment_only: bool,
    collapse_burst: bool,
    collapse_date_session: bool = False,
) -> list[dict]:
    records = []
    for upstream_split in SPLITS:
        image_dir = root / upstream_split / "images"
        label_dir = root / upstream_split / "labels"
        for image_path in sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = []
            for row in label_path.read_text(encoding="utf-8").splitlines():
                box = parse_annotation(row, segment_only=segment_only)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                continue
            source_group = (
                chip_det_session_group(image_path.stem)
                if collapse_date_session
                else roboflow_group(image_path.stem, collapse_burst=collapse_burst)
            )
            records.append(
                {
                    "dataset_id": dataset_id,
                    "source_image": image_path,
                    "source_label": label_path,
                    "source_group": f"{dataset_id}:{source_group}",
                    "upstream_split": upstream_split,
                    "boxes": boxes,
                }
            )
    return records


def discover_local(root: Path) -> list[dict]:
    records = []
    for image_path in sorted(
        path for path in (root / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        label_path = root / "labels" / f"{image_path.stem}.txt"
        boxes = [
            box
            for row in label_path.read_text(encoding="utf-8").splitlines()
            if (box := parse_annotation(row, segment_only=False)) is not None
        ]
        if not boxes:
            continue
        record_id = image_path.stem.split("_", 1)[0]
        records.append(
            {
                "dataset_id": "target-camera-single-chip-v1",
                "source_image": image_path,
                "source_label": label_path,
                "source_group": f"target-camera:{record_id}",
                "upstream_split": "fit_only",
                "split": "train",
                "boxes": boxes,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip-counting", type=Path, required=True)
    parser.add_argument("--chips-demo", type=Path, required=True)
    parser.add_argument("--chip-det", type=Path, required=True)
    parser.add_argument("--local-augmented", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {args.output}")

    chip_counting = discover_roboflow(
        args.chip_counting,
        dataset_id="chip-counting-rovys-v2",
        segment_only=True,
        collapse_burst=False,
    )
    chips_demo = discover_roboflow(
        args.chips_demo,
        dataset_id="chips-demo-v1",
        segment_only=False,
        collapse_burst=True,
    )
    chip_det = discover_roboflow(
        args.chip_det,
        dataset_id="chip-det-v16",
        segment_only=False,
        collapse_burst=False,
        collapse_date_session=True,
    )
    local = discover_local(args.local_augmented)

    for dataset_offset, records in enumerate((chip_counting, chips_demo, chip_det)):
        split_map = grouped_splits(
            [record["source_group"] for record in records],
            seed=args.seed + dataset_offset,
        )
        for record in records:
            record["split"] = split_map[record["source_group"]]

    all_records = chip_counting + chips_demo + chip_det + local
    args.output.mkdir(parents=True)
    for split in SPLITS:
        (args.output / split / "images").mkdir(parents=True)
        (args.output / split / "labels").mkdir(parents=True)

    manifest_records = []
    split_images: Counter[str] = Counter()
    split_instances: Counter[str] = Counter()
    for index, record in enumerate(all_records, start=1):
        split = record["split"]
        source_image: Path = record["source_image"]
        source_label: Path = record["source_label"]
        output_stem = f"{record['dataset_id']}_{index:05d}_{source_image.stem}"
        output_image = args.output / split / "images" / f"{output_stem}{source_image.suffix.lower()}"
        output_label = args.output / split / "labels" / f"{output_stem}.txt"
        shutil.copy2(source_image, output_image)
        output_label.write_text(
            "".join(
                f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n"
                for x, y, width, height in record["boxes"]
            ),
            encoding="utf-8",
        )
        split_images[split] += 1
        split_instances[split] += len(record["boxes"])
        manifest_records.append(
            {
                "dataset_id": record["dataset_id"],
                "split": split,
                "source_group": record["source_group"],
                "upstream_split": record["upstream_split"],
                "source_image": str(source_image.resolve()),
                "source_image_sha256": sha256(source_image),
                "source_label": str(source_label.resolve()),
                "source_label_sha256": sha256(source_label),
                "output_image": str(output_image.resolve()),
                "output_image_sha256": sha256(output_image),
                "output_label": str(output_label.resolve()),
                "output_label_sha256": sha256(output_label),
                "instances": len(record["boxes"]),
            }
        )

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output.resolve()}",
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

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "poker-chip-localization-public-target-camera-v1",
        "status": "development_no_target_camera_holdout_no_negative_samples",
        "seed": args.seed,
        "class_map": {"0": "poker_chip"},
        "split_policy": (
            "public data grouped by Roboflow source stem, Chips_Demo burst timestamp, "
            "and chip_det capture date; "
            "all target-camera single-session views restricted to train"
        ),
        "source_audit_sha256": sha256(args.audit_report),
        "local_manifest_sha256": sha256(args.local_manifest),
        "upstream_sources": [
            {
                "root": item["root"],
                "archive": item["archive"],
                "archive_sha256": item["archive_sha256"],
                "source_snapshot_sha256": item["source_snapshot_sha256"],
                "license": item["license"],
                "source_url": item["source_url"],
            }
            for item in audit
        ],
        "split_image_counts": dict(split_images),
        "split_instance_counts": dict(split_instances),
        "limitations": [
            "Public chip designs, camera geometry, and backgrounds differ from the target robot camera.",
            "No target-camera capture session is held out for validation or test.",
            "No pure negative images are included, so false-positive rejection is not validated.",
            "Only visible chips can be localized; fully occluded stack members cannot be counted.",
            "Outputs are perception evidence and are not authoritative ledger values.",
        ],
        "records": manifest_records,
    }
    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(args.output.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "split_image_counts": dict(split_images),
                "split_instance_counts": dict(split_instances),
                "source_group_counts": {
                    split: len(
                        {
                            record["source_group"]
                            for record in manifest_records
                            if record["split"] == split
                        }
                    )
                    for split in SPLITS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
