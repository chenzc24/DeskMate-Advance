"""Audit downloaded Roboflow YOLO datasets without modifying source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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


def source_group(stem: str) -> str:
    return stem.split(".rf.", 1)[0]


def read_names(config: dict) -> list[str]:
    names = config["names"]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda item: int(item))]
    return [str(item) for item in names]


def audit_dataset(root: Path, archive: Path) -> tuple[dict, list[dict]]:
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = read_names(config)
    source_files = []
    snapshot_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        file_sha256 = sha256(path)
        snapshot_digest.update(relative_path.encode("utf-8"))
        snapshot_digest.update(b"\0")
        snapshot_digest.update(file_sha256.encode("ascii"))
        snapshot_digest.update(b"\n")
        source_files.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": file_sha256,
            }
        )
    report: dict[str, object] = {
        "root": str(root.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "license": config.get("roboflow", {}).get("license"),
        "source_url": config.get("roboflow", {}).get("url"),
        "source_snapshot_sha256": snapshot_digest.hexdigest(),
        "source_files": source_files,
        "dataset_yaml_sha256": sha256(root / "data.yaml"),
        "classes": {str(index): name for index, name in enumerate(names)},
        "splits": {},
        "issues": [],
    }
    records: list[dict] = []
    all_hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    all_group_splits: defaultdict[str, set[str]] = defaultdict(set)
    total_instances: Counter[str] = Counter()

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        class_counts: Counter[str] = Counter()
        annotation_kinds: Counter[str] = Counter()
        invalid_rows: list[dict] = []
        empty_labels = 0

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                report["issues"].append({"type": "unreadable_image", "path": str(image_path)})
                continue
            height, width = image.shape[:2]
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes: list[dict] = []
            rows = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            if not rows:
                empty_labels += 1
            for line_number, row in enumerate(rows, start=1):
                fields = row.split()
                try:
                    class_id = int(fields[0])
                    coordinates = list(map(float, fields[1:]))
                    if len(coordinates) == 4:
                        x_center, y_center, box_width, box_height = coordinates
                        annotation_kind = "box"
                        valid = (
                            0 <= class_id < len(names)
                            and 0.0 <= x_center <= 1.0
                            and 0.0 <= y_center <= 1.0
                            and 0.0 < box_width <= 1.0
                            and 0.0 < box_height <= 1.0
                            and x_center - box_width / 2 >= -1e-5
                            and y_center - box_height / 2 >= -1e-5
                            and x_center + box_width / 2 <= 1.0 + 1e-5
                            and y_center + box_height / 2 <= 1.0 + 1e-5
                        )
                    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                        xs = coordinates[0::2]
                        ys = coordinates[1::2]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        box_width = x2 - x1
                        box_height = y2 - y1
                        annotation_kind = "segment"
                        valid = (
                            0 <= class_id < len(names)
                            and all(0.0 <= value <= 1.0 for value in coordinates)
                            and box_width > 0.0
                            and box_height > 0.0
                        )
                    else:
                        valid = False
                except (ValueError, IndexError):
                    valid = False
                if not valid:
                    invalid_rows.append(
                        {"path": str(label_path), "line": line_number, "content": row}
                    )
                    continue
                class_name = names[class_id]
                annotation_kinds[annotation_kind] += 1
                class_counts[class_name] += 1
                total_instances[class_name] += 1
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "xywhn": [x_center, y_center, box_width, box_height],
                    }
                )

            image_hash = sha256(image_path)
            group = source_group(image_path.stem)
            all_hash_splits[image_hash].add(split)
            all_group_splits[group].add(split)
            records.append(
                {
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": image_hash,
                    "source_group": group,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "source_groups": len({source_group(path.stem) for path in images}),
            "empty_labels": empty_labels,
            "class_instances": dict(sorted(class_counts.items())),
            "annotation_kinds": dict(sorted(annotation_kinds.items())),
            "images_without_labels": sorted(image_stems - label_stems),
            "labels_without_images": sorted(label_stems - image_stems),
            "invalid_label_rows": invalid_rows,
        }

    report["total_images"] = len(records)
    report["total_source_groups"] = len(all_group_splits)
    report["total_instances"] = sum(total_instances.values())
    report["class_instances"] = dict(sorted(total_instances.items()))
    report["exact_image_hash_cross_split_leaks"] = {
        digest: sorted(splits)
        for digest, splits in all_hash_splits.items()
        if len(splits) > 1
    }
    report["source_group_cross_split_leaks"] = {
        group: sorted(splits)
        for group, splits in all_group_splits.items()
        if len(splits) > 1
    }
    return report, records


def draw_contact_sheet(records: list[dict], output: Path, seed: int, limit: int = 24) -> None:
    randomizer = random.Random(seed)
    chosen = randomizer.sample(records, min(limit, len(records)))
    tiles = []
    palette = [(80, 220, 80), (255, 150, 60), (80, 80, 255), (255, 80, 200)]
    for record in chosen:
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        for box in record["boxes"]:
            x_center, y_center, box_width, box_height = box["xywhn"]
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)
            color = palette[box["class_id"] % len(palette)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                box["class_name"],
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        caption = f"{record['split']} | {record['source_group'][:35]}"
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        scale = min(320 / width, 220 / height)
        resized = cv2.resize(image, (int(width * scale), int(height * scale)))
        canvas = 255 * __import__("numpy").ones((240, 340, 3), dtype="uint8")
        y_offset = (240 - resized.shape[0]) // 2
        x_offset = (340 - resized.shape[1]) // 2
        canvas[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
        tiles.append(canvas)

    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start : start + 4]
        while len(row) < 4:
            row.append(255 * __import__("numpy").ones((240, 340, 3), dtype="uint8"))
        rows.append(cv2.hconcat(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.vconcat(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--archive", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if len(args.dataset) != len(args.archive):
        parser.error("provide exactly one --archive for every --dataset")

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (root, archive) in enumerate(zip(args.dataset, args.archive)):
        report, records = audit_dataset(root, archive)
        reports.append(report)
        draw_contact_sheet(
            records,
            args.output / f"{root.name}_contact_sheet.jpg",
            seed=args.seed + index,
        )
    destination = args.output / "external_dataset_audit.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "root": report["root"],
                    "archive_sha256": report["archive_sha256"],
                    "source_snapshot_sha256": report["source_snapshot_sha256"],
                    "total_images": report["total_images"],
                    "total_source_groups": report["total_source_groups"],
                    "total_instances": report["total_instances"],
                    "class_instances": report["class_instances"],
                    "split_counts": {
                        split: {
                            "images": details["images"],
                            "source_groups": details["source_groups"],
                            "annotation_kinds": details["annotation_kinds"],
                            "invalid_rows": len(details["invalid_label_rows"]),
                        }
                        for split, details in report["splits"].items()
                    },
                    "source_group_cross_split_leaks": len(
                        report["source_group_cross_split_leaks"]
                    ),
                }
                for report in reports
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
