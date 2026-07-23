"""Convert the external Pascal VOC cards into a pinned YOLO training view."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL = (
    ROOT
    / "data"
    / "raw"
    / "poker_label"
    / "external"
    / "kaggle_andy8744_playing_cards_object_detection"
)
DEFAULT_TARGET = ROOT / "data" / "work" / "poker_new_augmented_v1"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "card_finetune_v1"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "lgd-cards-gen3"
    / "model.classes.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_card_code(value: str) -> str:
    value = value.strip()
    if len(value) < 2:
        raise ValueError(f"invalid card code: {value!r}")
    rank = value[:-1].upper()
    suit = value[-1].upper()
    if rank not in {"A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"}:
        raise ValueError(f"invalid card rank: {value!r}")
    if suit not in {"C", "D", "H", "S"}:
        raise ValueError(f"invalid card suit: {value!r}")
    return f"{rank}{suit}"


def convert_annotation(
    xml_path: Path, class_to_id: dict[str, int]
) -> tuple[str, list[str], Counter[int]]:
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or "").strip()
    width = int(root.findtext("size/width") or 0)
    height = int(root.findtext("size/height") or 0)
    if not filename or width <= 0 or height <= 0:
        raise ValueError(f"invalid filename or size: {xml_path}")
    rows: list[str] = []
    counts: Counter[int] = Counter()
    for obj in root.findall("object"):
        code = normalize_card_code(obj.findtext("name") or "")
        if code not in class_to_id:
            raise ValueError(f"unknown card class {code}: {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing bndbox: {xml_path}")
        xmin = float(box.findtext("xmin") or math.nan)
        ymin = float(box.findtext("ymin") or math.nan)
        xmax = float(box.findtext("xmax") or math.nan)
        ymax = float(box.findtext("ymax") or math.nan)
        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
            raise ValueError(f"non-finite bndbox: {xml_path}")
        if not (0.0 <= xmin < xmax <= width and 0.0 <= ymin < ymax <= height):
            raise ValueError(f"out-of-frame bndbox: {xml_path}")
        class_id = class_to_id[code]
        x_center = ((xmin + xmax) * 0.5) / width
        y_center = ((ymin + ymax) * 0.5) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        rows.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )
        counts[class_id] += 1
    if not rows:
        raise ValueError(f"annotation has no objects: {xml_path}")
    return filename, rows, counts


def hardlink_same_volume(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_ino == destination_stat.st_ino
            and source_stat.st_dev == destination_stat.st_dev
        ):
            return
        raise FileExistsError(f"destination exists but is not the expected hardlink: {destination}")
    os.link(source, destination)


def write_dataset_yaml(
    path: Path,
    dataset_root: Path,
    train_path: str,
    val_path: str,
    class_names: Sequence[str],
) -> None:
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: {train_path}",
        f"val: {val_path}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_view(
    external_root: Path,
    target_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    smoke_images: int,
) -> dict[str, object]:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    source_images = external_root / "extracted" / "cards" / "scenes" / "generated"
    source_xml = external_root / "extracted" / "cards" / "scenes" / "xml"
    source_manifest = external_root / "snapshot_manifest.json"
    target_manifest = target_root / "manifest.json"
    for required in (source_images, source_xml, source_manifest, target_manifest):
        if not required.exists():
            raise FileNotFoundError(required)

    external_images = output_root / "external" / "images" / "train"
    external_labels = output_root / "external" / "labels" / "train"
    records: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    xml_paths = sorted(source_xml.glob("*.xml"), key=lambda path: path.name)
    for index, xml_path in enumerate(xml_paths):
        filename, rows, counts = convert_annotation(xml_path, class_to_id)
        source_image = source_images / filename
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        output_image = external_images / filename
        output_label = external_labels / f"{Path(filename).stem}.txt"
        hardlink_same_volume(source_image, output_image)
        label_text = "\n".join(rows) + "\n"
        if output_label.exists():
            if output_label.read_text(encoding="utf-8") != label_text:
                raise ValueError(f"existing converted label differs: {output_label}")
        else:
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(label_text, encoding="utf-8")
        records.append(
            {
                "filename": filename,
                "class_ids": sorted(counts),
                "objects": sum(counts.values()),
            }
        )
        class_counts.update(counts)
        if (index + 1) % 2000 == 0:
            print(f"converted {index + 1}/{len(xml_paths)} external annotations", flush=True)

    missing_classes = sorted(set(range(len(class_names))) - set(class_counts))
    if missing_classes:
        raise ValueError(f"external view is missing classes: {missing_classes}")

    smoke_selected: list[dict[str, object]] = []
    smoke_names: set[str] = set()
    smoke_class_counts: Counter[int] = Counter()
    for record in records:
        class_ids = [int(value) for value in record["class_ids"]]
        if any(smoke_class_counts[class_id] < 3 for class_id in class_ids):
            smoke_selected.append(record)
            smoke_names.add(str(record["filename"]))
            smoke_class_counts.update(class_ids)
    for record in records:
        if len(smoke_selected) >= smoke_images:
            break
        filename = str(record["filename"])
        if filename not in smoke_names:
            smoke_selected.append(record)
            smoke_names.add(filename)
            smoke_class_counts.update(int(value) for value in record["class_ids"])
    if len(smoke_selected) < smoke_images:
        raise ValueError("not enough images for the requested smoke view")
    smoke_selected = smoke_selected[:smoke_images]
    if set(smoke_class_counts) != set(range(len(class_names))):
        raise ValueError("smoke view does not cover all classes")

    monitor_selected: list[dict[str, object]] = []
    monitor_names: set[str] = set()
    monitor_class_counts: Counter[int] = Counter()
    for record in smoke_selected:
        class_ids = [int(value) for value in record["class_ids"]]
        if any(monitor_class_counts[class_id] < 2 for class_id in class_ids):
            monitor_selected.append(record)
            monitor_names.add(str(record["filename"]))
            monitor_class_counts.update(class_ids)
    for record in smoke_selected:
        if len(monitor_selected) >= 104:
            break
        filename = str(record["filename"])
        if filename not in monitor_names:
            monitor_selected.append(record)
            monitor_names.add(filename)
            monitor_class_counts.update(int(value) for value in record["class_ids"])
    monitor_selected = monitor_selected[:104]
    if set(monitor_class_counts) != set(range(len(class_names))):
        raise ValueError("monitor view does not cover all classes")

    smoke_train = [
        record
        for record in smoke_selected
        if str(record["filename"]) not in monitor_names
    ]
    smoke_train_class_counts: Counter[int] = Counter()
    for record in smoke_train:
        smoke_train_class_counts.update(int(value) for value in record["class_ids"])
    if set(smoke_train_class_counts) != set(range(len(class_names))):
        raise ValueError("smoke training view does not cover all classes")
    external_train = [
        record for record in records if str(record["filename"]) not in monitor_names
    ]
    smoke_root = output_root / "external_smoke"
    monitor_root = output_root / "external_monitor"
    for record in smoke_selected:
        filename = str(record["filename"])
        hardlink_same_volume(
            external_images / filename, smoke_root / "images" / "train" / filename
        )
        label_name = f"{Path(filename).stem}.txt"
        hardlink_same_volume(
            external_labels / label_name,
            smoke_root / "labels" / "train" / label_name,
        )
    for record in monitor_selected:
        filename = str(record["filename"])
        hardlink_same_volume(
            external_images / filename,
            monitor_root / "images" / "val" / filename,
        )
        label_name = f"{Path(filename).stem}.txt"
        hardlink_same_volume(
            external_labels / label_name,
            monitor_root / "labels" / "val" / label_name,
        )

    external_train_list = output_root / "external_train.txt"
    external_train_list.write_text(
        "\n".join(
            (external_images / str(record["filename"])).resolve().as_posix()
            for record in external_train
        )
        + "\n",
        encoding="utf-8",
    )
    smoke_train_list = output_root / "external_smoke_train.txt"
    smoke_train_list.write_text(
        "\n".join(
            (smoke_root / "images" / "train" / str(record["filename"]))
            .resolve()
            .as_posix()
            for record in smoke_train
        )
        + "\n",
        encoding="utf-8",
    )
    monitor_images = (monitor_root / "images" / "val").resolve().as_posix()

    write_dataset_yaml(
        output_root / "external.yaml",
        output_root / "external",
        external_train_list.resolve().as_posix(),
        monitor_images,
        class_names,
    )
    write_dataset_yaml(
        output_root / "external_smoke.yaml",
        smoke_root,
        smoke_train_list.resolve().as_posix(),
        monitor_images,
        class_names,
    )
    write_dataset_yaml(
        output_root / "target.yaml",
        target_root,
        "images/train",
        monitor_images,
        class_names,
    )
    manifest: dict[str, object] = {
        "schema_version": "poker_dealer.card_finetune_view.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "training_only_no_independent_validation",
        "classes": list(class_names),
        "external": {
            "snapshot_manifest": str(source_manifest.resolve()),
            "snapshot_manifest_sha256": sha256_file(source_manifest),
            "images": len(records),
            "objects": sum(class_counts.values()),
            "class_object_counts": {
                class_names[class_id]: class_counts[class_id]
                for class_id in range(len(class_names))
            },
            "image_storage": "NTFS hardlinks to immutable raw snapshot",
            "annotation_conversion": "Pascal VOC to YOLO by normalized card code",
        },
        "external_smoke": {
            "images": len(smoke_train),
            "objects": sum(int(record["objects"]) for record in smoke_train),
            "class_coverage": len(smoke_train_class_counts),
        },
        "external_monitor": {
            "status": "training_health_only_not_target_or_gate_validation",
            "images": len(monitor_selected),
            "objects": sum(int(record["objects"]) for record in monitor_selected),
            "class_coverage": len(monitor_class_counts),
            "excluded_from_external_train": True,
        },
        "target": {
            "manifest": str(target_manifest.resolve()),
            "manifest_sha256": sha256_file(target_manifest),
            "dataset_yaml": str((output_root / "target.yaml").resolve()),
        },
    }
    manifest_path = output_root / "view_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--smoke-images", type=int, default=520)
    args = parser.parse_args(argv)
    if args.smoke_images < 52:
        parser.error("--smoke-images must be at least 52")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) != 52:
        raise ValueError("expected the pinned 52-class JSON list")
    manifest = build_view(
        args.external.resolve(),
        args.target.resolve(),
        args.output.resolve(),
        class_names,
        args.smoke_images,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
