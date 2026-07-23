"""Finalize reviewed chip boxes, crop clean views, and create deterministic augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_ROOT = ROOT / "data" / "work" / "chips" / "2026-07-22-denominations"
DEFAULT_PRELABEL = SNAPSHOT_ROOT / "prelabel" / "prelabel_manifest.json"
DEFAULT_REVIEW = ROOT / "chip_recognition_workspace" / "chip_annotation_review_v1.json"
DEFAULT_OUTPUT = SNAPSHOT_ROOT / "labeled_augmented_v1"
AUGMENTATION_SEED = 20260722
AUGMENTED_VARIANTS_PER_SOURCE = 5


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_png(path: Path, image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"could not encode PNG: {path}")
    contents = encoded.tobytes()
    with path.open("xb") as stream:
        stream.write(contents)
    return sha256_bytes(contents)


def write_label(path: Path, class_id: int, box_xywh: list[int], image_shape: tuple[int, int]) -> str:
    height, width = image_shape
    x, y, box_width, box_height = box_xywh
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    contents = (
        f"{class_id} {center_x:.8f} {center_y:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(contents)
    return sha256_bytes(contents)


def validate_box(box: list[int], width: int, height: int, record_id: str) -> None:
    if len(box) != 4 or any(not isinstance(value, int) for value in box):
        raise SystemExit(f"invalid integer xywh box for {record_id}: {box}")
    x, y, box_width, box_height = box
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise SystemExit(f"invalid positive xywh box for {record_id}: {box}")
    if x + box_width > width or y + box_height > height:
        raise SystemExit(f"box exceeds image bounds for {record_id}: {box}")


def crop_view(image: np.ndarray, box: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    height, width = image.shape[:2]
    x, y, box_width, box_height = box
    margin_x = max(24, int(round(box_width * 0.55)))
    margin_y = max(24, int(round(box_height * 0.55)))
    crop_x1 = max(0, x - margin_x)
    crop_y1 = max(0, y - margin_y)
    crop_x2 = min(width, x + box_width + margin_x)
    crop_y2 = min(height, y + box_height + margin_y)
    crop = np.ascontiguousarray(image[crop_y1:crop_y2, crop_x1:crop_x2])
    crop_box = [x - crop_x1, y - crop_y1, box_width, box_height]
    return crop, crop_box, [crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1]


def transform_box(matrix: np.ndarray, box: list[int], width: int, height: int) -> list[int]:
    x, y, box_width, box_height = box
    corners = np.array(
        [[[x, y], [x + box_width, y], [x + box_width, y + box_height], [x, y + box_height]]],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, matrix)[0]
    x1 = max(0, int(math.floor(float(transformed[:, 0].min()))))
    y1 = max(0, int(math.floor(float(transformed[:, 1].min()))))
    x2 = min(width, int(math.ceil(float(transformed[:, 0].max()))))
    y2 = min(height, int(math.ceil(float(transformed[:, 1].max()))))
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def augment_view(
    image: np.ndarray,
    box: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, list[int], dict[str, object]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-22.0, 22.0)
    scale = rng.uniform(0.90, 1.10)
    translate_x = rng.uniform(-0.06, 0.06) * width
    translate_y = rng.uniform(-0.06, 0.06) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    augmented = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    augmented_box = transform_box(matrix, box, width, height)

    contrast = rng.uniform(0.82, 1.18)
    brightness = rng.uniform(-16.0, 16.0)
    augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=brightness)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = rng.uniform(0.88, 1.12)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blur_kernel = 0
    if rng.random() < 0.35:
        blur_kernel = rng.choice((3, 5))
        augmented = cv2.GaussianBlur(augmented, (blur_kernel, blur_kernel), 0)
    noise_sigma = rng.uniform(0.0, 4.0)
    if noise_sigma > 0.5:
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        noise = noise_rng.normal(0.0, noise_sigma, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(augmented), augmented_box, {
        "angle_degrees": angle,
        "scale": scale,
        "translate_x_pixels": translate_x,
        "translate_y_pixels": translate_y,
        "contrast": contrast,
        "brightness": brightness,
        "saturation": saturation,
        "blur_kernel": blur_kernel,
        "noise_sigma": noise_sigma,
    }


def draw_contact_sheet(records: list[dict[str, object]], output_path: Path) -> None:
    columns, rows = 4, 4
    cell_width, cell_height = 320, 270
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 24, np.uint8)
    for index, record in enumerate(records):
        image = cv2.imread(str(ROOT / record["image_path"]), cv2.IMREAD_COLOR)
        x, y, width, height = record["bbox_xywh"]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)
        label = f"{record['view_id']} {record['class_name']}"
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        scale = min(cell_width / image.shape[1], (cell_height - 28) / image.shape[0])
        resized = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        )
        row, column = divmod(index, columns)
        x_offset = column * cell_width + (cell_width - resized.shape[1]) // 2
        y_offset = row * cell_height + 28
        sheet[
            y_offset : y_offset + resized.shape[0],
            x_offset : x_offset + resized.shape[1],
        ] = resized
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelabel", type=Path, default=DEFAULT_PRELABEL)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prelabel_path = args.prelabel.resolve()
    review_path = args.review.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output snapshot already exists; refusing overwrite: {output}")

    prelabel = json.loads(prelabel_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if prelabel["source_snapshot_sha256"] != review["source_snapshot_sha256"]:
        raise SystemExit("annotation review does not match the source snapshot")
    if not review.get("accept_all_non_overridden_candidates"):
        raise SystemExit("annotation review must explicitly accept non-overridden candidates")
    expected_classes = {str(key): value for key, value in prelabel["class_names"].items()}
    if expected_classes != review["class_names"]:
        raise SystemExit("review class map does not match prelabel class map")

    labeled_images = output / "labeled_single_chip" / "images"
    labeled_labels = output / "labeled_single_chip" / "labels"
    augmented_images = output / "augmented" / "images"
    augmented_labels = output / "augmented" / "labels"
    qa_dir = output / "qa"
    for directory in (labeled_images, labeled_labels, augmented_images, augmented_labels, qa_dir):
        directory.mkdir(parents=True, exist_ok=False)

    overrides = review["manual_overrides_xywh"]
    labeled_records: list[dict[str, object]] = []
    augmented_records: list[dict[str, object]] = []
    class_counts = {value: 0 for value in review["class_names"].values()}
    for record in prelabel["records"]:
        record_id = record["record_id"]
        source_path = ROOT / record["source_path"]
        if sha256_file(source_path) != record["source_sha256"]:
            raise SystemExit(f"source image hash changed after review: {source_path}")
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"unreadable source image: {source_path}")
        height, width = image.shape[:2]
        selected = overrides.get(record_id, record["candidate_bbox_xywh"])
        if selected is None:
            raise SystemExit(f"reviewed record still has no annotation: {record_id}")
        box = [int(value) for value in selected]
        validate_box(box, width, height, record_id)
        crop, crop_box, crop_xywh = crop_view(image, box)
        crop_height, crop_width = crop.shape[:2]
        validate_box(crop_box, crop_width, crop_height, record_id)

        base_name = f"{record_id}_{source_path.stem}"
        crop_image_path = labeled_images / f"{base_name}.png"
        crop_label_path = labeled_labels / f"{base_name}.txt"
        image_hash = write_png(crop_image_path, crop)
        label_hash = write_label(
            crop_label_path, record["class_id"], crop_box, (crop_height, crop_width)
        )
        labeled_record = {
            "record_id": record_id,
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "annotation_origin": "manual_override" if record_id in overrides else "reviewed_candidate",
            "source_bbox_xywh": box,
            "crop_xywh": crop_xywh,
            "image_path": str(crop_image_path.relative_to(ROOT)).replace("\\", "/"),
            "image_sha256": image_hash,
            "label_path": str(crop_label_path.relative_to(ROOT)).replace("\\", "/"),
            "label_sha256": label_hash,
            "width": crop_width,
            "height": crop_height,
            "bbox_xywh": crop_box,
            "class_id": record["class_id"],
            "class_name": record["class_name"],
            "group_id": record_id,
        }
        labeled_records.append(labeled_record)
        class_counts[record["class_name"]] += 1

        for variant in range(AUGMENTED_VARIANTS_PER_SOURCE + 1):
            view_id = f"{record_id}_v{variant:02d}"
            if variant == 0:
                view = crop.copy()
                view_box = crop_box.copy()
                transform = {"type": "identity"}
            else:
                rng = random.Random(AUGMENTATION_SEED + int(record_id[1:]) * 100 + variant)
                view, view_box, parameters = augment_view(crop, crop_box, rng)
                transform = {"type": "affine_photometric", **parameters}
            view_image_path = augmented_images / f"{view_id}.png"
            view_label_path = augmented_labels / f"{view_id}.txt"
            view_image_hash = write_png(view_image_path, view)
            view_label_hash = write_label(
                view_label_path,
                record["class_id"],
                view_box,
                (view.shape[0], view.shape[1]),
            )
            augmented_records.append(
                {
                    "view_id": view_id,
                    "group_id": record_id,
                    "source_record_id": record_id,
                    "image_path": str(view_image_path.relative_to(ROOT)).replace("\\", "/"),
                    "image_sha256": view_image_hash,
                    "label_path": str(view_label_path.relative_to(ROOT)).replace("\\", "/"),
                    "label_sha256": view_label_hash,
                    "width": view.shape[1],
                    "height": view.shape[0],
                    "bbox_xywh": view_box,
                    "class_id": record["class_id"],
                    "class_name": record["class_name"],
                    "transform": transform,
                }
            )

    yaml_contents = {
        "path": str((output / "augmented").resolve()),
        "train": "images",
        "names": {int(key): value for key, value in review["class_names"].items()},
    }
    (output / "augmented" / "dataset_fit_only.yaml").write_text(
        yaml.safe_dump(yaml_contents, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "chip-denomination-single-foreground-2026-07-22-v1",
        "status": "development_fit_only_no_independent_validation",
        "source_snapshot_sha256": prelabel["source_snapshot_sha256"],
        "prelabel_manifest_sha256": sha256_file(prelabel_path),
        "annotation_review_sha256": sha256_file(review_path),
        "source_image_count": len(labeled_records),
        "labeled_box_count": len(labeled_records),
        "augmented_view_count": len(augmented_records),
        "classes": review["class_names"],
        "source_class_counts": class_counts,
        "augmentation": {
            "seed": AUGMENTATION_SEED,
            "variants_per_source_excluding_identity": AUGMENTED_VARIANTS_PER_SOURCE,
            "split_rule": "all views sharing group_id must remain in one split",
            "hue_shift": False,
            "horizontal_or_vertical_flip": False,
        },
        "limitations": review["limitations"],
        "labeled_records": labeled_records,
        "augmented_records": augmented_records,
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for sheet_index, start in enumerate(range(0, len(labeled_records), 16), 1):
        qa_records = [
            {
                **record,
                "view_id": record["record_id"],
            }
            for record in labeled_records[start : start + 16]
        ]
        draw_contact_sheet(qa_records, qa_dir / f"labeled_contact_{sheet_index:02d}.jpg")
    sample_rng = random.Random(AUGMENTATION_SEED)
    augmented_sample: list[dict[str, object]] = []
    for class_name in review["class_names"].values():
        class_records = [
            record for record in augmented_records if record["class_name"] == class_name
        ]
        augmented_sample.extend(sample_rng.sample(class_records, 8))
    sample_rng.shuffle(augmented_sample)
    for sheet_index, start in enumerate(range(0, len(augmented_sample), 16), 1):
        draw_contact_sheet(
            augmented_sample[start : start + 16],
            qa_dir / f"augmented_sample_contact_{sheet_index:02d}.jpg",
        )

    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "source_images": len(labeled_records),
                "labeled_boxes": len(labeled_records),
                "augmented_views": len(augmented_records),
                "source_class_counts": class_counts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
