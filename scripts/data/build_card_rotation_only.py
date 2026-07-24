"""Generate exactly 520 single-card, rotation-only training images."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.augment_poker_cards import (
    SourceItem,
    YoloBox,
    detect_card_region,
    discover_sources,
    encode_jpeg,
    load_boxes,
    read_image,
    sha256_file,
    transform_boxes,
    write_bytes,
)


DEFAULT_SOURCE = ROOT / "data" / "raw" / "poker_label" / "new big poker"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "poker-dealer-v3"
    / "model.classes.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "work" / "poker_big_data_v3" / "rotation_only_520"
)
ROTATION_ANGLES = (0, 30, 60, 90, 135, 180, 225, 270, 300, 330)


def load_classes(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, list)
        or len(value) != 52
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("expected the pinned 52-class JSON list")
    return [str(item) for item in value]


def rotation_homography(
    card_quad: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    center = np.mean(card_quad, axis=0)
    affine = cv2.getRotationMatrix2D(
        (float(center[0]), float(center[1])),
        float(angle_degrees),
        1.0,
    )
    return np.vstack((affine, np.array([0.0, 0.0, 1.0])))


def compose_rotated_card(
    source_image: np.ndarray,
    source_background: np.ndarray,
    card_mask: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    height, width = source_image.shape[:2]
    warped_card = cv2.warpPerspective(
        source_image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        card_mask,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=0.8).astype(np.float32)
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)[:, :, None]
    composed = (
        warped_card.astype(np.float32) * alpha
        + source_background.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(composed, 0, 255).astype(np.uint8)


def restore_card_free_background(
    source_image: np.ndarray,
    card_mask: np.ndarray,
) -> np.ndarray:
    expanded_mask = cv2.dilate(
        card_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)),
        iterations=1,
    )
    return cv2.inpaint(source_image, expanded_mask, 3.0, cv2.INPAINT_TELEA)


def _safe_class_name(name: str) -> str:
    cleaned = "".join(character for character in name if character.isalnum())
    return cleaned or "class"


def _draw_boxes(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    label: str,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for box in boxes:
        x0 = int(round((box.x_center - box.width * 0.5) * width))
        y0 = int(round((box.y_center - box.height * 0.5) * height))
        x1 = int(round((box.x_center + box.width * 0.5) * width))
        y1 = int(round((box.y_center + box.height * 0.5) * height))
        cv2.rectangle(result, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.rectangle(result, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _render_contact_sheet(
    output_path: Path,
    records: Sequence[dict[str, object]],
    images_dir: Path,
    labels_dir: Path,
    class_names: Sequence[str],
    *,
    columns: int = 8,
    tile_width: int = 240,
    tile_height: int = 190,
) -> None:
    if not records:
        return
    rows = math.ceil(len(records) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    for index, record in enumerate(records):
        image = read_image(images_dir / str(record["image"]))
        boxes = load_boxes(labels_dir / str(record["label"]), len(class_names))
        image = _draw_boxes(
            image,
            boxes,
            f"{record['class_name']} {record['rotation_degrees']}deg",
        )
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        y0 = row * tile_height + (tile_height - resized.shape[0]) // 2
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    write_bytes(output_path, encode_jpeg(sheet, 92))


def generate_rotation_dataset(
    source_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    *,
    resume: bool = False,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    sources = discover_sources(
        source_root,
        source_root / "labels",
        len(class_names),
    )
    source_classes = Counter(item.boxes[0].class_id for item in sources)
    if source_classes != Counter({class_id: 1 for class_id in range(52)}):
        raise ValueError("source must contain exactly one image for every class")

    prepared: list[
        tuple[SourceItem, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for item in sources:
        image = read_image(item.image_path)
        mask, quad = detect_card_region(image, item.boxes)
        prepared.append((item, image, mask, quad))
    images_dir = output_root / "images" / "train"
    labels_dir = output_root / "labels" / "train"
    review_dir = output_root / "review"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source_index, (item, image, mask, quad) in enumerate(prepared):
        height, width = image.shape[:2]
        source_background = restore_card_free_background(image, mask)
        class_id = item.boxes[0].class_id
        class_name = str(class_names[class_id])
        source_token = item.image_sha256[:8]
        for angle in ROTATION_ANGLES:
            homography = rotation_homography(quad, angle)
            determinant = float(np.linalg.det(homography[:2, :2]))
            if not math.isclose(determinant, 1.0, abs_tol=1e-8):
                raise ValueError(f"rotation determinant is not unit: {determinant}")
            transformed_quad = cv2.perspectiveTransform(
                quad.reshape(1, 4, 2),
                homography,
            ).reshape(4, 2)
            body_cropped = bool(
                np.min(transformed_quad[:, 0]) < 0
                or np.max(transformed_quad[:, 0]) >= width
                or np.min(transformed_quad[:, 1]) < 0
                or np.max(transformed_quad[:, 1]) >= height
            )
            transformed_mask = cv2.warpPerspective(
                mask,
                homography,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            visible_fraction = min(
                1.0,
                float(
                    np.count_nonzero(transformed_mask)
                    / max(1, np.count_nonzero(mask))
                ),
            )
            if visible_fraction < 0.90:
                raise ValueError(
                    f"less than 90% of rotated card remains visible: "
                    f"{item.image_path.name} {angle} {visible_fraction:.3f}"
                )
            boxes = transform_boxes(
                item.boxes,
                homography,
                width,
                height,
                width,
                height,
            )
            if boxes is None or len(boxes) != len(item.boxes):
                raise ValueError(
                    f"could not transform both boxes: {item.image_path.name} {angle}"
                )
            result = compose_rotated_card(
                image,
                source_background,
                mask,
                homography,
            )
            stem = (
                f"c{class_id:02d}_{_safe_class_name(class_name)}_"
                f"{source_token}_r{angle:03d}"
            )
            image_name = f"{stem}.jpg"
            label_name = f"{stem}.txt"
            image_sha256 = write_bytes(
                images_dir / image_name,
                encode_jpeg(result, 95),
            )
            label_payload = (
                "\n".join(box.to_line() for box in boxes) + "\n"
            ).encode("utf-8")
            label_sha256 = write_bytes(
                labels_dir / label_name,
                label_payload,
            )
            records.append(
                {
                    "image": image_name,
                    "label": label_name,
                    "image_sha256": image_sha256,
                    "label_sha256": label_sha256,
                    "source_index": source_index,
                    "source_image": item.image_path.name,
                    "source_image_sha256": item.image_sha256,
                    "source_width": width,
                    "source_height": height,
                    "class_id": class_id,
                    "class_name": class_name,
                    "rotation_degrees": angle,
                    "mirror": False,
                    "scale": 1.0,
                    "translation_pixels": [0.0, 0.0],
                    "perspective": False,
                    "photometric_change": False,
                    "source_card_count": 1,
                    "output_card_count": 1,
                    "card_body_cropped_by_source_frame": body_cropped,
                    "card_visible_fraction": round(visible_fraction, 8),
                    "box_count": len(boxes),
                    "transform_determinant": determinant,
                    "homography": [
                        [round(float(value), 10) for value in row]
                        for row in homography
                    ],
                }
            )
        print(
            f"[{source_index + 1:02d}/{len(prepared):02d}] "
            f"class={class_name} variants={len(ROTATION_ANGLES)}",
            flush=True,
        )

    manifest = {
        "schema_version": "poker_dealer.card_rotation_only_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "training_only_not_independent_validation",
        "source": {
            "directory": str(source_root.resolve()),
            "image_count": len(sources),
            "sources": [
                {
                    "image": item.image_path.name,
                    "label": item.label_path.name,
                    "image_sha256": item.image_sha256,
                    "label_sha256": item.label_sha256,
                    "class_id": item.boxes[0].class_id,
                    "class_name": class_names[item.boxes[0].class_id],
                    "source_card_count": 1,
                    "box_count": len(item.boxes),
                }
                for item in sources
            ],
        },
        "contract": {
            "angles_degrees": list(ROTATION_ANGLES),
            "variants_per_source": len(ROTATION_ANGLES),
            "rotation_only": True,
            "mirror": False,
            "scale": 1.0,
            "translation_pixels": [0.0, 0.0],
            "perspective": False,
            "photometric_change": False,
            "background_restoration": (
                "per-source OpenCV Telea inpaint under original card mask only"
            ),
            "source_card_count": 1,
            "output_card_count": 1,
            "split_note": "keep all siblings of each source in one future split",
        },
        "classes": list(class_names),
        "summary": {
            "images": len(records),
            "labels": len(records),
            "annotations": sum(int(record["box_count"]) for record in records),
            "classes": len({int(record["class_id"]) for record in records}),
            "angle_counts": dict(
                sorted(
                    Counter(
                        int(record["rotation_degrees"]) for record in records
                    ).items()
                )
            ),
            "mirror_images": sum(bool(record["mirror"]) for record in records),
            "one_card_images": sum(
                int(record["output_card_count"]) == 1 for record in records
            ),
            "source_edge_cropped_images": sum(
                bool(record["card_body_cropped_by_source_frame"])
                for record in records
            ),
            "minimum_card_visible_fraction": min(
                float(record["card_visible_fraction"]) for record in records
            ),
        },
        "records": records,
    }
    write_bytes(
        output_root / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    _render_contact_sheet(
        review_dir / "all_52_at_90deg.jpg",
        [
            record
            for record in records
            if int(record["rotation_degrees"]) == 90
        ],
        images_dir,
        labels_dir,
        class_names,
    )
    representative_ids = {0, 17, 30, 47}
    _render_contact_sheet(
        review_dir / "angles_4_classes.jpg",
        [
            record
            for record in records
            if int(record["class_id"]) in representative_ids
        ],
        images_dir,
        labels_dir,
        class_names,
        columns=10,
    )
    return manifest


def validate_rotation_dataset(output_root: Path) -> dict[str, object]:
    manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    class_names = [str(value) for value in manifest["classes"]]
    records = list(manifest["records"])
    source_root = Path(str(manifest["source"]["directory"]))
    images_dir = output_root / "images" / "train"
    labels_dir = output_root / "labels" / "train"
    errors: list[str] = []
    counts_by_class: Counter[int] = Counter()
    counts_by_angle: Counter[int] = Counter()
    for source in manifest["source"]["sources"]:
        source_image = source_root / str(source["image"])
        source_label = source_root / "labels" / str(source["label"])
        if (
            not source_image.is_file()
            or sha256_file(source_image) != source["image_sha256"]
        ):
            errors.append(f"source image changed: {source['image']}")
        if (
            not source_label.is_file()
            or sha256_file(source_label) != source["label_sha256"]
        ):
            errors.append(f"source label changed: {source['label']}")
    for record in records:
        image_path = images_dir / str(record["image"])
        label_path = labels_dir / str(record["label"])
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"missing pair: {record['image']}")
            continue
        if sha256_file(image_path) != record["image_sha256"]:
            errors.append(f"image hash mismatch: {image_path.name}")
        if sha256_file(label_path) != record["label_sha256"]:
            errors.append(f"label hash mismatch: {label_path.name}")
        try:
            boxes = load_boxes(label_path, len(class_names))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        class_id = int(record["class_id"])
        if len(boxes) != 2 or {box.class_id for box in boxes} != {class_id}:
            errors.append(f"card corner contract failed: {label_path.name}")
        if (
            bool(record["mirror"])
            or float(record["scale"]) != 1.0
            or bool(record["perspective"])
            or bool(record["photometric_change"])
            or record["translation_pixels"] != [0.0, 0.0]
        ):
            errors.append(f"non-rotation transform: {record['image']}")
        if (
            int(record["source_card_count"]) != 1
            or int(record["output_card_count"]) != 1
        ):
            errors.append(f"card count changed: {record['image']}")
        if float(record["card_visible_fraction"]) < 0.90:
            errors.append(f"insufficient visible card area: {record['image']}")
        if not math.isclose(
            float(record["transform_determinant"]),
            1.0,
            abs_tol=1e-8,
        ):
            errors.append(f"non-unit determinant: {record['image']}")
        counts_by_class[class_id] += 1
        counts_by_angle[int(record["rotation_degrees"])] += 1
    image_count = len(list(images_dir.glob("*.jpg")))
    label_count = len(list(labels_dir.glob("*.txt")))
    if len(records) != 520 or image_count != 520 or label_count != 520:
        errors.append(
            f"expected 520 records/images/labels, got "
            f"{len(records)}/{image_count}/{label_count}"
        )
    if counts_by_class != Counter({class_id: 10 for class_id in range(52)}):
        errors.append("expected ten rotation variants for every class")
    if counts_by_angle != Counter({angle: 52 for angle in ROTATION_ANGLES}):
        errors.append("angle coverage is not 52 images per configured angle")
    return {
        "valid": not errors,
        "errors": errors,
        "records": len(records),
        "images": image_count,
        "labels": label_count,
        "annotations": sum(
            len(load_boxes(path, len(class_names)))
            for path in labels_dir.glob("*.txt")
        ),
        "classes": len(counts_by_class),
        "angles": dict(sorted(counts_by_angle.items())),
        "mirror_images": sum(bool(record["mirror"]) for record in records),
        "one_card_images": sum(
            int(record["output_card_count"]) == 1 for record in records
        ),
        "source_edge_cropped_images": sum(
            bool(record["card_body_cropped_by_source_frame"])
            for record in records
        ),
        "minimum_card_visible_fraction": min(
            float(record["card_visible_fraction"]) for record in records
        ),
        "source_hashes_valid": not any(
            error.startswith("source ") for error in errors
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    generate_rotation_dataset(
        args.source.resolve(),
        output,
        load_classes(args.classes.resolve()),
        resume=args.resume,
    )
    validation = validate_rotation_dataset(output)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return int(not validation["valid"])


if __name__ == "__main__":
    sys.exit(main())
