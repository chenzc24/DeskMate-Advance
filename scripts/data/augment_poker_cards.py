"""Build a reproducible, training-only YOLO view for small card glyphs.

The raw source bytes are never modified. Geometry is applied to images and
their YOLO boxes together; photometric degradation is then applied to images.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence
import warnings

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "poker_label" / "poker new data"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "poker_new_augmented_v1"
DEFAULT_CLASSES = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "lgd-cards-gen3"
    / "model.classes.json"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def parse(cls, line: str) -> "YoloBox":
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"expected 5 YOLO fields, got {len(fields)}: {line!r}")
        box = cls(int(fields[0]), *(float(value) for value in fields[1:]))
        values = (box.x_center, box.y_center, box.width, box.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite YOLO box: {line!r}")
        if not (0.0 <= box.x_center <= 1.0 and 0.0 <= box.y_center <= 1.0):
            raise ValueError(f"YOLO center outside image: {line!r}")
        if not (0.0 < box.width <= 1.0 and 0.0 < box.height <= 1.0):
            raise ValueError(f"invalid YOLO extent: {line!r}")
        return box

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


@dataclass(frozen=True)
class AugmentConfig:
    width: int = 960
    height: int = 720
    variants_per_image: int = 200
    total_variants: int | None = None
    profile: str = "train"
    seed: int = 20260723
    output_jpeg_quality: int = 92
    workers: int = 4

    def __post_init__(self) -> None:
        if self.variants_per_image <= 0:
            raise ValueError("variants_per_image must be positive")
        if self.total_variants is not None and self.total_variants <= 0:
            raise ValueError("total_variants must be positive")
        if self.profile not in {"train", "validation"}:
            raise ValueError("profile must be 'train' or 'validation'")


@dataclass(frozen=True)
class SourceItem:
    image_path: Path
    label_path: Path
    image_sha256: str
    label_sha256: str
    boxes: tuple[YoloBox, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    """Read an image through bytes so Windows Unicode paths are reliable."""
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return image


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("OpenCV could not encode JPEG")
    return encoded.tobytes()


def write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_boxes(path: Path, class_count: int) -> tuple[YoloBox, ...]:
    boxes = tuple(
        YoloBox.parse(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not boxes:
        raise ValueError(f"empty label file: {path}")
    for box in boxes:
        if not 0 <= box.class_id < class_count:
            raise ValueError(f"class {box.class_id} outside 0..{class_count - 1}: {path}")
    return boxes


def discover_sources(
    source_dir: Path, labels_dir: Path, class_count: int
) -> list[SourceItem]:
    image_paths = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"no source images found in {source_dir}")
    items: list[SourceItem] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label for {image_path.name}: {label_path}")
        boxes = load_boxes(label_path, class_count)
        if not 1 <= len(boxes) <= 2:
            raise ValueError(
                f"expected one or two visible corner-glyph boxes for "
                f"{image_path.name}, got {len(boxes)}"
            )
        if len({box.class_id for box in boxes}) != 1:
            raise ValueError(f"source corners disagree on class: {label_path}")
        items.append(
            SourceItem(
                image_path=image_path,
                label_path=label_path,
                image_sha256=sha256_file(image_path),
                label_sha256=sha256_file(label_path),
                boxes=boxes,
            )
        )
    return items


def _sample_scene_scale(
    rng: np.random.Generator, profile: str, variant: int
) -> tuple[float, str]:
    ranges = {
        "very_far": (0.16, 0.30),
        "far": (0.30, 0.50),
        "medium": (0.50, 0.72),
        "near": (0.72, 0.96),
    }
    if profile == "validation":
        scale_bin = ("very_far", "far", "medium", "near")[variant % 4]
    else:
        selector = float(rng.random())
        if selector < 0.30:
            scale_bin = "very_far"
        elif selector < 0.65:
            scale_bin = "far"
        elif selector < 0.90:
            scale_bin = "medium"
        else:
            scale_bin = "near"
    low, high = ranges[scale_bin]
    return float(rng.uniform(low, high)), scale_bin


def _sample_rotation(
    rng: np.random.Generator, profile: str, variant: int
) -> tuple[float, str]:
    orientation_index = (
        variant % 4 if profile == "validation" else int(rng.integers(0, 4))
    )
    center, orientation = (
        (0.0, "upright"),
        (90.0, "right"),
        (180.0, "inverted"),
        (-90.0, "left"),
    )[orientation_index]
    jitter = 8.0 if profile == "validation" else 20.0
    return float(center + rng.uniform(-jitter, jitter)), orientation


def detect_card_region(
    image: np.ndarray, boxes: Sequence[YoloBox]
) -> tuple[np.ndarray, np.ndarray]:
    """Find the bright card body that encloses both annotated corner glyphs."""
    height, width = image.shape[:2]
    centers = [
        (float(box.x_center * width), float(box.y_center * height)) for box in boxes
    ]
    center_x = [point[0] for point in centers]
    center_y = [point[1] for point in centers]
    span_x = max(max(center_x) - min(center_x), 24.0)
    span_y = max(max(center_y) - min(center_y), 32.0)
    search_x0 = max(0, int(min(center_x) - 0.58 * span_x))
    search_x1 = min(width, int(max(center_x) + 0.58 * span_x))
    search_y0 = max(0, int(min(center_y) - 0.44 * span_y))
    search_y1 = min(height, int(max(center_y) + 0.44 * span_y))
    if len(boxes) == 1:
        search_mask = np.full((height, width), 255, dtype=np.uint8)
    else:
        search_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(
            search_mask, (search_x0, search_y0), (search_x1, search_y1), 255, -1
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_neutral = np.where((value >= 145) & (saturation <= 165), 255, 0).astype(
        np.uint8
    )
    bright_neutral = cv2.bitwise_and(bright_neutral, search_mask)
    bright_neutral = cv2.morphologyEx(
        bright_neutral,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    bright_neutral = cv2.morphologyEx(
        bright_neutral,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(
        bright_neutral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = float(width * height)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 0.01 * image_area <= area <= 0.40 * image_area:
            continue
        signed_distances = [
            cv2.pointPolygonTest(contour, center, True) for center in centers
        ]
        if min(signed_distances) < -12.0:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        if min(rect_width, rect_height) < 40.0:
            continue
        aspect = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
        # The oblique Raspberry Pi view can foreshorten a physical card until
        # its image-space minimum rectangle is nearly square.
        if not 1.0 <= aspect <= 2.50:
            continue
        score = area + 1000.0 * sum(distance >= 0.0 for distance in signed_distances)
        candidates.append((score, contour))
    if not candidates:
        raise ValueError(
            "could not locate a card body enclosing the annotated corner glyphs"
        )
    contour = max(candidates, key=lambda item: item[0])[1]
    center, size, angle = cv2.minAreaRect(contour)
    expanded_rect = (center, (size[0] * 1.035, size[1] * 1.035), angle)
    card_quad = cv2.boxPoints(expanded_rect).astype(np.float32)
    card_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(card_mask, np.round(card_quad).astype(np.int32), 255)
    return card_mask, card_quad


def prepare_global_background(
    prepared_sources: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: AugmentConfig,
) -> np.ndarray:
    """Build one card-free background from all aligned capture-session frames."""
    frames = np.stack(
        [
            cv2.resize(image, (config.width, config.height), interpolation=cv2.INTER_AREA)
            for image, _, _ in prepared_sources
        ],
        axis=0,
    )
    masks = np.stack(
        [
            cv2.resize(
                mask,
                (config.width, config.height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
            for _, mask, _ in prepared_sources
        ],
        axis=0,
    )
    background = np.empty((config.height, config.width, 3), dtype=np.uint8)
    missing = np.zeros((config.height, config.width), dtype=np.uint8)
    for row_start in range(0, config.height, 48):
        row_end = min(config.height, row_start + 48)
        frame_block = frames[:, row_start:row_end].astype(np.float32)
        mask_block = masks[:, row_start:row_end, :, None]
        frame_block = np.where(mask_block, np.nan, frame_block)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(frame_block, axis=0)
        invalid = np.isnan(median[:, :, 0])
        missing[row_start:row_end][invalid] = 255
        median[invalid] = 0.0
        background[row_start:row_end] = np.clip(median, 0, 255).astype(np.uint8)
    if np.any(missing):
        missing = cv2.dilate(
            missing,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        background = cv2.inpaint(background, missing, 9.0, cv2.INPAINT_TELEA)
    return background


def compose_source_background(
    image: np.ndarray,
    card_mask: np.ndarray,
    global_background: np.ndarray,
    config: AugmentConfig,
) -> np.ndarray:
    resized = cv2.resize(
        image, (config.width, config.height), interpolation=cv2.INTER_AREA
    )
    resized_mask = cv2.resize(
        card_mask,
        (config.width, config.height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_mask = cv2.dilate(
        resized_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    alpha = cv2.GaussianBlur(resized_mask, (0, 0), sigmaX=7.0).astype(np.float32)
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)[:, :, None]
    background = (
        global_background.astype(np.float32) * alpha
        + resized.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(background, 0, 255).astype(np.uint8)


def _object_homography(
    source_width: int,
    source_height: int,
    card_quad: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
    scene_scale: float,
    rotation_degrees: float,
) -> np.ndarray:
    resize_homography = np.array(
        [
            [config.width / source_width, 0.0, 0.0],
            [0.0, config.height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    resized_quad = cv2.perspectiveTransform(
        card_quad.reshape(1, 4, 2), resize_homography
    ).reshape(4, 2)
    center = np.mean(resized_quad, axis=0)
    rotation_affine = cv2.getRotationMatrix2D(
        (float(center[0]), float(center[1])),
        rotation_degrees,
        scene_scale,
    )
    rotation_affine[:, 2] += np.array(
        [
            float(rng.uniform(-0.14, 0.14) * config.width),
            float(rng.uniform(-0.12, 0.12) * config.height),
        ]
    )
    rotation_homography = np.vstack(
        (rotation_affine, np.array([0.0, 0.0, 1.0]))
    )
    affine_homography = rotation_homography @ resize_homography
    affine_quad = cv2.perspectiveTransform(
        card_quad.reshape(1, 4, 2), affine_homography
    ).reshape(4, 2)
    quad_extent = np.ptp(affine_quad, axis=0)
    perspective = float(rng.uniform(0.0, 0.055))
    jitter = np.empty_like(affine_quad)
    jitter[:, 0] = rng.uniform(
        -quad_extent[0] * perspective,
        quad_extent[0] * perspective,
        size=4,
    )
    jitter[:, 1] = rng.uniform(
        -quad_extent[1] * perspective,
        quad_extent[1] * perspective,
        size=4,
    )
    perspective_adjustment = cv2.getPerspectiveTransform(
        affine_quad.astype(np.float32),
        (affine_quad + jitter).astype(np.float32),
    )
    return perspective_adjustment @ affine_homography


def _box_corners(box: YoloBox, width: int, height: int) -> np.ndarray:
    x0 = (box.x_center - box.width * 0.5) * width
    y0 = (box.y_center - box.height * 0.5) * height
    x1 = (box.x_center + box.width * 0.5) * width
    y1 = (box.y_center + box.height * 0.5) * height
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def transform_boxes(
    boxes: Sequence[YoloBox],
    homography: np.ndarray,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[YoloBox, ...] | None:
    transformed: list[YoloBox] = []
    for box in boxes:
        corners = _box_corners(box, source_width, source_height).reshape(1, 4, 2)
        warped = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
        raw_x0, raw_y0 = np.min(warped, axis=0)
        raw_x1, raw_y1 = np.max(warped, axis=0)
        raw_area = max(0.0, float(raw_x1 - raw_x0)) * max(
            0.0, float(raw_y1 - raw_y0)
        )
        x0 = float(np.clip(raw_x0, 0.0, output_width))
        y0 = float(np.clip(raw_y0, 0.0, output_height))
        x1 = float(np.clip(raw_x1, 0.0, output_width))
        y1 = float(np.clip(raw_y1, 0.0, output_height))
        clipped_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if (
            raw_area <= 0.0
            or clipped_area / raw_area < 0.70
            or x1 - x0 < 3.0
            or y1 - y0 < 3.0
        ):
            return None
        transformed.append(
            YoloBox(
                class_id=box.class_id,
                x_center=((x0 + x1) * 0.5) / output_width,
                y_center=((y0 + y1) * 0.5) / output_height,
                width=(x1 - x0) / output_width,
                height=(y1 - y0) / output_height,
            )
        )
    return tuple(transformed)


def sample_geometry(
    source_shape: tuple[int, ...],
    boxes: Sequence[YoloBox],
    card_quad: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
    variant: int,
) -> tuple[np.ndarray, tuple[YoloBox, ...], float, str, float, str]:
    source_height, source_width = source_shape[:2]
    for _ in range(30):
        scene_scale, scale_bin = _sample_scene_scale(rng, config.profile, variant)
        rotation_degrees, orientation = _sample_rotation(
            rng, config.profile, variant
        )
        homography = _object_homography(
            source_width,
            source_height,
            card_quad,
            config,
            rng,
            scene_scale,
            rotation_degrees,
        )
        transformed = transform_boxes(
            boxes,
            homography,
            source_width,
            source_height,
            config.width,
            config.height,
        )
        if transformed is not None:
            return (
                homography,
                transformed,
                scene_scale,
                scale_bin,
                rotation_degrees,
                orientation,
            )
    raise RuntimeError("could not sample an in-frame geometry after 30 attempts")


def _background_color(image: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate(
        (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]), axis=0
    )
    median = np.median(border, axis=0)
    return tuple(int(value) for value in median)


def _add_background_texture(
    image: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    height, width = image.shape[:2]
    low_res = rng.normal(0.0, float(rng.uniform(1.0, 5.0)), size=(9, 12)).astype(
        np.float32
    )
    texture = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(image.astype(np.float32) + texture[:, :, None], 0, 255).astype(
        np.uint8
    )


def _apply_shadow(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    if rng.random() >= 0.38:
        return image, False
    height, width = image.shape[:2]
    points = rng.uniform(
        [0.0, 0.0], [float(width - 1), float(height - 1)], size=(4, 2)
    ).astype(np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(rng.uniform(20.0, 70.0)))
    strength = float(rng.uniform(0.48, 0.82))
    factor = 1.0 - (mask.astype(np.float32) / 255.0) * (1.0 - strength)
    result = image.astype(np.float32) * factor[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8), True


def _apply_glare(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    if rng.random() >= 0.35:
        return image, False
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (
        int(rng.uniform(0.15, 0.85) * width),
        int(rng.uniform(0.15, 0.85) * height),
    )
    axes = (
        max(12, int(rng.uniform(0.04, 0.18) * width)),
        max(8, int(rng.uniform(0.02, 0.10) * height)),
    )
    cv2.ellipse(
        mask,
        center,
        axes,
        float(rng.uniform(-40.0, 40.0)),
        0,
        360,
        255,
        -1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(rng.uniform(8.0, 28.0)))
    alpha = (mask.astype(np.float32) / 255.0) * float(rng.uniform(0.10, 0.38))
    result = image.astype(np.float32) * (1.0 - alpha[:, :, None]) + 255.0 * alpha[
        :, :, None
    ]
    return np.clip(result, 0, 255).astype(np.uint8), True


def _apply_partial_occlusion(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    if rng.random() >= 0.22:
        return image, False
    height, width = image.shape[:2]
    box = boxes[int(rng.integers(0, len(boxes)))]
    box_width = max(3, int(box.width * width))
    box_height = max(3, int(box.height * height))
    center_x = int(box.x_center * width)
    center_y = int(box.y_center * height)
    occlusion_width = max(2, int(box_width * rng.uniform(0.15, 0.32)))
    occlusion_height = max(2, int(box_height * rng.uniform(0.18, 0.38)))
    center_x += int(rng.uniform(-0.35, 0.35) * box_width)
    center_y += int(rng.uniform(-0.35, 0.35) * box_height)
    color_choices = (
        (int(rng.integers(130, 205)), int(rng.integers(155, 220)), int(rng.integers(180, 235))),
        (int(rng.integers(30, 90)), int(rng.integers(40, 100)), int(rng.integers(45, 110))),
    )
    color = color_choices[int(rng.integers(0, len(color_choices)))]
    overlay = image.copy()
    cv2.ellipse(
        overlay,
        (center_x, center_y),
        (occlusion_width, occlusion_height),
        float(rng.uniform(-45.0, 45.0)),
        0,
        360,
        color,
        -1,
    )
    return cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0), True


def _motion_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.choice(np.array([3, 5, 7, 9])))
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    angle = float(rng.uniform(0.0, 180.0))
    rotation = cv2.getRotationMatrix2D((size * 0.5 - 0.5, size * 0.5 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0.0:
        kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)


def apply_photometric(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    working = image.astype(np.float32) / 255.0
    gamma = float(rng.uniform(0.72, 1.38))
    contrast = float(rng.uniform(0.72, 1.30))
    brightness = float(rng.uniform(-0.13, 0.13))
    working = np.power(np.clip(working, 0.0, 1.0), gamma)
    working = (working - 0.5) * contrast + 0.5 + brightness
    channel_gains = rng.uniform(0.88, 1.12, size=3).astype(np.float32)
    working *= channel_gains[None, None, :]
    result = np.clip(working * 255.0, 0, 255).astype(np.uint8)

    result, shadow = _apply_shadow(result, rng)
    result, glare = _apply_glare(result, rng)
    result, occlusion = _apply_partial_occlusion(result, boxes, rng)

    blur_selector = float(rng.random())
    blur_mode = "none"
    if blur_selector < 0.26:
        sigma = float(rng.uniform(0.45, 1.8))
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
        blur_mode = "gaussian"
    elif blur_selector < 0.44:
        result = _motion_blur(result, rng)
        blur_mode = "motion"
    elif blur_selector < 0.66:
        factor = float(rng.uniform(0.38, 0.78))
        small = cv2.resize(
            result,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA,
        )
        result = cv2.resize(
            small,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        blur_mode = "resample"

    noise_sigma = float(rng.uniform(0.0, 7.0))
    if noise_sigma > 0.8:
        noise = rng.normal(0.0, noise_sigma, size=result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    mjpeg_quality = int(rng.integers(38, 93))
    payload = encode_jpeg(result, mjpeg_quality)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("could not decode simulated MJPEG frame")
    metadata: dict[str, object] = {
        "gamma": round(gamma, 6),
        "contrast": round(contrast, 6),
        "brightness": round(brightness, 6),
        "channel_gains_bgr": [round(float(value), 6) for value in channel_gains],
        "shadow": shadow,
        "glare": glare,
        "partial_occlusion": occlusion,
        "blur": blur_mode,
        "noise_sigma": round(noise_sigma, 6),
        "mjpeg_quality": mjpeg_quality,
    }
    return decoded, metadata


def augment_one(
    source_image: np.ndarray,
    source_background: np.ndarray,
    card_mask: np.ndarray,
    card_quad: np.ndarray,
    source_boxes: Sequence[YoloBox],
    config: AugmentConfig,
    rng: np.random.Generator,
    variant: int,
) -> tuple[np.ndarray, tuple[YoloBox, ...], dict[str, object]]:
    (
        homography,
        boxes,
        scene_scale,
        scale_bin,
        rotation_degrees,
        orientation,
    ) = sample_geometry(
        source_image.shape, source_boxes, card_quad, config, rng, variant
    )
    background = cv2.resize(
        source_background, (config.width, config.height), interpolation=cv2.INTER_AREA
    )
    warped_card = cv2.warpPerspective(
        source_image,
        homography,
        (config.width, config.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        card_mask,
        homography,
        (config.width, config.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=0.8).astype(np.float32)
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)[:, :, None]
    composed = (
        warped_card.astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha)
    )
    composed = _add_background_texture(
        np.clip(composed, 0, 255).astype(np.uint8), rng
    )
    augmented, photometric = apply_photometric(composed, boxes, rng)
    metadata: dict[str, object] = {
        "scene_scale": round(scene_scale, 6),
        "scale_bin": scale_bin,
        "rotation_degrees": round(rotation_degrees, 6),
        "orientation": orientation,
        "homography": [
            [round(float(value), 8) for value in row] for row in homography
        ],
        **photometric,
    }
    return augmented, boxes, metadata


def _sample_seed(base_seed: int, source_sha256: str, variant: int) -> int:
    material = f"{base_seed}|{source_sha256}|{variant}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _safe_class_name(name: str) -> str:
    cleaned = "".join(character for character in name if character.isalnum())
    return cleaned or "class"


def _render_contact_sheet(
    output_path: Path,
    records: Sequence[dict[str, object]],
    images_dir: Path,
    labels_dir: Path,
    class_names: Sequence[str],
    columns: int = 8,
    tile_width: int = 240,
    tile_height: int = 180,
) -> None:
    if not records:
        return
    rows = math.ceil(len(records) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 32, np.uint8)
    for index, record in enumerate(records):
        filename = str(record["image"])
        image = read_image(images_dir / filename)
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        boxes = load_boxes(label_path, len(class_names))
        image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        for box in boxes:
            x0 = int((box.x_center - box.width * 0.5) * tile_width)
            y0 = int((box.y_center - box.height * 0.5) * tile_height)
            x1 = int((box.x_center + box.width * 0.5) * tile_width)
            y1 = int((box.y_center + box.height * 0.5) * tile_height)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 1)
        title = (
            f"{class_names[int(record['primary_class_id'])]} "
            f"{record['augmentation']['orientation']} "
            f"{record['augmentation']['scale_bin']} "
            f"{record['augmentation']['blur']}"
        )
        effects = "".join(
            code
            for key, code in (
                ("glare", "G"),
                ("shadow", "S"),
                ("partial_occlusion", "O"),
            )
            if record["augmentation"][key]
        )
        if effects:
            title = f"{title} {effects}"
        cv2.rectangle(image, (0, 0), (tile_width, 18), (0, 0, 0), -1)
        cv2.putText(
            image,
            title,
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = image
    write_bytes(output_path, encode_jpeg(sheet, 92))


def _representative_records(
    records: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        per_class[int(record["primary_class_id"])].append(record)
    all_classes: list[dict[str, object]] = []
    for class_id in sorted(per_class):
        candidates = [
            record
            for record in per_class[class_id]
            if record["augmentation"]["scale_bin"] == "far"
        ]
        all_classes.append((candidates or per_class[class_id])[0])

    modes: list[dict[str, object]] = []
    predicates = (
        lambda aug: aug["scale_bin"] == "near",
        lambda aug: aug["scale_bin"] == "medium",
        lambda aug: aug["scale_bin"] == "far",
        lambda aug: aug["scale_bin"] == "very_far",
        lambda aug: aug["orientation"] == "right",
        lambda aug: aug["orientation"] == "left",
        lambda aug: aug["orientation"] == "inverted",
        lambda aug: aug["blur"] == "gaussian",
        lambda aug: aug["blur"] == "motion",
        lambda aug: aug["blur"] == "resample",
        lambda aug: bool(aug["glare"]),
        lambda aug: bool(aug["shadow"]),
        lambda aug: bool(aug["partial_occlusion"]),
        lambda aug: int(aug["mjpeg_quality"]) <= 50,
    )
    for predicate in predicates:
        matches = [
            record for record in records if predicate(record["augmentation"])
        ]
        selected: list[dict[str, object]] = []
        selected_classes: set[int] = set()
        for record in matches:
            class_id = int(record["primary_class_id"])
            if class_id in selected_classes:
                continue
            selected.append(record)
            selected_classes.add(class_id)
            if len(selected) == 3:
                break
        modes.extend(selected)
    return all_classes, modes


def generate_dataset(
    source_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: Sequence[str],
    config: AugmentConfig,
    *,
    resume: bool = False,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"refusing to merge with non-empty output directory: {output_dir}"
        )
    images_dir = output_dir / "images" / "train"
    output_labels_dir = output_dir / "labels" / "train"
    review_dir = output_dir / "review"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(source_dir, labels_dir, len(class_names))
    source_classes = Counter(item.boxes[0].class_id for item in sources)
    missing_classes = sorted(set(range(len(class_names))) - set(source_classes))
    if missing_classes:
        raise ValueError(f"source set does not cover classes: {missing_classes}")
    if config.total_variants is None:
        source_variant_counts = [config.variants_per_image] * len(sources)
    else:
        base_count, remainder = divmod(config.total_variants, len(sources))
        if base_count == 0:
            raise ValueError("total_variants must be at least the source count")
        source_variant_counts = [
            base_count + int(index < remainder) for index in range(len(sources))
        ]

    records: list[dict[str, object]] = []
    class_image_counts: Counter[int] = Counter()
    scale_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    blur_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    if config.workers > 1:
        cv2.setNumThreads(1)
    prepared_sources: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for item in sources:
        source_image = read_image(item.image_path)
        try:
            card_mask, card_quad = detect_card_region(source_image, item.boxes)
        except ValueError as exc:
            raise ValueError(f"{item.image_path}: {exc}") from exc
        prepared_sources.append((source_image, card_mask, card_quad))
    global_background = prepare_global_background(prepared_sources, config)

    for source_index, item in enumerate(sources):
        source_image, card_mask, card_quad = prepared_sources[source_index]
        source_background = compose_source_background(
            source_image, card_mask, global_background, config
        )
        class_id = item.boxes[0].class_id
        class_name = str(class_names[class_id])
        source_token = item.image_sha256[:8]
        variant_count = source_variant_counts[source_index]

        def build_variant(variant: int) -> dict[str, object]:
            sample_seed = _sample_seed(config.seed, item.image_sha256, variant)
            rng = np.random.default_rng(sample_seed)
            image, boxes, augmentation = augment_one(
                source_image,
                source_background,
                card_mask,
                card_quad,
                item.boxes,
                config,
                rng,
                variant,
            )
            stem = (
                f"c{class_id:02d}_{_safe_class_name(class_name)}_"
                f"{source_token}_v{variant:04d}"
            )
            image_name = f"{stem}.jpg"
            label_name = f"{stem}.txt"
            image_payload = encode_jpeg(image, config.output_jpeg_quality)
            label_text = "\n".join(box.to_line() for box in boxes) + "\n"
            image_sha256 = write_bytes(images_dir / image_name, image_payload)
            label_payload = label_text.encode("utf-8")
            label_sha256 = write_bytes(output_labels_dir / label_name, label_payload)
            return {
                "image": image_name,
                "label": label_name,
                "primary_class_id": class_id,
                "primary_class_name": class_name,
                "source_index": source_index,
                "source_image": item.image_path.name,
                "sample_seed": sample_seed,
                "image_sha256": image_sha256,
                "label_sha256": label_sha256,
                "image_bytes": len(image_payload),
                "augmentation": augmentation,
            }

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            source_records = executor.map(
                build_variant, range(variant_count)
            )
            for record in source_records:
                augmentation = record["augmentation"]
                assert isinstance(augmentation, dict)
                records.append(record)
                class_image_counts[class_id] += 1
                scale_counts[str(augmentation["scale_bin"])] += 1
                orientation_counts[str(augmentation["orientation"])] += 1
                blur_counts[str(augmentation["blur"])] += 1
                for effect in ("glare", "shadow", "partial_occlusion"):
                    if augmentation[effect]:
                        effect_counts[effect] += 1
        print(
            f"[{source_index + 1:02d}/{len(sources):02d}] "
            f"class={class_name} variants={variant_count}",
            flush=True,
        )

    class_counts_named = {
        class_names[class_id]: class_image_counts[class_id]
        for class_id in range(len(class_names))
    }
    manifest: dict[str, object] = {
        "schema_version": "poker_dealer.card_augmentation_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "training_only_derived_view_not_independent_validation",
        "source": {
            "directory": str(source_dir.resolve()),
            "labels_directory": str(labels_dir.resolve()),
            "image_count": len(sources),
            "sources": [
                {
                    "image": item.image_path.name,
                    "label": item.label_path.name,
                    "image_sha256": item.image_sha256,
                    "label_sha256": item.label_sha256,
                    "class_id": item.boxes[0].class_id,
                    "class_name": class_names[item.boxes[0].class_id],
                    "box_count": len(item.boxes),
                    "variant_count": source_variant_counts[index],
                }
                for index, item in enumerate(sources)
            ],
        },
        "config": asdict(config),
        "classes": list(class_names),
        "summary": {
            "image_count": len(records),
            "label_count": len(records),
            "annotation_count": sum(
                len(item.boxes) * source_variant_counts[index]
                for index, item in enumerate(sources)
            ),
            "class_image_counts": class_counts_named,
            "scale_counts": dict(sorted(scale_counts.items())),
            "orientation_counts": dict(sorted(orientation_counts.items())),
            "blur_counts": dict(sorted(blur_counts.items())),
            "effect_counts": dict(sorted(effect_counts.items())),
        },
        "records": records,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    write_bytes(output_dir / "manifest.json", manifest_payload)
    dataset_description = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "validation": None,
        "validation_note": (
            "Collect independent target-camera sessions; do not split augmented "
            "siblings into validation."
        ),
        "class_count": len(class_names),
        "names": list(class_names),
    }
    write_bytes(
        output_dir / "dataset.json",
        (json.dumps(dataset_description, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    )

    all_classes, modes = _representative_records(records)
    _render_contact_sheet(
        review_dir / "all_52_classes_far.jpg",
        all_classes,
        images_dir,
        output_labels_dir,
        class_names,
    )
    _render_contact_sheet(
        review_dir / "augmentation_modes.jpg",
        modes,
        images_dir,
        output_labels_dir,
        class_names,
        columns=6,
    )
    return manifest


def validate_dataset(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = list(manifest["classes"])
    records = list(manifest["records"])
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    errors: list[str] = []
    class_counts: Counter[int] = Counter()
    annotation_count = 0
    image_names = {path.name for path in images_dir.glob("*.jpg")}
    label_names = {path.name for path in labels_dir.glob("*.txt")}
    for record in records:
        image_path = images_dir / str(record["image"])
        label_path = labels_dir / str(record["label"])
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"missing artifact for {record['image']}")
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
        source_index = int(record["source_index"])
        expected_box_count = int(
            manifest["source"]["sources"][source_index]["box_count"]
        )
        if len(boxes) != expected_box_count:
            errors.append(
                f"expected {expected_box_count} annotations: {label_path.name}"
            )
        annotation_count += len(boxes)
        for box in boxes:
            class_counts[box.class_id] += 1
            if not (
                box.x_center - box.width * 0.5 >= -1e-7
                and box.y_center - box.height * 0.5 >= -1e-7
                and box.x_center + box.width * 0.5 <= 1.0 + 1e-7
                and box.y_center + box.height * 0.5 <= 1.0 + 1e-7
            ):
                errors.append(f"box outside frame: {label_path.name}")
    expected_images = {str(record["image"]) for record in records}
    expected_labels = {str(record["label"]) for record in records}
    extras = sorted(image_names - expected_images) + sorted(label_names - expected_labels)
    if extras:
        errors.append(f"unexpected artifacts: {extras[:10]}")
    expected_by_class = {
        int(source["class_id"]): int(source["variant_count"])
        * int(source["box_count"])
        for source in manifest["source"]["sources"]
    }
    unbalanced = {
        class_names[class_id]: class_counts[class_id]
        for class_id in range(len(class_names))
        if class_counts[class_id] != expected_by_class[class_id]
    }
    if unbalanced:
        errors.append(f"class annotation imbalance: {unbalanced}")
    expected_values = set(expected_by_class.values())
    expected_annotations: int | dict[str, int]
    if len(expected_values) == 1:
        expected_annotations = next(iter(expected_values))
    else:
        expected_annotations = {
            class_names[class_id]: expected_by_class[class_id]
            for class_id in range(len(class_names))
        }
    result = {
        "valid": not errors,
        "errors": errors,
        "image_count": len(image_names),
        "label_count": len(label_names),
        "annotation_count": annotation_count,
        "class_count": len(class_names),
        "expected_annotations_per_class": expected_annotations,
    }
    return result


def _load_class_names(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"expected a non-empty JSON string array: {path}")
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate class names: {path}")
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--variants-per-image", type=int, default=200)
    parser.add_argument(
        "--total-variants",
        type=int,
        help="exact total output count distributed as evenly as possible",
    )
    parser.add_argument(
        "--profile",
        choices=("train", "validation"),
        default="train",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-jpeg-quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="overwrite deterministic artifacts in an interrupted output directory",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.variants_per_image <= 0:
        parser.error("--variants-per-image must be positive")
    if args.total_variants is not None and args.total_variants <= 0:
        parser.error("--total-variants must be positive")
    if args.width < 160 or args.height < 120:
        parser.error("output dimensions are too small")
    if not 40 <= args.output_jpeg_quality <= 100:
        parser.error("--output-jpeg-quality must be between 40 and 100")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output.resolve()
    if args.validate_only:
        result = validate_dataset(output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1
    source_dir = args.source.resolve()
    labels_dir = (args.labels or source_dir / "labels").resolve()
    class_names = _load_class_names(args.classes.resolve())
    config = AugmentConfig(
        width=args.width,
        height=args.height,
        variants_per_image=args.variants_per_image,
        total_variants=args.total_variants,
        profile=args.profile,
        seed=args.seed,
        output_jpeg_quality=args.output_jpeg_quality,
        workers=args.workers,
    )
    manifest = generate_dataset(
        source_dir,
        labels_dir,
        output_dir,
        class_names,
        config,
        resume=args.resume,
    )
    validation = validate_dataset(output_dir)
    print(
        json.dumps(
            {"summary": manifest["summary"], "validation": validation},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
