"""Audit and build a reviewed YOLO view for the 2026-07-24 chip-v2 capture.

The source capture is immutable.  This helper creates deterministic manifests,
classical ellipse proposals and, once a reviewed annotation JSON is supplied,
a replay dataset layered on top of the existing hard-negative-v3 dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/raw/chips/2026-07-24-chip-v2-source"
DEFAULT_WORK = ROOT / "data/work/chips/2026-07-24-chip-v2-optimization"
DEFAULT_BASE = ROOT / "data/work/chips/2026-07-23-localization-hard-negative-v1"
DEFAULT_ANNOTATIONS = ROOT / "configs/perception/chip_v2_annotations.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MANUAL_ADDITIONS: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {
    "chip_v2/10v11.png": [("10", (588, 726, 818, 890))],
    "chip_v2/10v2.png": [("10", (580, 684, 792, 852))],
    "chip_v2/10v3.png": [("10", (824, 694, 1022, 856))],
    "chip_v2/10v9.png": [("10", (474, 686, 693, 861))],
    "chip_v2/20v5.png": [("20", (510, 689, 725, 868))],
    "chip_v2/20v13.png": [("20", (499, 696, 710, 870))],
    "chip_v2/mixv1.png": [("10", (750, 689, 952, 852))],
    "chip_v2/mixv2.png": [
        ("10", (757, 691, 958, 854)),
        ("10", (402, 728, 637, 890)),
    ],
    "chip_v2/mixv3.png": [("10", (758, 689, 960, 854))],
    "chip_v2/mixv4.png": [("20", (543, 758, 792, 890))],
    "chip_v2/mixv5.png": [("10", (492, 707, 712, 890))],
    "chip_v2/mixv6.png": [
        ("10", (482, 701, 693, 883)),
        ("10", (724, 688, 927, 853)),
    ],
}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )

    def valid(self, minimum_side: float = 2.0) -> bool:
        return self.x2 - self.x1 >= minimum_side and self.y2 - self.y1 >= minimum_side

    def expanded(self, x_fraction: float, y_fraction: float) -> "Box":
        pad_x = (self.x2 - self.x1) * x_fraction
        pad_y = (self.y2 - self.y1) * y_fraction
        return Box(self.x1 - pad_x, self.y1 - pad_y, self.x2 + pad_x, self.y2 + pad_y)

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            (self.x1 + self.x2) / (2.0 * width),
            (self.y1 + self.y2) / (2.0 * height),
            (self.x2 - self.x1) / width,
            (self.y2 - self.y1) / height,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def capture_group(relative_path: Path) -> str:
    stem = relative_path.stem.lower().rstrip(".")
    parent = relative_path.parent.as_posix().lower()
    if stem.startswith("mix"):
        sequence = "mix"
    elif stem.startswith("10"):
        sequence = "10"
    elif stem.startswith("20"):
        sequence = "20"
    else:
        sequence = "unknown"
    return f"{parent}:{sequence}"


def declared_denomination(relative_path: Path) -> str | None:
    stem = relative_path.stem.lower().rstrip(".")
    if stem.startswith("10"):
        return "10"
    if stem.startswith("20"):
        return "20"
    return None


def audit(raw_root: Path, work_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}
    for path in image_paths(raw_root):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable image: {path}")
        height, width = image.shape[:2]
        relative = path.relative_to(raw_root)
        digest = sha256(path)
        hashes.setdefault(digest, []).append(relative.as_posix())
        records.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "capture_group": capture_group(relative),
                "declared_denomination": declared_denomination(relative),
            }
        )
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "chip-v2-source-20260724",
        "raw_root": str(raw_root.resolve()),
        "image_count": len(records),
        "exact_unique_count": len(hashes),
        "exact_duplicate_groups": duplicates,
        "records": records,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def intersection_over_union(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _ellipse_box(contour: np.ndarray, width: int, height: int) -> Box | None:
    if len(contour) < 5:
        return None
    (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if cy < height * 0.43 or minor < 8 or major > width * 0.20:
        return None
    if major / max(minor, 1.0) > 3.8:
        return None
    # The pale printed centre is roughly 55-72% of the physical chip diameter.
    chip_w = major * 1.72
    chip_h = minor * 1.72
    return Box(cx - chip_w / 2, cy - chip_h / 2, cx + chip_w / 2, cy + chip_h / 2).clipped(
        width, height
    )


def classical_proposals(image: np.ndarray) -> list[Box]:
    """Find pale centre ellipses; proposals are review aids, never ground truth."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Warm room lighting gives the cream centre substantial saturation.
    mask = np.where((saturation < 158) & (value > 92), 255, 0).astype(np.uint8)
    mask[: int(height * 0.43), :] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 45 or area > width * height * 0.025:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 5:
            continue
        fill = area / max(float(w * h), 1.0)
        if fill < 0.24:
            continue
        candidate = _ellipse_box(contour, width, height)
        if candidate is None or not candidate.valid():
            continue
        if any(intersection_over_union(candidate, old) > 0.45 for old in candidates):
            continue
        candidates.append(candidate)
    return sorted(candidates, key=lambda box: (box.y1, box.x1))


def blue_ring_proposals(image: np.ndarray) -> list[Box]:
    """Find the dark-blue connected ring of this capture's value-10 chips."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (88, 65, 25), (138, 255, 255))
    mask[: int(height * 0.43), :] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 90 or area > width * height * 0.04 or len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if cy < height * 0.48 or minor < 8 or major / max(minor, 1.0) > 4.2:
            continue
        box = Box(
            cx - major * 0.62,
            cy - minor * 0.68,
            cx + major * 0.62,
            cy + minor * 0.68,
        ).clipped(width, height)
        if box.valid() and not any(intersection_over_union(box, old) > 0.45 for old in candidates):
            candidates.append(box)
    return sorted(candidates, key=lambda box: (box.y1, box.x1))


def pale_center_proposals(image: np.ndarray) -> list[Box]:
    """Separate touching chips through thick pale-centre distance peaks."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.where((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 105), 255, 0).astype(np.uint8)
    mask[: int(height * 0.48), :] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = np.where(distance >= 14.0, 255, 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)
    candidates: list[Box] = []
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        cx, cy = centroids[component]
        if area < 18 or cy < height * 0.55 or w < 5 or h < 4:
            continue
        if w > width * 0.14 or h > height * 0.13:
            continue
        # Peak dimensions approximate the pale centre after a 14-pixel erosion.
        centre_w = w + 28.0
        centre_h = h + 28.0
        chip_w = centre_w * 1.42
        chip_h = centre_h * 1.12
        box = Box(
            cx - chip_w / 2,
            cy - chip_h / 2,
            cx + chip_w / 2,
            cy + chip_h / 2,
        ).clipped(width, height)
        if box.valid(12) and not any(intersection_over_union(box, old) > 0.4 for old in candidates):
            candidates.append(box)
    return sorted(candidates, key=lambda box: (box.y1, box.x1))


def green_corner_fraction(image: np.ndarray, box: Box) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 35)
    )
    height, width = image.shape[:2]
    clipped = box.clipped(width, height)
    x1, y1, x2, y2 = (
        round(clipped.x1),
        round(clipped.y1),
        round(clipped.x2),
        round(clipped.y2),
    )
    crop = green[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    corners = (
        ((xx < crop.shape[1] * 0.20) | (xx > crop.shape[1] * 0.80))
        & ((yy < crop.shape[0] * 0.20) | (yy > crop.shape[0] * 0.80))
    )
    return float(crop[corners].mean()) if corners.any() else 0.0


def infer_mixed_denomination(image: np.ndarray, box: Box) -> str:
    height, width = image.shape[:2]
    clipped = box.clipped(width, height)
    x1, y1, x2, y2 = map(
        round, (clipped.x1, clipped.y1, clipped.x2, clipped.y2)
    )
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[:, :, 0] >= 88)
        & (hsv[:, :, 0] <= 138)
        & (hsv[:, :, 1] >= 65)
        & (hsv[:, :, 2] >= 25)
    )
    return "10" if float(blue.mean()) >= 0.055 else "20"


def reviewed_candidate(raw_root: Path, work_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    previews: list[tuple[str, np.ndarray]] = []
    for path in image_paths(raw_root):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        height, width = image.shape[:2]
        relative = path.relative_to(raw_root)
        relative_key = relative.as_posix()
        instances: list[dict[str, object]] = []
        for box in pale_center_proposals(image):
            if box.x2 - box.x1 < 75 or box.y2 - box.y1 < 45:
                continue
            if green_corner_fraction(image, box) <= 0.70:
                continue
            box = box.expanded(0.06, 0.08).clipped(width, height)
            denomination = declared_denomination(relative)
            if denomination is None:
                denomination = infer_mixed_denomination(image, box)
            instances.append(
                {
                    "denomination": denomination,
                    "box_xyxy": [round(value, 2) for value in (box.x1, box.y1, box.x2, box.y2)],
                    "review_origin": "pale-centre-proposal-reviewed",
                }
            )
        for denomination, coordinates in MANUAL_ADDITIONS.get(relative_key, []):
            box = Box(*map(float, coordinates)).clipped(width, height)
            instances.append(
                {
                    "denomination": denomination,
                    "box_xyxy": list(coordinates),
                    "review_origin": "manual-visual-correction",
                }
            )
        records.append(
            {
                "relative_path": relative_key,
                "source_sha256": sha256(path),
                "width": width,
                "height": height,
                "capture_group": capture_group(relative),
                "instances": instances,
            }
        )
        drawn = image.copy()
        for instance in instances:
            x1, y1, x2, y2 = map(round, instance["box_xyxy"])
            denomination = str(instance["denomination"])
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                drawn,
                denomination,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        previews.append((relative_key, drawn))
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "annotation_id": "chip-v2-reviewed-20260724-v1",
        "class_map": {"0": "poker_chip"},
        "denomination_scope": ["10", "20"],
        "records": records,
    }
    (work_root / "reviewed_annotations_candidate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_sheet(previews, work_root / "reviewed_annotations_contact_sheet.jpg")
    return payload


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_yolo_label(path: Path, boxes: list[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        x, y, w, h = box.clipped(width, height).to_yolo(width, height)
        if min(x, y, w, h) < 0 or max(x, y, w, h) > 1:
            raise ValueError(f"out-of-range label for {path}")
        lines.append(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def transform_boxes(boxes: list[Box], matrix: np.ndarray, width: int, height: int) -> list[Box]:
    transformed: list[Box] = []
    for box in boxes:
        corners = np.asarray(
            [[[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(corners, matrix)[0]
        candidate = Box(
            float(np.min(warped[:, 0])),
            float(np.min(warped[:, 1])),
            float(np.max(warped[:, 0])),
            float(np.max(warped[:, 1])),
        ).clipped(width, height)
        if candidate.valid(6):
            transformed.append(candidate)
    return transformed


def augment_image(
    image: np.ndarray, boxes: list[Box], seed: int
) -> tuple[np.ndarray, list[Box], dict[str, object]]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    angle = float(rng.uniform(-17.0, 17.0))
    scale = float(rng.uniform(0.88, 1.10))
    translate_x = float(rng.uniform(-0.055, 0.055) * width)
    translate_y = float(rng.uniform(-0.045, 0.045) * height)
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    affine[:, 2] += (translate_x, translate_y)
    affine_h = np.vstack([affine, [0.0, 0.0, 1.0]])
    source_corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    jitter = np.column_stack(
        [
            rng.uniform(-0.022, 0.022, 4) * width,
            rng.uniform(-0.018, 0.018, 4) * height,
        ]
    ).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(source_corners, source_corners + jitter)
    matrix = perspective @ affine_h
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-4.0, 4.0)) % 180.0
    hsv[:, :, 1] *= rng.uniform(0.82, 1.18)
    hsv[:, :, 2] *= rng.uniform(0.72, 1.22)
    warped = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    gamma = float(rng.uniform(0.78, 1.28))
    lookup = np.asarray([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype(
        np.uint8
    )
    warped = cv2.LUT(warped, lookup)
    effect = str(rng.choice(["none", "blur", "noise", "shadow", "glare"]))
    if effect == "blur":
        warped = cv2.GaussianBlur(warped, (3, 3), float(rng.uniform(0.45, 1.0)))
    elif effect == "noise":
        noise = rng.normal(0.0, rng.uniform(2.0, 7.0), warped.shape)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif effect in {"shadow", "glare"}:
        overlay = np.zeros((height, width), dtype=np.float32)
        center = (round(rng.uniform(0, width)), round(rng.uniform(0, height)))
        axes = (round(rng.uniform(0.18, 0.45) * width), round(rng.uniform(0.12, 0.32) * height))
        cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(width, height) * 0.05)
        strength = rng.uniform(0.12, 0.28) * (-1 if effect == "shadow" else 1)
        warped = np.clip(
            warped.astype(np.float32) * (1.0 + overlay[:, :, None] * strength), 0, 255
        ).astype(np.uint8)
    transformed = transform_boxes(boxes, matrix, width, height)
    if len(transformed) != len(boxes):
        raise ValueError("augmentation clipped a reviewed chip instance")
    return warped, transformed, {
        "seed": seed,
        "angle_degrees": round(angle, 4),
        "scale": round(scale, 5),
        "translate_fraction": [round(translate_x / width, 5), round(translate_y / height, 5)],
        "perspective_jitter_fraction": [0.022, 0.018],
        "gamma": round(gamma, 5),
        "effect": effect,
    }


def build_dataset(
    raw_root: Path,
    work_root: Path,
    base_root: Path,
    annotations: dict[str, object],
    augmentations_per_image: int,
) -> dict[str, object]:
    dataset_root = work_root / "dataset"
    if dataset_root.exists():
        resolved = dataset_root.resolve()
        if resolved.parent != work_root.resolve():
            raise ValueError(f"refusing unexpected dataset cleanup: {resolved}")
        shutil.rmtree(resolved)
    copy_modes: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        source_images = base_root / split / "images"
        source_labels = base_root / split / "labels"
        destination_images = dataset_root / split / "images"
        destination_labels = dataset_root / split / "labels"
        count = 0
        for image_path in sorted(path for path in source_images.iterdir() if path.is_file()):
            label_path = source_labels / f"{image_path.stem}.txt"
            mode = _link_or_copy(image_path, destination_images / image_path.name)
            copy_modes[mode] = copy_modes.get(mode, 0) + 1
            if not label_path.is_file():
                raise ValueError(f"base label is missing: {label_path}")
            mode = _link_or_copy(label_path, destination_labels / label_path.name)
            copy_modes[mode] = copy_modes.get(mode, 0) + 1
            count += 1
        split_counts[split] = count

    new_records: list[dict[str, object]] = []
    for record in annotations["records"]:
        relative = Path(record["relative_path"])
        group = str(record["capture_group"])
        if group.startswith("straight:"):
            split = "template_only"
        elif group == "chip_v2:20":
            split = "valid"
        else:
            split = "train"
        source = raw_root / relative
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable reviewed source: {source}")
        height, width = image.shape[:2]
        boxes = [Box(*map(float, instance["box_xyxy"])) for instance in record["instances"]]
        if split == "template_only":
            new_records.append(
                {
                    "relative_path": relative.as_posix(),
                    "capture_group": group,
                    "split": split,
                    "instances": len(boxes),
                }
            )
            continue
        repetitions = augmentations_per_image if split == "train" else 1
        for repetition in range(repetitions):
            output_image = image
            output_boxes = boxes
            augmentation: dict[str, object] | None = None
            if repetition:
                seed = int(
                    hashlib.sha256(f"{record['source_sha256']}:{repetition}".encode()).hexdigest()[:8],
                    16,
                )
                output_image, output_boxes, augmentation = augment_image(image, boxes, seed)
            base_name = (
                f"chip_v2_{relative.parent.name}_{relative.stem.rstrip('.')}_"
                f"{record['source_sha256'][:10]}_a{repetition:02d}"
            )
            image_output = dataset_root / split / "images" / f"{base_name}.jpg"
            label_output = dataset_root / split / "labels" / f"{base_name}.txt"
            image_output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(image_output), output_image, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to write: {image_output}")
            _write_yolo_label(label_output, output_boxes, width, height)
            split_counts[split] += 1
            new_records.append(
                {
                    "relative_path": relative.as_posix(),
                    "source_sha256": record["source_sha256"],
                    "capture_group": group,
                    "split": split,
                    "output_image": image_output.relative_to(dataset_root).as_posix(),
                    "output_image_sha256": sha256(image_output),
                    "instances": len(output_boxes),
                    "augmentation": augmentation,
                }
            )
    data_yaml = (
        f"path: {dataset_root.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: poker_chip\n"
    )
    (dataset_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    target_validation_images = [
        str((dataset_root / record["output_image"]).resolve())
        for record in new_records
        if record["split"] == "valid"
    ]
    target_list = dataset_root / "target_validation.txt"
    target_list.write_text("\n".join(target_validation_images) + "\n", encoding="utf-8")
    target_yaml = (
        f"path: {dataset_root.resolve()}\n"
        "train: target_validation.txt\n"
        "val: target_validation.txt\n"
        "names:\n"
        "  0: poker_chip\n"
    )
    (dataset_root / "target_validation.yaml").write_text(target_yaml, encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "poker-chip-localization-chip-v2-20260724-v1",
        "class_map": {"0": "poker_chip"},
        "base_dataset": str(base_root.resolve()),
        "base_manifest_sha256": sha256(base_root / "dataset_manifest.json"),
        "source_manifest_sha256": sha256(work_root / "source_manifest.json"),
        "annotation_policy": "reviewed pale-centre proposals plus explicit visual corrections",
        "split_policy": (
            "base splits preserved; chip_v2:10 and chip_v2:mix complete sequences train-only; "
            "chip_v2:20 complete sequence target validation; straight sequences template-only; "
            "augmentations remain with their source split"
        ),
        "augmentations_per_train_source_including_original": augmentations_per_image,
        "split_image_counts": split_counts,
        "target_validation_image_count": len(target_validation_images),
        "target_validation_list_sha256": sha256(target_list),
        "copy_modes": copy_modes,
        "new_records": new_records,
    }
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_proposal_review(raw_root: Path, work_root: Path) -> None:
    review_dir = work_root / "proposal_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    thumbs: list[tuple[str, np.ndarray]] = []
    for path in image_paths(raw_root):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        relative = path.relative_to(raw_root)
        boxes = classical_proposals(image)
        blue_boxes = blue_ring_proposals(image)
        centre_boxes = pale_center_proposals(image)
        drawn = image.copy()
        for index, box in enumerate(boxes):
            cv2.rectangle(
                drawn,
                (round(box.x1), round(box.y1)),
                (round(box.x2), round(box.y2)),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                drawn,
                f"C{index}",
                (round(box.x1), max(18, round(box.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for index, box in enumerate(blue_boxes):
            cv2.rectangle(
                drawn,
                (round(box.x1), round(box.y1)),
                (round(box.x2), round(box.y2)),
                (255, 0, 255),
                2,
            )
            cv2.putText(
                drawn,
                f"B{index}",
                (round(box.x1), max(18, round(box.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        for index, box in enumerate(centre_boxes):
            cv2.rectangle(
                drawn,
                (round(box.x1), round(box.y1)),
                (round(box.x2), round(box.y2)),
                (255, 255, 0),
                2,
            )
            cv2.putText(
                drawn,
                f"P{index}",
                (round(box.x1), max(18, round(box.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        output_name = relative.as_posix().replace("/", "__")
        cv2.imwrite(str(review_dir / output_name), drawn)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "proposals": [[box.x1, box.y1, box.x2, box.y2] for box in boxes],
                "blue_ring_proposals": [
                    [box.x1, box.y1, box.x2, box.y2] for box in blue_boxes
                ],
                "pale_center_proposals": [
                    [box.x1, box.y1, box.x2, box.y2] for box in centre_boxes
                ],
            }
        )
        thumbs.append((relative.as_posix(), drawn))
    (work_root / "classical_proposals.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_sheet(thumbs, work_root / "classical_proposals_contact_sheet.jpg")


def write_contact_sheet(items: Iterable[tuple[str, np.ndarray]], output: Path) -> None:
    entries = list(items)
    cell_width, cell_height, header = 320, 250, 25
    columns = 5
    rows = math.ceil(len(entries) / columns)
    canvas = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
    for index, (label, image) in enumerate(entries):
        scale = min(cell_width / image.shape[1], (cell_height - header) / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas[y + header : y + header + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            label,
            (x + 3, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--augmentations-per-image", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    work_root = args.work_root.resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"raw snapshot is missing: {raw_root}")
    manifest = audit(raw_root, work_root)
    if manifest["exact_duplicate_groups"]:
        raise SystemExit("exact duplicate source images require review")
    if not args.audit_only:
        write_proposal_review(raw_root, work_root)
        candidate = reviewed_candidate(raw_root, work_root)
        instance_count = sum(len(record["instances"]) for record in candidate["records"])
        dataset_manifest = (
            build_dataset(
                raw_root,
                work_root,
                args.base_root.resolve(),
                candidate,
                args.augmentations_per_image,
            )
            if args.build_dataset
            else None
        )
    else:
        instance_count = 0
    print(
        json.dumps(
            {
                "images": manifest["image_count"],
                "candidate_instances": instance_count,
                "dataset_split_counts": (
                    dataset_manifest["split_image_counts"] if dataset_manifest else None
                ),
                "work_root": str(work_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
