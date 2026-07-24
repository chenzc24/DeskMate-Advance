from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.data.build_chip_v2_optimization_view import (
    Box,
    augment_image,
    capture_group,
    declared_denomination,
    intersection_over_union,
    transform_boxes,
)


def test_capture_groups_keep_complete_sequences_together() -> None:
    assert capture_group(Path("chip_v2/10v11.png")) == "chip_v2:10"
    assert capture_group(Path("chip_v2/20v2..png")) == "chip_v2:20"
    assert capture_group(Path("chip_v2/mixv6.png")) == "chip_v2:mix"
    assert capture_group(Path("straight/10vv1.png")) == "straight:10"


def test_declared_denomination_does_not_invent_mixed_labels() -> None:
    assert declared_denomination(Path("chip_v2/10v1.png")) == "10"
    assert declared_denomination(Path("straight/20vv1.png")) == "20"
    assert declared_denomination(Path("chip_v2/mixv1.png")) is None


def test_box_conversion_and_iou() -> None:
    box = Box(10, 20, 50, 60)
    assert box.to_yolo(100, 100) == (0.3, 0.4, 0.4, 0.4)
    assert intersection_over_union(box, box) == 1.0
    assert intersection_over_union(box, Box(60, 60, 70, 70)) == 0.0


def test_identity_homography_preserves_boxes() -> None:
    boxes = [Box(10, 20, 50, 60)]
    result = transform_boxes(boxes, np.eye(3, dtype=np.float64), 100, 100)
    assert result == boxes


def test_augmentation_is_deterministic_and_keeps_instances() -> None:
    image = np.full((240, 320, 3), (30, 160, 40), dtype=np.uint8)
    cv2.circle(image, (160, 150), 35, (220, 220, 220), -1)
    boxes = [Box(120, 110, 200, 190)]
    first_image, first_boxes, first_metadata = augment_image(image, boxes, 12345)
    second_image, second_boxes, second_metadata = augment_image(image, boxes, 12345)
    assert np.array_equal(first_image, second_image)
    assert first_boxes == second_boxes
    assert first_metadata == second_metadata
    assert len(first_boxes) == len(boxes)
    assert first_boxes[0].valid(6)
