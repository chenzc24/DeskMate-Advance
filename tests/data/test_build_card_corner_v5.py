from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.data.build_card_corner_v5 import (
    TRAIN_SESSIONS,
    VALIDATION_SESSIONS,
    hamming_distance,
    image_dhash,
)
from scripts.data.augment_poker_cards import YoloBox, detect_card_region


def test_session_split_keeps_complete_capture_sessions_apart() -> None:
    assert TRAIN_SESSIONS == ("heng", "slope1")
    assert VALIDATION_SESSIONS == ("slope2",)
    assert set(TRAIN_SESSIONS).isdisjoint(VALIDATION_SESSIONS)


def test_hamming_distance_counts_changed_bits() -> None:
    assert hamming_distance(0b0000, 0b1111) == 4
    assert hamming_distance(0b1010, 0b0011) == 2
    assert hamming_distance(123, 123) == 0


def test_image_dhash_is_stable_and_detects_direction(tmp_path: Path) -> None:
    horizontal = np.tile(np.arange(18, dtype=np.uint8), (16, 1))
    reversed_horizontal = np.fliplr(horizontal)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), horizontal)
    assert cv2.imwrite(str(second), reversed_horizontal)

    assert image_dhash(first) == image_dhash(first)
    assert hamming_distance(image_dhash(first), image_dhash(second)) == 64


def test_card_region_handles_two_boxes_in_one_physical_corner() -> None:
    image = np.full((480, 640, 3), (30, 100, 30), dtype=np.uint8)
    card = cv2.boxPoints(((340.0, 260.0), (260.0, 100.0), -8.0))
    cv2.fillConvexPoly(image, np.round(card).astype(np.int32), (235, 235, 235))
    boxes = (
        YoloBox(0, 0.40, 0.50, 0.04, 0.05),
        YoloBox(0, 0.43, 0.52, 0.04, 0.05),
    )

    mask, quad = detect_card_region(image, boxes)

    assert cv2.countNonZero(mask) > 20_000
    assert quad.shape == (4, 2)
