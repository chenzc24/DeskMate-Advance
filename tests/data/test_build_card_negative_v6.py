from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.data.build_card_negative_v6 import augment_negative, sha256_file


def test_negative_augmentation_is_deterministic_and_preserves_shape() -> None:
    image = np.full((180, 240, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (180, 140), (0, 0, 255), -1)

    first, first_metadata = augment_negative(image, 20260725)
    second, second_metadata = augment_negative(image, 20260725)

    assert np.array_equal(first, second)
    assert first.shape == image.shape
    assert first_metadata == second_metadata
    assert first_metadata["mirrored"] is False
    assert -25.0 <= float(first_metadata["angle_degrees"]) <= 25.0
    assert 0.92 <= float(first_metadata["scale"]) <= 1.08


def test_empty_yolo_label_has_stable_hash(tmp_path: Path) -> None:
    label = tmp_path / "negative.txt"
    label.write_bytes(b"")

    assert label.stat().st_size == 0
    assert (
        sha256_file(label)
        == "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
