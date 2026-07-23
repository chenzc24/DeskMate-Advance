from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.data.augment_poker_cards import (
    AugmentConfig,
    YoloBox,
    generate_dataset,
    validate_dataset,
)


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_generate_and_validate_unicode_small_target_view(tmp_path: Path) -> None:
    source = tmp_path / "新数据"
    labels = source / "labels"
    labels.mkdir(parents=True)
    image = np.full((240, 320, 3), (70, 105, 70), dtype=np.uint8)
    cv2.rectangle(image, (100, 35), (220, 215), (245, 245, 245), -1)
    cv2.putText(
        image,
        "5D",
        (108, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    _write_png(source / "方片5.png", image)
    source_boxes = (
        YoloBox(0, 0.37, 0.22, 0.10, 0.12),
        YoloBox(0, 0.63, 0.78, 0.10, 0.12),
    )
    (labels / "方片5.txt").write_text(
        "\n".join(box.to_line() for box in source_boxes) + "\n", encoding="utf-8"
    )

    output = tmp_path / "derived"
    manifest = generate_dataset(
        source,
        labels,
        output,
        ["5D"],
        AugmentConfig(
            width=320,
            height=240,
            variants_per_image=4,
            profile="validation",
            seed=17,
            output_jpeg_quality=90,
        ),
    )

    assert manifest["summary"]["image_count"] == 4
    assert manifest["summary"]["annotation_count"] == 8
    assert set(manifest["summary"]["scale_counts"]) == {
        "very_far",
        "far",
        "medium",
        "near",
    }
    assert set(manifest["summary"]["orientation_counts"]) == {
        "upright",
        "right",
        "inverted",
        "left",
    }
    result = validate_dataset(output)
    assert result == {
        "valid": True,
        "errors": [],
        "image_count": 4,
        "label_count": 4,
        "annotation_count": 8,
        "class_count": 1,
        "expected_annotations_per_class": 8,
    }
