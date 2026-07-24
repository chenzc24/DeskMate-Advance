from __future__ import annotations

import pytest

from scripts.data.auto_label_card_images import (
    box_iou,
    card_code_from_stem,
    select_detections,
    yolo_line,
)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("梅花10", "10C"),
        ("方片5", "5D"),
        ("红桃J", "JH"),
        ("黑桃A", "AS"),
    ],
)
def test_card_code_from_chinese_filename(stem: str, expected: str) -> None:
    assert card_code_from_stem(stem) == expected


def test_select_detections_prefers_expected_then_remaps_distinct_corner() -> None:
    detections = [
        {"class_id": 7, "confidence": 0.91, "xyxy": [10, 10, 30, 40]},
        {"class_id": 8, "confidence": 0.75, "xyxy": [80, 60, 100, 90]},
        {"class_id": 9, "confidence": 0.99, "xyxy": [11, 11, 31, 41]},
    ]

    selected = select_detections(detections, expected_class_id=7)

    assert len(selected) == 2
    assert selected[0]["class_id"] == 7
    assert selected[1]["class_id"] == 8
    assert selected[1]["class_remapped"] is True


def test_box_iou_and_yolo_line() -> None:
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    assert yolo_line(3, [10, 20, 30, 60], 100, 100) == (
        "3 0.20000000 0.40000000 0.20000000 0.40000000"
    )
