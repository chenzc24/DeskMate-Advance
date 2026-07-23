from __future__ import annotations

import cv2
import numpy as np

from chip_best_frame import ChipBestFrameSelector, frame_quality


def _detection(track_id=1, confidence=0.9):
    return {
        "track_id": track_id,
        "confidence": confidence,
        "bbox_xyxy": [20, 20, 100, 100],
    }


def _textured_frame():
    image = np.full((120, 120, 3), 100, dtype=np.uint8)
    for coordinate in range(20, 101, 8):
        cv2.line(image, (20, coordinate), (100, coordinate), (240, 40, 40), 2)
        cv2.line(image, (coordinate, 20), (coordinate, 100), (20, 220, 20), 2)
    return image


def test_frame_quality_prefers_sharp_view_over_blurred_view():
    sharp = _textured_frame()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 5)

    sharp_score, sharpness, _ = frame_quality(
        sharp, (20, 20, 100, 100), 0.9
    )
    blurred_score, blurred_sharpness, _ = frame_quality(
        blurred, (20, 20, 100, 100), 0.9
    )

    assert sharpness > blurred_sharpness
    assert sharp_score > blurred_score


def test_selector_emits_only_best_frame_after_bounded_track_window():
    selector = ChipBestFrameSelector(window_samples=3)
    sharp = _textured_frame()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 5)

    selector.observe(1, blurred, [_detection()])
    selector.observe(2, sharp, [_detection()])
    selector.observe(3, blurred, [_detection()])
    ready = selector.take_ready()

    assert len(ready) == 1
    assert ready[0].track_id == 1
    assert ready[0].source_frame == 2
    assert ready[0].source_bbox_xyxy == (20, 20, 100, 100)
    assert ready[0].image.shape[0] < sharp.shape[0]
    assert selector.take_ready() == ()


def test_selector_keeps_track_windows_independent():
    selector = ChipBestFrameSelector(window_samples=2)
    image = _textured_frame()

    selector.observe(1, image, [_detection(1), _detection(2)])
    selector.observe(2, image, [_detection(1), _detection(2)])

    assert [candidate.track_id for candidate in selector.take_ready()] == [1, 2]
