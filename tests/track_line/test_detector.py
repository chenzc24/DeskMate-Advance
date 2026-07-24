"""Deterministic synthetic tests for the OpenCV line detector."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from track_line.config import LineDetectorConfig
from track_line.detector import OpenCVLineDetector


WIDTH = 640
HEIGHT = 480


def _frame_with_line(points: list[tuple[int, int]], thickness: int = 34):
    frame = np.full((HEIGHT, WIDTH, 3), 235, dtype=np.uint8)
    cv2.polylines(
        frame,
        [np.asarray(points, dtype=np.int32)],
        False,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def _green_cloth_frame():
    return np.full((HEIGHT, WIDTH, 3), (50, 145, 45), dtype=np.uint8)


def _white_on_green_detector():
    return OpenCVLineDetector(
        LineDetectorConfig(
            roi_top_ratio=0.35,
            segmentation_mode="hsv_white_on_green",
            line_polarity="light",
            minimum_confidence=0.30,
        )
    )


class TestOpenCVLineDetector:
    @pytest.fixture(autouse=True)
    def _detector(self) -> None:
        self.detector = OpenCVLineDetector(
            LineDetectorConfig(
                roi_top_ratio=0.35,
                minimum_confidence=0.30,
            )
        )

    def test_centered_straight_dark_line(self) -> None:
        frame = _frame_with_line([(320, 150), (320, 479)])
        result = self.detector.detect(frame)

        assert not result.observation.line_lost
        assert result.observation.valid_bands == 3
        assert (result.observation.offset or 0.0) == pytest.approx(0.0, abs=0.04)
        assert (result.observation.heading or 0.0) == pytest.approx(0.0, abs=0.04)

    def test_right_displaced_line_has_positive_offset(self) -> None:
        frame = _frame_with_line([(430, 150), (430, 479)])
        observation = self.detector.detect(frame).observation

        assert not observation.line_lost
        assert observation.offset is not None
        assert observation.offset > 0.25

    def test_line_bending_right_has_positive_heading(self) -> None:
        frame = _frame_with_line([(440, 150), (365, 310), (315, 479)])
        observation = self.detector.detect(frame).observation

        assert not observation.line_lost
        assert observation.heading is not None
        assert observation.heading > 0.15

    def test_blank_floor_is_rejected_as_line_lost(self) -> None:
        frame = np.full((HEIGHT, WIDTH, 3), 235, dtype=np.uint8)
        observation = self.detector.detect(frame).observation

        assert observation.line_lost
        assert observation.valid_bands == 0
        assert observation.offset is None
        assert observation.rejection_reason == "insufficient_valid_bands"

    def test_light_line_mode(self) -> None:
        frame = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
        cv2.line(frame, (320, 150), (320, 479), (240, 240, 240), 34)
        detector = OpenCVLineDetector(
            LineDetectorConfig(
                roi_top_ratio=0.35,
                line_polarity="light",
                minimum_confidence=0.30,
            )
        )
        observation = detector.detect(frame).observation

        assert not observation.line_lost
        assert (observation.offset or 0.0) == pytest.approx(0.0, abs=0.04)

    def test_invalid_empty_frame_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            self.detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    def test_white_line_on_green_cloth_is_detected(self) -> None:
        frame = _green_cloth_frame()
        cv2.line(frame, (420, 150), (340, 479), (245, 245, 245), 34, cv2.LINE_AA)
        observation = _white_on_green_detector().detect(frame).observation

        assert not observation.line_lost
        assert observation.valid_bands == 3
        assert observation.heading is not None
        assert observation.heading > 0.10

    def test_green_cloth_without_white_line_is_line_lost(self) -> None:
        observation = _white_on_green_detector().detect(
            _green_cloth_frame()
        ).observation

        assert observation.line_lost
        assert observation.valid_bands == 0

    def test_white_object_outside_green_floor_is_rejected(self) -> None:
        frame = np.full((HEIGHT, WIDTH, 3), (175, 200, 220), dtype=np.uint8)
        cv2.rectangle(frame, (0, 120), (430, 479), (50, 145, 45), cv2.FILLED)
        cv2.circle(frame, (560, 360), 55, (250, 250, 250), cv2.FILLED)
        observation = _white_on_green_detector().detect(frame).observation

        assert observation.line_lost
        assert observation.valid_bands == 0
