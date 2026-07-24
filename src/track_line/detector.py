"""Classical OpenCV detector for a high-contrast floor guide line."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from .config import LineDetectorConfig
    from .observations import LineDetectionResult, LineObservation
except ImportError:  # Direct script/test execution from this directory.
    from config import LineDetectorConfig
    from observations import LineDetectionResult, LineObservation


Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _BandCandidate:
    band_index: int
    center_x: float
    center_y: float
    area_ratio: float
    width_ratio: float


class OpenCVLineDetector:
    """Detect a dark or light guide line in three horizontal ROI bands."""

    def __init__(self, config: LineDetectorConfig | None = None) -> None:
        self.config = config or LineDetectorConfig()

    def _white_on_green_mask(self, roi: Image) -> Image:
        """Select low-saturation white pixels inside the observed green cloth."""

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv,
            np.asarray([0, 0, self.config.white_value_min], dtype=np.uint8),
            np.asarray(
                [179, self.config.white_saturation_max, 255],
                dtype=np.uint8,
            ),
        )
        green = cv2.inRange(
            hsv,
            np.asarray(
                [
                    self.config.green_hue_min,
                    self.config.green_saturation_min,
                    self.config.green_value_min,
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [self.config.green_hue_max, 255, 255],
                dtype=np.uint8,
            ),
        )

        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        green_ratio = cv2.countNonZero(green) / roi_area
        if green_ratio < self.config.minimum_green_roi_ratio:
            return np.zeros(green.shape, dtype=np.uint8)

        contours, _ = cv2.findContours(
            green,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        floor_components = [
            contour
            for contour in contours
            if cv2.contourArea(contour) / roi_area
            >= self.config.minimum_green_component_ratio
        ]
        if not floor_components:
            return np.zeros(green.shape, dtype=np.uint8)

        # Combining the substantial green components before taking the hull lets
        # a white tape stripe split the cloth without splitting the floor mask.
        floor_points = np.concatenate(floor_components, axis=0)
        floor_hull = cv2.convexHull(floor_points)
        floor_mask = np.zeros(green.shape, dtype=np.uint8)
        cv2.drawContours(floor_mask, [floor_hull], -1, 255, cv2.FILLED)
        return cv2.bitwise_and(white, floor_mask)

    def _make_mask(self, roi: Image) -> Image:
        blurred = cv2.GaussianBlur(
            roi,
            (self.config.blur_kernel, self.config.blur_kernel),
            0,
        )
        if self.config.segmentation_mode == "hsv_white_on_green":
            mask = self._white_on_green_mask(blurred)
        else:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            threshold_type = (
                cv2.THRESH_BINARY_INV
                if self.config.line_polarity == "dark"
                else cv2.THRESH_BINARY
            )
            _, mask = cv2.threshold(
                gray,
                0,
                255,
                threshold_type | cv2.THRESH_OTSU,
            )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _band_candidates(
        self,
        band: Image,
        *,
        band_index: int,
        y_offset: int,
    ) -> list[_BandCandidate]:
        contours, _ = cv2.findContours(
            band,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        band_height, band_width = band.shape
        band_area = float(max(1, band_height * band_width))
        candidates: list[_BandCandidate] = []

        for contour in contours:
            area_ratio = cv2.contourArea(contour) / band_area
            x, y, width, height = cv2.boundingRect(contour)
            width_ratio = width / max(1, band_width)
            if not (
                self.config.minimum_component_area_ratio
                <= area_ratio
                <= self.config.maximum_component_area_ratio
            ):
                continue
            if not (
                self.config.minimum_component_width_ratio
                <= width_ratio
                <= self.config.maximum_component_width_ratio
            ):
                continue

            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center_x = moments["m10"] / moments["m00"]
            center_y = y_offset + moments["m01"] / moments["m00"]
            candidates.append(
                _BandCandidate(
                    band_index=band_index,
                    center_x=center_x,
                    center_y=center_y,
                    area_ratio=area_ratio,
                    width_ratio=width_ratio,
                )
            )
        return candidates

    def _choose_candidate(
        self,
        candidates: list[_BandCandidate],
        *,
        expected_x: float | None,
        image_width: int,
    ) -> _BandCandidate | None:
        if not candidates:
            return None

        maximum_area = max(candidate.area_ratio for candidate in candidates)

        def score(candidate: _BandCandidate) -> float:
            area_score = candidate.area_ratio / max(maximum_area, 1e-9)
            if expected_x is None:
                continuity_score = 1.0
            else:
                normalized_distance = abs(candidate.center_x - expected_x) / max(1, image_width)
                continuity_score = math.exp(
                    -normalized_distance / self.config.continuity_scale
                )
            return 0.65 * area_score + 0.35 * continuity_score

        return max(candidates, key=score)

    def detect(
        self,
        frame: Image,
        *,
        frame_index: int = 0,
        timestamp_ns: int | None = None,
    ) -> LineDetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape HxWx3")

        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        frame_height, frame_width = frame.shape[:2]
        roi_top = int(round(frame_height * self.config.roi_top_ratio))
        roi = frame[roi_top:, :]
        if roi.shape[0] < 12 or roi.shape[1] < 12:
            raise ValueError("configured ROI is too small")

        mask = self._make_mask(roi)
        roi_height = mask.shape[0]
        boundaries = np.linspace(0, roi_height, 4, dtype=np.int32)

        selected_by_band: dict[int, _BandCandidate] = {}
        expected_x: float | None = None
        # Start with the near band and follow the same component toward the horizon.
        for band_index in (2, 1, 0):
            y0 = int(boundaries[band_index])
            y1 = int(boundaries[band_index + 1])
            band = mask[y0:y1, :]
            candidates = self._band_candidates(
                band,
                band_index=band_index,
                y_offset=y0,
            )
            selected = self._choose_candidate(
                candidates,
                expected_x=expected_x,
                image_width=frame_width,
            )
            if selected is not None:
                selected_by_band[band_index] = selected
                expected_x = selected.center_x

        selected = [selected_by_band[index] for index in sorted(selected_by_band)]
        valid_bands = len(selected)
        coverage = valid_bands / 3.0

        if selected:
            area_quality = float(
                np.mean(
                    [
                        min(1.0, item.area_ratio / self.config.target_component_area_ratio)
                        for item in selected
                    ]
                )
            )
            width_quality = math.exp(
                -float(np.std([item.width_ratio for item in selected])) / 0.12
            )
        else:
            area_quality = 0.0
            width_quality = 0.0

        if valid_bands >= 2:
            consecutive = [
                abs(first.center_x - second.center_x) / max(1, frame_width)
                for first, second in zip(selected, selected[1:])
            ]
            continuity = math.exp(
                -float(np.mean(consecutive)) / self.config.continuity_scale
            )
        else:
            continuity = 0.0

        confidence = coverage * (
            0.35
            + 0.25 * area_quality
            + 0.25 * continuity
            + 0.15 * width_quality
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        insufficient_bands = valid_bands < self.config.minimum_valid_bands
        low_confidence = confidence < self.config.minimum_confidence
        line_lost = insufficient_bands or low_confidence

        rejection_reason: str | None = None
        if insufficient_bands:
            rejection_reason = "insufficient_valid_bands"
        elif low_confidence:
            rejection_reason = "low_confidence"

        half_width = max(1.0, frame_width / 2.0)
        if line_lost:
            offset = None
            heading = None
            curvature = None
        else:
            available_weights = [
                self.config.band_weights[item.band_index] for item in selected
            ]
            weighted_center = float(
                np.average(
                    [item.center_x for item in selected],
                    weights=available_weights,
                )
            )
            offset = float(np.clip((weighted_center - half_width) / half_width, -1, 1))
            far = selected[0]
            near = selected[-1]
            heading = float(
                np.clip((far.center_x - near.center_x) / half_width, -1, 1)
            )
            if all(index in selected_by_band for index in (0, 1, 2)):
                far_x = selected_by_band[0].center_x
                middle_x = selected_by_band[1].center_x
                near_x = selected_by_band[2].center_x
                curvature = float(
                    np.clip((far_x - 2.0 * middle_x + near_x) / half_width, -1, 1)
                )
            else:
                curvature = 0.0

        points_px = tuple(
            (int(round(item.center_x)), int(round(item.center_y + roi_top)))
            for item in selected
        )
        points_normalized = tuple(
            (
                float(np.clip(x / max(1, frame_width - 1), 0, 1)),
                float(np.clip(y / max(1, frame_height - 1), 0, 1)),
            )
            for x, y in points_px
        )
        observation = LineObservation(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            offset=offset,
            heading=heading,
            curvature=curvature,
            confidence=confidence,
            line_lost=line_lost,
            valid_bands=valid_bands,
            points_normalized=points_normalized,
            rejection_reason=rejection_reason,
        )
        return LineDetectionResult(
            observation=observation,
            mask=mask,
            roi_top=roi_top,
            points_px=points_px,
            band_boundaries_px=tuple(int(value + roi_top) for value in boundaries),
        )
