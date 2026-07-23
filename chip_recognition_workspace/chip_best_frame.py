"""Bounded per-track selection of useful raw chip frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import cv2
import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]
Detection = MutableMapping[str, object]


@dataclass(frozen=True, slots=True)
class BestFrameCandidate:
    """One bounded crop selected for denomination recognition."""

    track_id: int
    source_frame: int
    image: Image
    local_bbox_xyxy: tuple[int, int, int, int]
    source_bbox_xyxy: tuple[int, int, int, int]
    quality_score: float
    sharpness_score: float
    glare_ratio: float


@dataclass(slots=True)
class _TrackWindow:
    samples: int = 0
    best: BestFrameCandidate | None = None
    last_seen_frame: int = 0


def _bounded_crop(
    image: Image,
    bbox_xyxy: tuple[int, int, int, int],
    padding: float = 0.18,
) -> tuple[Image, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    crop_x1 = max(0, int(np.floor(x1 - box_width * padding)))
    crop_y1 = max(0, int(np.floor(y1 - box_height * padding)))
    crop_x2 = min(width, int(np.ceil(x2 + box_width * padding)))
    crop_y2 = min(height, int(np.ceil(y2 + box_height * padding)))
    crop = np.ascontiguousarray(image[crop_y1:crop_y2, crop_x1:crop_x2])
    local_bbox = (
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1,
    )
    return crop, local_bbox


def frame_quality(
    image: Image,
    bbox_xyxy: tuple[int, int, int, int],
    detector_confidence: float,
) -> tuple[float, float, float]:
    """Score an inexpensive raw view before running ellipse rectification."""

    x1, y1, x2, y2 = bbox_xyxy
    height, width = image.shape[:2]
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0, 0.0, 1.0
    chip = np.ascontiguousarray(image[y1:y2, x1:x2])
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_score = min(1.0, laplacian_variance / 320.0)

    hsv = cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)
    glare = (hsv[:, :, 2] >= 248) & (hsv[:, :, 1] <= 35)
    glare_ratio = float(np.count_nonzero(glare)) / glare.size
    glare_score = max(0.0, 1.0 - glare_ratio / 0.22)

    box_width = x2 - x1
    box_height = y2 - y1
    minor = min(box_width, box_height)
    size_score = min(1.0, minor / 120.0)
    box_aspect = minor / max(box_width, box_height)
    aspect_score = min(1.0, box_aspect / 0.72)
    confidence_score = float(np.clip(detector_confidence, 0.0, 1.0))
    score = (
        0.30 * size_score
        + 0.30 * sharpness_score
        + 0.17 * glare_score
        + 0.13 * aspect_score
        + 0.10 * confidence_score
    )
    return round(score, 6), round(sharpness_score, 6), round(glare_ratio, 6)


class ChipBestFrameSelector:
    """Collect a small window and emit only its best frame for each track."""

    def __init__(self, *, window_samples: int = 5, max_idle_frames: int = 20) -> None:
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if max_idle_frames <= 0:
            raise ValueError("max_idle_frames must be positive")
        self.window_samples = window_samples
        self.max_idle_frames = max_idle_frames
        self._windows: dict[int, _TrackWindow] = {}

    def observe(
        self,
        frame: int,
        image: Image,
        detections: list[Detection],
    ) -> None:
        for detection in detections:
            track_id = detection.get("track_id")
            if not isinstance(track_id, int):
                continue
            bbox = tuple(int(value) for value in detection["bbox_xyxy"])
            confidence = float(detection.get("confidence", 0.0))
            score, sharpness, glare = frame_quality(image, bbox, confidence)
            window = self._windows.setdefault(track_id, _TrackWindow())
            window.samples += 1
            window.last_seen_frame = frame
            if window.best is not None and score <= window.best.quality_score:
                continue
            crop, local_bbox = _bounded_crop(image, bbox)
            window.best = BestFrameCandidate(
                track_id=track_id,
                source_frame=frame,
                image=crop,
                local_bbox_xyxy=local_bbox,
                source_bbox_xyxy=bbox,
                quality_score=score,
                sharpness_score=sharpness,
                glare_ratio=glare,
            )

        expired = [
            track_id
            for track_id, window in self._windows.items()
            if frame - window.last_seen_frame > self.max_idle_frames
        ]
        for track_id in expired:
            del self._windows[track_id]

    def take_ready(self) -> tuple[BestFrameCandidate, ...]:
        ready: list[BestFrameCandidate] = []
        for window in self._windows.values():
            if window.samples < self.window_samples or window.best is None:
                continue
            ready.append(window.best)
            window.samples = 0
            window.best = None
        return tuple(sorted(ready, key=lambda candidate: candidate.track_id))
