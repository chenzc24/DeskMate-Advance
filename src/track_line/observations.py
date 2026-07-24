"""Data contracts emitted by the line detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class LineObservation:
    """Non-authoritative evidence describing one detected guide line."""

    frame_index: int
    timestamp_ns: int
    offset: float | None
    heading: float | None
    curvature: float | None
    confidence: float
    line_lost: bool
    valid_bands: int
    points_normalized: tuple[tuple[float, float], ...]
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "offset": self.offset,
            "heading": self.heading,
            "curvature": self.curvature,
            "confidence": self.confidence,
            "line_lost": self.line_lost,
            "valid_bands": self.valid_bands,
            "points_normalized": [list(point) for point in self.points_normalized],
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class LineDetectionResult:
    """Observation plus debug-only image products."""

    observation: LineObservation
    mask: Image
    roi_top: int
    points_px: tuple[tuple[int, int], ...]
    band_boundaries_px: tuple[int, ...]
