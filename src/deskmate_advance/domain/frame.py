"""Project-owned video frame records.

The runtime uses the monotonic timestamp for all duration calculations. The
negotiated FPS is metadata only and must not be used as a clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ColorSpace(StrEnum):
    """Supported image encodings at the perception boundary."""

    BGR = "bgr"
    RGB = "rgb"


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One immutable, timestamped frame owned by the project runtime."""

    sequence_id: int
    captured_at_ns: int
    source_id: str
    device_index: int
    width: int
    height: int
    color_space: ColorSpace
    nominal_fps: float
    dropped_before: int
    image: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.captured_at_ns < 0:
            raise ValueError("captured_at_ns must be non-negative")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.dropped_before < 0:
            raise ValueError("dropped_before must be non-negative")
        if self.image.dtype != np.uint8:
            raise ValueError("image must use uint8 pixels")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if self.image.shape[:2] != (self.height, self.width):
            raise ValueError("declared dimensions do not match image")
