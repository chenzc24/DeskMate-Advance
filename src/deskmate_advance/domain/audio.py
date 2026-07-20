"""Project-owned audio sample records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AudioPacket:
    """One immutable, timestamped block of normalized microphone samples."""

    sequence_id: int
    captured_at_ns: int
    source_id: str
    device_index: int
    sample_rate_hz: int
    channel_count: int
    sample_count: int
    input_overflowed: bool
    samples: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.captured_at_ns < 0:
            raise ValueError("captured_at_ns must be non-negative")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channel_count <= 0 or self.sample_count <= 0:
            raise ValueError("channel_count and sample_count must be positive")
        if self.samples.dtype != np.float32:
            raise ValueError("samples must use float32")
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape (sample_count, channels)")
        if self.samples.shape != (self.sample_count, self.channel_count):
            raise ValueError("declared audio dimensions do not match samples")
