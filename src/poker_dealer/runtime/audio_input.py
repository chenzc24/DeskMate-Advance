"""Native-rate PCM capture helpers for speech-model input."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Callable

import numpy as np


class StreamingPcm16Resampler:
    """Continuously resample mono int16 PCM without resetting at block edges."""

    def __init__(self, source_rate_hz: int, target_rate_hz: int) -> None:
        if source_rate_hz <= 0 or target_rate_hz <= 0:
            raise ValueError("audio sample rates must be positive")
        self.source_rate_hz = source_rate_hz
        self.target_rate_hz = target_rate_hz
        self._step = source_rate_hz / target_rate_hz
        self._buffer = np.empty(0, dtype=np.float64)
        self._next_position = 0.0

    def process(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("int16 PCM byte count must be even")
        incoming = np.frombuffer(pcm, dtype="<i2")
        if not len(incoming):
            return b""
        if self.source_rate_hz == self.target_rate_hz:
            return bytes(pcm)
        self._buffer = np.concatenate(
            (self._buffer, incoming.astype(np.float64, copy=False))
        )
        if len(self._buffer) < 2 or self._next_position >= len(self._buffer) - 1:
            return b""
        count = int(
            np.floor(
                (len(self._buffer) - 1 - self._next_position) / self._step
            )
        ) + 1
        positions = self._next_position + np.arange(count) * self._step
        output = np.interp(
            positions,
            np.arange(len(self._buffer), dtype=np.float64),
            self._buffer,
        )
        self._next_position = float(positions[-1] + self._step)
        drop = min(int(self._next_position), len(self._buffer))
        if drop:
            self._buffer = self._buffer[drop:]
            self._next_position -= drop
        return np.clip(np.rint(output), -32768, 32767).astype("<i2").tobytes()

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float64)
        self._next_position = 0.0


@dataclass(frozen=True, slots=True)
class AudioInputHealthSnapshot:
    opened_at_ns: int
    last_callback_at_ns: int | None
    callback_blocks: int
    callback_frames: int
    status_events: int
    last_status: str | None
    rms_level: float
    peak_level: float

    def is_stale(self, observed_at_ns: int, stale_after_ms: int) -> bool:
        baseline = self.last_callback_at_ns or self.opened_at_ns
        return observed_at_ns - baseline > stale_after_ms * 1_000_000


class AudioInputHealth:
    """Thread-safe callback liveness and PortAudio-status counters."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._clock_ns = clock_ns
        self._lock = Lock()
        self._opened_at_ns = clock_ns()
        self._last_callback_at_ns: int | None = None
        self._callback_blocks = 0
        self._callback_frames = 0
        self._status_events = 0
        self._last_status: str | None = None
        self._rms_level = 0.0
        self._peak_level = 0.0

    def reset_opened(self) -> None:
        with self._lock:
            self._opened_at_ns = self._clock_ns()
            self._last_callback_at_ns = None
            self._rms_level = 0.0
            self._peak_level = 0.0

    def record_callback(
        self,
        frames: int,
        status: object,
        *,
        rms_level: float = 0.0,
        peak_level: float = 0.0,
    ) -> None:
        if frames < 0:
            raise ValueError("audio callback frame count must be non-negative")
        if (
            not 0.0 <= rms_level <= 1.0
            or not 0.0 <= peak_level <= 1.0
            or rms_level > peak_level
        ):
            raise ValueError("normalized audio levels are invalid")
        status_text = str(status).strip()
        with self._lock:
            self._last_callback_at_ns = self._clock_ns()
            self._callback_blocks += 1
            self._callback_frames += frames
            self._rms_level = rms_level
            self._peak_level = peak_level
            if status_text:
                self._status_events += 1
                self._last_status = status_text

    def snapshot(self) -> AudioInputHealthSnapshot:
        with self._lock:
            return AudioInputHealthSnapshot(
                opened_at_ns=self._opened_at_ns,
                last_callback_at_ns=self._last_callback_at_ns,
                callback_blocks=self._callback_blocks,
                callback_frames=self._callback_frames,
                status_events=self._status_events,
                last_status=self._last_status,
                rms_level=self._rms_level,
                peak_level=self._peak_level,
            )


__all__ = [
    "AudioInputHealth",
    "AudioInputHealthSnapshot",
    "StreamingPcm16Resampler",
]
