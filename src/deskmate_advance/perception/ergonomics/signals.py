"""Deterministic Part A luminance and audio-level calculations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from deskmate_advance.domain.frame import ColorSpace, FramePacket

from .observations import (
    AudioLevelObservation,
    LuminanceObservation,
    ObservationState,
)


class LuminanceCalculator:
    """Compute full-frame Rec. 709 luminance statistics without thresholds."""

    def observe(self, frame: FramePacket) -> LuminanceObservation:
        pixels = frame.image.astype(np.float32, copy=False)
        if frame.color_space is ColorSpace.BGR:
            blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        elif frame.color_space is ColorSpace.RGB:
            red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        else:
            return LuminanceObservation(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=frame.captured_at_ns,
                state=ObservationState.ERROR,
                mean=None,
                median=None,
                p10=None,
                p90=None,
                reason=f"unsupported_color_space:{frame.color_space}",
            )
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return LuminanceObservation(
            source_id=frame.source_id,
            sequence_id=frame.sequence_id,
            captured_at_ns=frame.captured_at_ns,
            state=ObservationState.VALID,
            mean=float(np.mean(luminance)),
            median=float(np.median(luminance)),
            p10=float(np.percentile(luminance, 10)),
            p90=float(np.percentile(luminance, 90)),
        )


class AudioLevelCalculator:
    """Compute channel-agnostic RMS and dBFS from one owned audio window."""

    def __init__(self, *, silence_floor_dbfs: float = -120.0) -> None:
        if not math.isfinite(silence_floor_dbfs) or silence_floor_dbfs >= 0:
            raise ValueError("silence_floor_dbfs must be a finite negative value")
        self.silence_floor_dbfs = silence_floor_dbfs

    def observe(
        self,
        samples: NDArray[np.generic],
        *,
        source_id: str,
        window_started_at_ns: int,
        window_ended_at_ns: int,
        sample_rate_hz: int,
    ) -> AudioLevelObservation:
        if not source_id.strip():
            raise ValueError("source_id must not be empty")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if window_started_at_ns < 0 or window_ended_at_ns <= window_started_at_ns:
            raise ValueError("audio window timestamps must be positive and increasing")
        array = np.asarray(samples)
        sample_count = int(array.shape[0]) if array.ndim in {1, 2} else 0
        if sample_count == 0 or array.ndim not in {1, 2}:
            return AudioLevelObservation(
                source_id=source_id,
                window_started_at_ns=window_started_at_ns,
                window_ended_at_ns=window_ended_at_ns,
                sample_rate_hz=sample_rate_hz,
                sample_count=sample_count,
                state=ObservationState.MISSING,
                rms=None,
                dbfs=None,
                reason="empty_or_invalid_audio_window",
            )
        normalized = self._normalize(array)
        if normalized is None or not np.all(np.isfinite(normalized)):
            return AudioLevelObservation(
                source_id=source_id,
                window_started_at_ns=window_started_at_ns,
                window_ended_at_ns=window_ended_at_ns,
                sample_rate_hz=sample_rate_hz,
                sample_count=sample_count,
                state=ObservationState.ERROR,
                rms=None,
                dbfs=None,
                reason="unsupported_or_non_finite_audio",
            )
        rms = float(np.sqrt(np.mean(np.square(normalized, dtype=np.float64))))
        dbfs = self.silence_floor_dbfs if rms == 0.0 else max(
            self.silence_floor_dbfs,
            20.0 * math.log10(rms),
        )
        return AudioLevelObservation(
            source_id=source_id,
            window_started_at_ns=window_started_at_ns,
            window_ended_at_ns=window_ended_at_ns,
            sample_rate_hz=sample_rate_hz,
            sample_count=sample_count,
            state=ObservationState.VALID,
            rms=rms,
            dbfs=dbfs,
        )

    @staticmethod
    def _normalize(samples: NDArray[np.generic]) -> NDArray[np.float64] | None:
        if np.issubdtype(samples.dtype, np.floating):
            return samples.astype(np.float64, copy=False)
        if np.issubdtype(samples.dtype, np.signedinteger):
            scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
            return samples.astype(np.float64) / scale
        if np.issubdtype(samples.dtype, np.unsignedinteger):
            info = np.iinfo(samples.dtype)
            midpoint = (float(info.max) + 1.0) / 2.0
            return (samples.astype(np.float64) - midpoint) / midpoint
        return None
