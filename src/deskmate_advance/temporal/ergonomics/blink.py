"""Bounded, timestamp-derived blink-rate evidence for Part A."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
import math


class EyeState(StrEnum):
    """Observed eye state without treating ambiguous evidence as open."""

    OPEN = "open"
    CLOSED = "closed"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BlinkTrackerConfig:
    closed_score: float
    open_score: float
    window_ms: int
    minimum_valid_ms: int
    minimum_blinks_per_minute: float
    minimum_closed_ms: int
    maximum_closed_ms: int
    maximum_valid_gap_ms: int = 500
    maximum_segments: int = 256
    maximum_blinks: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("closed_score", self.closed_score),
            ("open_score", self.open_score),
            ("minimum_blinks_per_minute", self.minimum_blinks_per_minute),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        for name, value in (
            ("window_ms", self.window_ms),
            ("minimum_valid_ms", self.minimum_valid_ms),
            ("minimum_closed_ms", self.minimum_closed_ms),
            ("maximum_closed_ms", self.maximum_closed_ms),
            ("maximum_valid_gap_ms", self.maximum_valid_gap_ms),
            ("maximum_segments", self.maximum_segments),
            ("maximum_blinks", self.maximum_blinks),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        scores = (self.open_score, self.closed_score)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores):
            raise ValueError("eye scores must be finite and in [0, 1]")
        if self.open_score >= self.closed_score:
            raise ValueError("open_score must be lower than closed_score")
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if not 0 < self.minimum_valid_ms <= self.window_ms:
            raise ValueError("minimum_valid_ms must be in (0, window_ms]")
        if (
            not math.isfinite(self.minimum_blinks_per_minute)
            or self.minimum_blinks_per_minute < 0
        ):
            raise ValueError("minimum_blinks_per_minute must be finite and non-negative")
        if self.minimum_closed_ms < 0 or self.maximum_closed_ms < self.minimum_closed_ms:
            raise ValueError("closed-duration bounds are invalid")
        if self.maximum_valid_gap_ms <= 0:
            raise ValueError("maximum_valid_gap_ms must be positive")
        if self.maximum_segments <= 0 or self.maximum_blinks <= 0:
            raise ValueError("tracker capacities must be positive")


@dataclass(frozen=True, slots=True)
class BlinkRateSnapshot:
    observed_at_ns: int
    window_started_at_ns: int
    eye_state: EyeState
    blink_count: int
    valid_duration_ms: float
    blinks_per_minute: float | None
    low_rate: bool | None
    reason: str | None


class BlinkRateTracker:
    """Count complete open-closed-open blinks over valid observation time."""

    def __init__(self, config: BlinkTrackerConfig) -> None:
        self.config = config
        self._segments: deque[tuple[int, int]] = deque()
        self._blinks: deque[int] = deque()
        self._last_timestamp_ns: int | None = None
        self._last_valid_ns: int | None = None
        self._closed_started_ns: int | None = None
        self._open_seen = False
        self._eye_state = EyeState.UNKNOWN
        self._truncated_until_ns: int | None = None

    def reset(self) -> None:
        self._segments.clear()
        self._blinks.clear()
        self._last_timestamp_ns = None
        self._last_valid_ns = None
        self._closed_started_ns = None
        self._open_seen = False
        self._eye_state = EyeState.UNKNOWN
        self._truncated_until_ns = None

    def update(
        self,
        *,
        timestamp_ns: int,
        left_score: float | None,
        right_score: float | None,
        valid: bool,
    ) -> BlinkRateSnapshot:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("blink timestamps must increase strictly")
        self._last_timestamp_ns = timestamp_ns

        scores_valid = (
            valid
            and left_score is not None
            and right_score is not None
            and all(
                math.isfinite(value) and 0 <= value <= 1
                for value in (left_score, right_score)
            )
        )
        if not scores_valid:
            self._last_valid_ns = None
            self._closed_started_ns = None
            self._open_seen = False
            self._eye_state = EyeState.UNKNOWN
            return self._snapshot(timestamp_ns, "eye_evidence_unavailable")

        if self._last_valid_ns is not None:
            gap_ns = timestamp_ns - self._last_valid_ns
            if gap_ns <= self.config.maximum_valid_gap_ms * 1_000_000:
                self._append_segment(self._last_valid_ns, timestamp_ns)
            else:
                self._closed_started_ns = None
                self._open_seen = False
        self._last_valid_ns = timestamp_ns

        if left_score >= self.config.closed_score and right_score >= self.config.closed_score:
            self._eye_state = EyeState.CLOSED
            if self._open_seen and self._closed_started_ns is None:
                self._closed_started_ns = timestamp_ns
        elif left_score <= self.config.open_score and right_score <= self.config.open_score:
            self._eye_state = EyeState.OPEN
            if self._closed_started_ns is not None:
                closed_ms = (timestamp_ns - self._closed_started_ns) / 1_000_000
                if self.config.minimum_closed_ms <= closed_ms <= self.config.maximum_closed_ms:
                    self._append_blink(timestamp_ns)
                self._closed_started_ns = None
            self._open_seen = True
        else:
            self._eye_state = EyeState.AMBIGUOUS

        return self._snapshot(timestamp_ns, None)

    def snapshot(self, timestamp_ns: int | None = None) -> BlinkRateSnapshot:
        if timestamp_ns is None:
            if self._last_timestamp_ns is None:
                raise RuntimeError("blink tracker has not received an observation")
            timestamp_ns = self._last_timestamp_ns
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError("snapshot timestamp cannot move backward")
        return self._snapshot(timestamp_ns, None)

    def _append_segment(self, started_at_ns: int, ended_at_ns: int) -> None:
        if self._segments and self._segments[-1][1] == started_at_ns:
            first, _ = self._segments.pop()
            self._segments.append((first, ended_at_ns))
            return
        if len(self._segments) >= self.config.maximum_segments:
            _, dropped_end = self._segments.popleft()
            self._truncated_until_ns = max(self._truncated_until_ns or 0, dropped_end)
        self._segments.append((started_at_ns, ended_at_ns))

    def _append_blink(self, timestamp_ns: int) -> None:
        if len(self._blinks) >= self.config.maximum_blinks:
            dropped = self._blinks.popleft()
            self._truncated_until_ns = max(self._truncated_until_ns or 0, dropped)
        self._blinks.append(timestamp_ns)

    def _snapshot(self, timestamp_ns: int, missing_reason: str | None) -> BlinkRateSnapshot:
        window_ns = self.config.window_ms * 1_000_000
        window_start = max(0, timestamp_ns - window_ns)
        while self._segments and self._segments[0][1] <= window_start:
            self._segments.popleft()
        while self._blinks and self._blinks[0] < window_start:
            self._blinks.popleft()
        if self._truncated_until_ns is not None and window_start >= self._truncated_until_ns:
            self._truncated_until_ns = None

        valid_ns = sum(
            max(0, end - max(start, window_start))
            for start, end in self._segments
            if end > window_start and start <= timestamp_ns
        )
        valid_ms = valid_ns / 1_000_000
        blink_count = sum(window_start <= item <= timestamp_ns for item in self._blinks)
        rate = (
            blink_count * 60_000 / valid_ms
            if valid_ms > 0 and self._truncated_until_ns is None
            else None
        )
        if missing_reason is not None:
            low_rate = None
            reason = missing_reason
        elif self._truncated_until_ns is not None:
            low_rate = None
            reason = "window_capacity_exceeded"
        elif valid_ms < self.config.minimum_valid_ms:
            low_rate = None
            reason = "insufficient_valid_eye_time"
        else:
            low_rate = bool(rate is not None and rate < self.config.minimum_blinks_per_minute)
            reason = None
        return BlinkRateSnapshot(
            observed_at_ns=timestamp_ns,
            window_started_at_ns=window_start,
            eye_state=self._eye_state,
            blink_count=blink_count,
            valid_duration_ms=valid_ms,
            blinks_per_minute=rate,
            low_rate=low_rate,
            reason=reason,
        )
