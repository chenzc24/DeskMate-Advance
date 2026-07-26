"""Monotonic timing guard for consecutive physical navigation movements."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_INTER_MOTION_DELAY_MS = 2500


@dataclass(slots=True)
class NavigationTimingGate:
    """Non-blocking minimum gap measured from the previous successful ACK."""

    inter_motion_delay_ms: int = DEFAULT_INTER_MOTION_DELAY_MS
    next_motion_not_before_ns: int = 0

    def __post_init__(self) -> None:
        if self.inter_motion_delay_ms < 0:
            raise ValueError("inter-motion delay must be non-negative")
        if self.next_motion_not_before_ns < 0:
            raise ValueError("next motion timestamp must be non-negative")

    def can_start(self, now_ns: int) -> bool:
        if now_ns < 0:
            raise ValueError("motion clock must be non-negative")
        return now_ns >= self.next_motion_not_before_ns

    def remaining_ms(self, now_ns: int) -> int:
        if now_ns < 0:
            raise ValueError("motion clock must be non-negative")
        remaining_ns = max(0, self.next_motion_not_before_ns - now_ns)
        return (remaining_ns + 999_999) // 1_000_000

    def record_success(self, observed_at_ns: int) -> None:
        if observed_at_ns < 0:
            raise ValueError("navigation ACK timestamp must be non-negative")
        self.next_motion_not_before_ns = (
            observed_at_ns + self.inter_motion_delay_ms * 1_000_000
        )


__all__ = ["DEFAULT_INTER_MOTION_DELAY_MS", "NavigationTimingGate"]
