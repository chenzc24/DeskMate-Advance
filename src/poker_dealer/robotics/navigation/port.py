"""Runtime boundary for mobile-robot navigation and player alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poker_dealer.domain import NavigationAck, NavigationCommand, RobotPoseNode


class NavigationUnavailableError(RuntimeError):
    """Raised when the selected navigation transport is unavailable."""


@dataclass(frozen=True, slots=True)
class NavigationHealth:
    device_id: str
    available: bool
    opened: bool
    physical_motion: bool
    pose: RobotPoseNode
    pose_version: int
    reason: str | None = None


class NavigationPort(Protocol):
    @property
    def device_id(self) -> str: ...

    @property
    def physical_motion(self) -> bool: ...

    def open(self) -> None: ...

    def execute(
        self, command: NavigationCommand, observed_at_ns: int | None = None
    ) -> NavigationAck: ...

    def health(self) -> NavigationHealth: ...

    def close(self) -> None: ...


__all__ = [
    "NavigationHealth",
    "NavigationPort",
    "NavigationUnavailableError",
]
