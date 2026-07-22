"""Transport-neutral boundary for semantic dealer commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poker_dealer.domain import DealerAck, DealerCommand


class DealerUnavailableError(RuntimeError):
    """Raised when a requested dealer implementation is not safely available."""


@dataclass(frozen=True, slots=True)
class DealerHealth:
    device_id: str
    available: bool
    opened: bool
    physical_motion: bool
    reason: str | None = None


class DealerPort(Protocol):
    """The runtime-facing dealer API; implementations own transport details."""

    @property
    def device_id(self) -> str: ...

    @property
    def physical_motion(self) -> bool: ...

    def open(self) -> None: ...

    def execute(
        self, command: DealerCommand, observed_at_ns: int | None = None
    ) -> DealerAck: ...

    def health(self) -> DealerHealth: ...

    def close(self) -> None: ...


__all__ = ["DealerHealth", "DealerPort", "DealerUnavailableError"]
