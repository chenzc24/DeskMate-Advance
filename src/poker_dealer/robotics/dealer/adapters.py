"""Safe dealer-port implementations used by runtime profiles."""

from __future__ import annotations

import time

from poker_dealer.domain import DealerAck, DealerCommand
from poker_dealer.game.simulators import SimulatedDealer

from .port import DealerHealth, DealerUnavailableError


class SimulatedDealerAdapter:
    """Explicit in-process simulator; it never controls physical hardware."""

    physical_motion = False

    def __init__(self, device_id: str) -> None:
        self._device_id = device_id
        self._opened = False
        self._dealer = SimulatedDealer()

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def simulator(self) -> SimulatedDealer:
        return self._dealer

    def open(self) -> None:
        # The simulator starts in a declared ready fixture state. Real homing is
        # never implied by this behavior and remains Robotics-owned motion.
        self._dealer.homed = True
        self._opened = True

    def execute(
        self, command: DealerCommand, observed_at_ns: int | None = None
    ) -> DealerAck:
        if not self._opened:
            raise DealerUnavailableError("simulated dealer is not open")
        return self._dealer.execute(
            command,
            observed_at_ns=(
                time.monotonic_ns() if observed_at_ns is None else observed_at_ns
            ),
        )

    def health(self) -> DealerHealth:
        return DealerHealth(
            device_id=self.device_id,
            available=True,
            opened=self._opened,
            physical_motion=False,
        )

    def close(self) -> None:
        self._opened = False


class UnavailableDealerAdapter:
    """Fail-closed placeholder for a real transport that is not integrated."""

    physical_motion = True

    def __init__(self, device_id: str, reason: str) -> None:
        self._device_id = device_id
        self._reason = reason

    @property
    def device_id(self) -> str:
        return self._device_id

    def open(self) -> None:
        raise DealerUnavailableError(self._reason)

    def execute(
        self, command: DealerCommand, observed_at_ns: int | None = None
    ) -> DealerAck:
        del command, observed_at_ns
        raise DealerUnavailableError(self._reason)

    def health(self) -> DealerHealth:
        return DealerHealth(
            device_id=self.device_id,
            available=False,
            opened=False,
            physical_motion=True,
            reason=self._reason,
        )

    def close(self) -> None:
        return None


__all__ = ["SimulatedDealerAdapter", "UnavailableDealerAdapter"]
