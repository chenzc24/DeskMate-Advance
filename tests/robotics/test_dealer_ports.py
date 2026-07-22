from __future__ import annotations

import pytest

from poker_dealer.domain import DealerCommand, DealerCommandType
from poker_dealer.robotics.dealer import (
    DealerUnavailableError,
    SimulatedDealerAdapter,
    UnavailableDealerAdapter,
)


def test_simulated_adapter_is_explicit_and_requires_open() -> None:
    adapter = SimulatedDealerAdapter("sim")
    command = DealerCommand("home-1", 1, DealerCommandType.HOME)
    with pytest.raises(DealerUnavailableError, match="not open"):
        adapter.execute(command)
    adapter.open()
    ack = adapter.execute(command, observed_at_ns=2)
    assert ack.command_id == command.command_id
    assert adapter.health().physical_motion is False
    adapter.close()


def test_real_adapter_placeholder_fails_closed() -> None:
    adapter = UnavailableDealerAdapter("robot", "safety release incomplete")
    assert adapter.health().available is False
    assert adapter.health().physical_motion is True
    with pytest.raises(DealerUnavailableError, match="safety release incomplete"):
        adapter.open()
