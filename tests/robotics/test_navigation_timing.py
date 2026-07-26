from __future__ import annotations

import pytest

from poker_dealer.robotics.navigation import NavigationTimingGate


def test_navigation_timing_gate_opens_at_exact_monotonic_boundary() -> None:
    gate = NavigationTimingGate(2500)

    assert gate.can_start(0)
    gate.record_success(100_000_000)

    assert not gate.can_start(2_599_999_999)
    assert gate.remaining_ms(2_599_999_999) == 1
    assert gate.can_start(2_600_000_000)
    assert gate.remaining_ms(2_600_000_000) == 0


def test_navigation_timing_gate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NavigationTimingGate(-1)

    gate = NavigationTimingGate()
    with pytest.raises(ValueError, match="non-negative"):
        gate.can_start(-1)
    with pytest.raises(ValueError, match="non-negative"):
        gate.record_success(-1)
