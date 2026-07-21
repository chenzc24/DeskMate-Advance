from __future__ import annotations

import importlib.util
from pathlib import Path

from poker_dealer.domain import Seat
from poker_dealer.game import HandEngine


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/perception/live_sequential_part_a.py"
SPEC = importlib.util.spec_from_file_location("live_sequential_part_a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_two_consecutive_enrolled_seats_start_at_first_registered_run() -> None:
    button, first = MODULE._two_player_pilot_start({Seat.A, Seat.B})
    engine = HandEngine.start("partial", button)
    assert button is Seat.B
    assert first is Seat.A
    assert engine.state.acting_seat is Seat.A


def test_wrapped_enrolled_run_is_preserved() -> None:
    button, first = MODULE._two_player_pilot_start({Seat.D, Seat.A})
    engine = HandEngine.start("partial", button)
    assert button is Seat.A
    assert first is Seat.D
    assert engine.state.acting_seat is Seat.D


def test_four_player_mode_blocks_incomplete_roster_and_accepts_all_four() -> None:
    _button, _first, error = MODULE._resolve_start_plan(
        "four_player_core", {Seat.A, Seat.B, Seat.C}, Seat.A
    )
    assert error is not None
    assert "seat_d" in error
    button, first, error = MODULE._resolve_start_plan(
        "four_player_core", set(Seat), Seat.A
    )
    assert error is None
    assert button is Seat.A
    assert first is None


def test_two_player_mode_is_explicit_and_requires_adjacent_seats() -> None:
    _button, _first, error = MODULE._resolve_start_plan(
        "two_player_pilot", {Seat.A, Seat.C}, Seat.A
    )
    assert error == "two_player_pilot requires adjacent clockwise seats"
    button, first, error = MODULE._resolve_start_plan(
        "two_player_pilot", {Seat.A, Seat.B}, Seat.A
    )
    assert error is None
    assert button is Seat.B
    assert first is Seat.A
