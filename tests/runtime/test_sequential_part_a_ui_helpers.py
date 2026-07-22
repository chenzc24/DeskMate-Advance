from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("seat", list(Seat))
def test_single_player_mode_aligns_engine_to_only_registered_seat(seat: Seat) -> None:
    button, first, error = MODULE._resolve_start_plan(
        "single_player_pilot", {seat}, Seat.A
    )
    assert error is None
    assert first is seat
    assert HandEngine.start("single", button).state.acting_seat is seat


def test_single_player_mode_rejects_zero_or_multiple_registrations() -> None:
    for enrolled in (set(), {Seat.A, Seat.B}):
        _button, _first, error = MODULE._resolve_start_plan(
            "single_player_pilot", enrolled, Seat.A
        )
        assert error == "single_player_pilot requires exactly one registered seat"


def test_single_player_pilot_stops_after_its_one_registered_action() -> None:
    remaining = {Seat.C}
    reason = MODULE._consume_pilot_action(
        "single_player_pilot", remaining, Seat.C
    )
    assert reason == "single_registered_player_completed_one_action"
    assert not remaining


def test_two_player_pilot_waits_for_both_registered_actions() -> None:
    remaining = {Seat.A, Seat.B}
    assert MODULE._consume_pilot_action(
        "two_player_pilot", remaining, Seat.A
    ) is None
    assert remaining == {Seat.B}
    assert MODULE._consume_pilot_action(
        "two_player_pilot", remaining, Seat.B
    ) == "two_registered_players_completed_one_action"


def test_event_record_has_acceptance_context_and_monotonic_timestamp() -> None:
    MODULE._configure_event_output(
        session_id="session-1",
        hand_id="hand-1",
        acceptance_case="FPA-01",
        log_path=None,
        acceptance_session_group="group-01",
    )

    event = MODULE._event_record("probe", value=3)

    assert event["type"] == "probe"
    assert event["session_id"] == "session-1"
    assert event["hand_id"] == "hand-1"
    assert event["acceptance_case"] == "FPA-01"
    assert event["acceptance_session_group"] == "group-01"
    assert isinstance(event["logged_at_monotonic_ns"], int)
    assert event["value"] == 3


def test_robot_stream_camera_config_preserves_existing_frame_contract() -> None:
    args = SimpleNamespace(
        stream_url="http://100.80.46.54:5000/video_feed",
        index=None,
        backend=None,
        stream_open_timeout_ms=4000,
        stream_read_timeout_ms=1500,
    )
    identity_config = SimpleNamespace(camera={})

    config = MODULE._camera_config(args, identity_config)

    assert config.stream_url == "http://100.80.46.54:5000/video_feed"
    assert config.source_id == "robot_mjpeg_stream"
    assert config.backend == "auto"
    assert config.width is None
    assert config.height is None
    assert config.fps is None
    assert config.open_timeout_ms == 4000
    assert config.read_timeout_ms == 1500


def test_robot_stream_rejects_a_competing_local_camera_index() -> None:
    args = SimpleNamespace(
        stream_url="http://100.80.46.54:5000/video_feed",
        index=0,
        backend=None,
        stream_open_timeout_ms=5000,
        stream_read_timeout_ms=2000,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE._camera_config(args, SimpleNamespace(camera={}))
