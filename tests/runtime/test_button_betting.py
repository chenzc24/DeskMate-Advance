from __future__ import annotations

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    ControlSource,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import HandEngine
from poker_dealer.runtime import ButtonBettingRuntime


def control(sequence: int, intent: ControlIntent, *, timestamp: int | None = None) -> ControlObservation:
    return ControlObservation(
        f"button:{sequence}",
        intent,
        ControlSource.SIMULATOR,
        sequence if timestamp is None else timestamp,
        "main",
        sequence,
    )


def test_button_selection_commits_fixed_limit_action_and_online_ledger() -> None:
    engine = HandEngine.start("button-hand", Seat.A)
    runtime = ButtonBettingRuntime(engine)
    assert engine.state.acting_seat is Seat.D
    assert runtime.selected_action is PlayerActionType.FOLD
    assert runtime.accept_control(control(1, ControlIntent.NEXT_OPTION)).selected_action is PlayerActionType.CALL
    outcome = runtime.accept_control(control(2, ControlIntent.CONFIRM))
    assert outcome.accepted
    assert outcome.selected_action is PlayerActionType.CALL
    assert engine.state.players[Seat.D].stack_units == 78
    assert engine.state.pot_units == 5
    assert engine.state.acting_seat is Seat.A


def test_robot_or_laptop_cannot_supply_arbitrary_fixed_limit_amount() -> None:
    engine = HandEngine.start("button-hand", Seat.A)
    runtime = ButtonBettingRuntime(engine)
    runtime.accept_control(control(1, ControlIntent.NEXT_OPTION))
    runtime.accept_control(control(2, ControlIntent.NEXT_OPTION))
    assert runtime.selected_action is PlayerActionType.RAISE
    outcome = runtime.accept_control(control(3, ControlIntent.CONFIRM))
    assert outcome.accepted
    assert engine.state.players[Seat.D].street_commit_units == 4


def test_stale_or_pre_window_robot_controls_do_not_change_ledger() -> None:
    engine = HandEngine.start("button-hand", Seat.A)
    runtime = ButtonBettingRuntime(engine)
    runtime.sync(window_opened_at_ns=100)
    before = engine.snapshot()
    stale = runtime.accept_control(
        ControlObservation(
            "old",
            ControlIntent.CONFIRM,
            ControlSource.ROBOT_BUTTON,
            99,
            "main",
            5,
        )
    )
    assert not stale.accepted
    assert stale.reason == "control_precedes_action_window"
    assert engine.state.state_version == before.state_version
    assert engine.state.pot_units == before.pot_units


def test_duplicate_and_regressing_device_versions_are_rejected() -> None:
    engine = HandEngine.start("button-hand", Seat.A)
    runtime = ButtonBettingRuntime(engine)
    first = control(1, ControlIntent.NEXT_OPTION)
    assert runtime.accept_control(first).accepted
    assert runtime.accept_control(first).reason == "duplicate_control"
    regressing = ControlObservation(
        "button:other",
        ControlIntent.NEXT_OPTION,
        ControlSource.SIMULATOR,
        2,
        "main",
        1,
    )
    assert runtime.accept_control(regressing).reason == "stale_device_state_version"
