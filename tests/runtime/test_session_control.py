from __future__ import annotations

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    ControlSource,
    Seat,
)
from poker_dealer.game import CoreGameConfig
from poker_dealer.runtime import (
    SessionOperatorController,
    SessionOperatorSignal,
    SessionRuntime,
    default_replay_roster,
)


def _control(sequence: int, intent: ControlIntent) -> ControlObservation:
    return ControlObservation(
        f"session-control:{sequence}",
        intent,
        ControlSource.SIMULATOR,
        sequence,
        "operator-panel",
        sequence,
    )


def _session() -> SessionRuntime:
    return SessionRuntime(
        default_replay_roster("session-control"),
        CoreGameConfig.from_json("configs/game/core_v1.json"),
    )


def test_between_hand_controls_require_clear_then_start() -> None:
    session = _session()
    runtime = session.start_hand("hand-1")
    runtime.void("void-1", "test")
    session.close_terminal_hand()
    controller = SessionOperatorController(session, operator_id="operator-a")

    assert not controller.accept(_control(1, ControlIntent.START)).accepted
    assert controller.accept(_control(2, ControlIntent.CONFIRM)).reason == "table_cleared"
    outcome = controller.accept(_control(3, ControlIntent.START))
    assert outcome.signal is SessionOperatorSignal.START_NEXT_HAND


def test_low_stack_requires_rebuy_or_session_end() -> None:
    session = _session()
    session.stacks[Seat.A] = 1
    controller = SessionOperatorController(
        session, operator_id="operator-a", rebuy_to_units=20
    )
    assert not controller.accept(_control(1, ControlIntent.START)).accepted
    rebuy = controller.accept(_control(2, ControlIntent.CONFIRM))
    assert rebuy.reason == "rebuy_applied"
    assert session.stacks[Seat.A] == 20


def test_paused_hand_can_retry_or_void_through_session_authority() -> None:
    session = _session()
    runtime = session.start_hand("hand-1")
    runtime.engine.pause("pause-1", "test_fault")
    controller = SessionOperatorController(session, operator_id="operator-a")
    retry = controller.accept(_control(1, ControlIntent.START))
    assert retry.signal is SessionOperatorSignal.RETRY_HAND

    runtime.engine.pause("pause-2", "second_fault")
    voided = controller.accept(_control(2, ControlIntent.CLEAR))
    assert voided.signal is SessionOperatorSignal.HAND_VOIDED
    assert runtime.phase.value == "voided"
