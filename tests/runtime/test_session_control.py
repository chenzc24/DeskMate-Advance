from __future__ import annotations

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    ControlSource,
    Seat,
)
from poker_dealer.game import CoreGameConfig
from poker_dealer.runtime import (
    FrameRead,
    FrameReadState,
    LiveSessionOperatorUI,
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


def test_hand_limit_requires_explicit_end_after_table_clear() -> None:
    session = _session()
    runtime = session.start_hand("hand-1")
    runtime.void("void-1", "test")
    session.close_terminal_hand()
    controller = SessionOperatorController(session, operator_id="operator-a")

    class FrameSource:
        def __init__(self) -> None:
            self.sequence = 0

        def set_status(self, *lines: str) -> None:
            del lines

        def read(self) -> FrameRead:
            self.sequence += 1
            return FrameRead(FrameReadState.MISSING, self.sequence, None)

    class Controls:
        def __init__(self) -> None:
            self.items = [
                _control(1, ControlIntent.CONFIRM),
                _control(2, ControlIntent.CLEAR),
            ]

        def poll_controls(self, observed_at_ns: int):
            del observed_at_ns
            return (self.items.pop(0),) if self.items else ()

    class Observer:
        def __init__(self) -> None:
            self.phases: list[str] = []

        def publish_session_state(self, session, **kwargs) -> None:
            if session.ended:
                phase = "session_ended"
            elif not session.table_cleared:
                phase = "table_clearance"
            elif kwargs["stop_after_clear"]:
                phase = "ready_session_end"
            else:
                phase = "ready_next_hand"
            self.phases.append(phase)

    observer = Observer()
    ui = LiveSessionOperatorUI(
        FrameSource(),
        Controls(),
        state_observer=observer,
    )

    result = ui.wait_for_decision(
        session,
        controller,
        timeout_seconds=1,
        stop_after_clear=True,
    )

    assert result.signal is SessionOperatorSignal.SESSION_ENDED
    assert observer.phases == [
        "table_clearance",
        "ready_session_end",
        "session_ended",
    ]
    assert [event.kind for event in session.events[-2:]] == [
        "table_cleared",
        "session_ended",
    ]
