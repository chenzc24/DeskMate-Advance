from pathlib import Path

from poker_dealer.domain import HandPhase, PlayerActionType, Seat
from poker_dealer.game import CoreGameConfig
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.runtime import (
    HandRuntimeLoop,
    RuntimeEventWriter,
    ScriptedReplaySources,
    SessionRuntime,
    StepClock,
    default_replay_roster,
)


def _finish_by_folding(runtime, path: Path) -> None:
    sources = ScriptedReplaySources(
        action_selector=lambda context: PlayerActionType.FOLD
    )
    dealer = SimulatedDealerAdapter(f"sim:{runtime.engine.state.hand_id}")
    dealer.open()
    with RuntimeEventWriter(path) as writer:
        result = HandRuntimeLoop(
            runtime,
            dealer,
            identity_source=sources,
            action_source=sources,
            card_source=sources,
            visual_settle_source=sources,
            event_writer=writer,
            clock_ns=StepClock(),
        ).run(max_steps=200)
    dealer.close()
    assert result.completed and runtime.phase is HandPhase.SETTLED


def test_two_hand_session_preserves_stacks_and_rotates_button(tmp_path: Path) -> None:
    config = CoreGameConfig.from_json("configs/game/core_v1.json")
    session = SessionRuntime(default_replay_roster(button=Seat.A), config)
    first = session.start_hand("hand-1")
    _finish_by_folding(first, tmp_path / "hand-1.jsonl")
    expected_stacks = {
        seat: first.engine.state.players[seat].stack_units for seat in Seat
    }
    session.close_terminal_hand()
    assert session.button is Seat.B
    assert session.stacks == expected_stacks

    session.confirm_table_cleared(operator_id="operator-1")
    second = session.start_hand("hand-2")
    assert second.engine.state.button is Seat.B
    assert {
        seat: player.stack_units + player.hand_commit_units
        for seat, player in second.engine.state.players.items()
    } == expected_stacks


def test_session_requires_table_clearance_and_audits_rebuy(tmp_path: Path) -> None:
    config = CoreGameConfig.from_json("configs/game/core_v1.json")
    session = SessionRuntime(default_replay_roster(), config)
    runtime = session.start_hand("hand-1")
    runtime.void("void-1", "operator_abort")
    session.close_terminal_hand()
    try:
        session.start_hand("hand-2")
    except ValueError as exc:
        assert "table clearance" in str(exc)
    else:
        raise AssertionError("next hand started before table clearance")
    session.adjust_stack(
        adjustment_id="rebuy-1",
        seat=Seat.A,
        amount_units=20,
        operator_id="operator-1",
        reason="consented_rebuy",
    )
    assert session.events[-1].kind == "stack_adjusted"
