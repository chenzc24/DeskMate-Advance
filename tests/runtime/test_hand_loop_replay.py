from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_dealer.domain import HandPhase, PlayerActionType, Seat
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.runtime import HandRuntime
from poker_dealer.runtime.event_log import (
    RuntimeEventLog,
    RuntimeEventWriter,
    check_runtime_hand_log,
)
from poker_dealer.runtime.hand_loop import HandRuntimeLoop
from poker_dealer.runtime.replay import (
    RecordedReplaySources,
    ScriptedReplaySources,
    StepClock,
    default_replay_roster,
)
from poker_dealer.runtime.ports import FrameRead, FrameReadState


def _run_complete_hand(
    path: Path,
    sources,
    *,
    session_id: str = "replay-session",
    hand_id: str = "replayed-hand",
    stacks=None,
) -> HandRuntime:
    runtime = HandRuntime.from_roster(
        hand_id=hand_id,
        roster=default_replay_roster(session_id, Seat.A),
        require_actor_binding=True,
        require_visual_settle=True,
        stacks=stacks,
    )
    dealer = SimulatedDealerAdapter(f"sim:{hand_id}")
    dealer.open()
    with RuntimeEventWriter(path) as writer:
        loop = HandRuntimeLoop(
            runtime,
            dealer,
            identity_source=sources,
            action_source=sources,
            card_source=sources,
            visual_settle_source=sources,
            event_writer=writer,
            clock_ns=StepClock(),
        )
        result = loop.run(max_steps=500)
    dealer.close()
    assert result.completed
    assert result.hand_phase is HandPhase.SETTLED
    return runtime


def test_vertical_replay_fold_path_settles_uncontested(tmp_path: Path) -> None:
    runtime = _run_complete_hand(
        tmp_path / "fold.jsonl",
        ScriptedReplaySources(
            action_selector=lambda context: PlayerActionType.FOLD
        ),
        hand_id="fold-path",
    )
    assert sum(player.folded for player in runtime.engine.state.players.values()) == 3
    assert runtime.engine.state.board == ()


def test_vertical_replay_raise_and_short_all_in_path(tmp_path: Path) -> None:
    raised = False

    def choose(context):
        nonlocal raised
        if not raised and PlayerActionType.RAISE in context.legal_actions:
            raised = True
            return PlayerActionType.RAISE
        if PlayerActionType.CALL in context.legal_actions:
            return PlayerActionType.CALL
        return PlayerActionType.CHECK

    runtime = _run_complete_hand(
        tmp_path / "all-in.jsonl",
        ScriptedReplaySources(action_selector=choose),
        hand_id="raise-all-in-path",
        stacks={seat: 4 for seat in Seat},
    )
    assert raised
    assert runtime.phase is HandPhase.SETTLED
    assert runtime.engine.state.total_units() == 16
    assert any(
        event.kind == "action_applied"
        and event.payload.get("action") == "raise"
        for event in runtime.engine.log.events
    )


def test_complete_hand_is_logged_checked_and_replayed_exactly(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    first = _run_complete_hand(first_path, ScriptedReplaySources())
    first_log = RuntimeEventLog.from_path(first_path)
    check = check_runtime_hand_log(first_log)
    assert check.passed, check.issues
    assert check.phase == "settled"
    assert check.evidence_records > 20

    second_path = tmp_path / "second.jsonl"
    second = _run_complete_hand(
        second_path,
        RecordedReplaySources(first_log),
    )
    assert second.engine.state.players == first.engine.state.players
    assert second.engine.state.awards == first.engine.state.awards
    assert second.engine.state.confirmed_cards == first.engine.state.confirmed_cards
    assert check_runtime_hand_log(RuntimeEventLog.from_path(second_path)).passed


def test_runtime_log_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "hand.jsonl"
    _run_complete_hand(path, ScriptedReplaySources())
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["kind"] = "forged"
    lines[-1] = json.dumps(record)
    with pytest.raises(ValueError, match="content hash"):
        RuntimeEventLog.from_jsonl("\n".join(lines))


def test_runtime_writer_never_overwrites_existing_log(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RuntimeEventWriter(path)


def test_camera_disconnect_pauses_before_card_source_can_advance(tmp_path: Path) -> None:
    class DisconnectedFrameSource:
        def open(self) -> None:
            return None

        def read(self) -> FrameRead:
            return FrameRead(
                FrameReadState.DISCONNECTED,
                5_000_000,
                None,
                reason="test_disconnect",
            )

        def close(self) -> None:
            return None

    sources = ScriptedReplaySources()
    runtime = HandRuntime.from_roster(
        hand_id="camera-disconnect",
        roster=default_replay_roster(),
        require_actor_binding=True,
        require_visual_settle=True,
    )
    dealer = SimulatedDealerAdapter("sim:disconnect")
    dealer.open()
    with RuntimeEventWriter(tmp_path / "disconnect.jsonl") as writer:
        result = HandRuntimeLoop(
            runtime,
            dealer,
            identity_source=sources,
            action_source=sources,
            card_source=sources,
            visual_settle_source=sources,
            frame_source=DisconnectedFrameSource(),
            event_writer=writer,
            clock_ns=StepClock(),
        ).run(max_steps=10)
    dealer.close()
    assert result.completed is False
    assert result.hand_phase is HandPhase.PAUSED_RECOVERY
    assert runtime.engine.state.paused_reason == "camera_disconnected"
    log = RuntimeEventLog.from_path(tmp_path / "disconnect.jsonl")
    strict = check_runtime_hand_log(log)
    assert not strict.passed
    assert "hand_not_settled:paused_recovery" in strict.issues
    assert check_runtime_hand_log(log, require_settled=False).passed
