from __future__ import annotations

import hashlib
import json
from pathlib import Path

from poker_dealer.domain import PlayerActionType, Seat
from poker_dealer.game import CoreGameConfig
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.runtime import (
    HandRuntimeLoop,
    RuntimeEventLog,
    RuntimeEventWriter,
    ScriptedReplaySources,
    SessionEventLog,
    SessionEventWriter,
    SessionRuntime,
    StepClock,
    check_runtime_hand_log,
    check_session_log,
    default_replay_roster,
)


def _run_hand(session: SessionRuntime, hand_id: str, path: Path) -> None:
    runtime = session.start_hand(hand_id)
    sources = ScriptedReplaySources(
        action_selector=lambda _context: PlayerActionType.FOLD
    )
    dealer = SimulatedDealerAdapter(f"session-log:{hand_id}")
    dealer.open()
    try:
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
    finally:
        dealer.close()
    checked = check_runtime_hand_log(RuntimeEventLog.from_path(path))
    assert result.completed and checked.passed
    session.close_terminal_hand(
        hand_log_path=str(path),
        hand_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        hand_log_check_passed=True,
    )


def test_session_log_checks_two_hand_continuity_and_round_trips(tmp_path: Path) -> None:
    session = SessionRuntime(
        default_replay_roster("session-a", Seat.A),
        CoreGameConfig.from_json("configs/game/core_v1.json"),
    )
    output = tmp_path / "session.jsonl"
    with SessionEventWriter(output) as writer:
        writer.sync(session.log)
        for index in (1, 2):
            _run_hand(session, f"hand-{index}", tmp_path / f"hand-{index}.jsonl")
            writer.sync(session.log)
            session.confirm_table_cleared(operator_id="operator-a")
            writer.sync(session.log)
        session.end_session(operator_id="operator-a", reason="test_complete")
        writer.sync(session.log)

    checked = check_session_log(SessionEventLog.from_path(output))
    assert checked.passed
    assert checked.hands_started == checked.hands_closed == 2
    assert checked.ended


def test_session_log_tampering_is_rejected(tmp_path: Path) -> None:
    session = SessionRuntime(
        default_replay_roster("session-a"),
        CoreGameConfig.from_json("configs/game/core_v1.json"),
    )
    session.end_session(operator_id="operator-a", reason="test_complete")
    lines = session.log.to_jsonl().splitlines()
    changed = json.loads(lines[-1])
    changed["payload"]["reason"] = "tampered"
    lines[-1] = json.dumps(changed)
    path = tmp_path / "tampered.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        SessionEventLog.from_path(path)
    except ValueError as exc:
        assert "content hash" in str(exc)
    else:
        raise AssertionError("tampered session log was accepted")


def test_voided_hand_keeps_button_and_can_be_followed_by_redeal(tmp_path: Path) -> None:
    session = SessionRuntime(
        default_replay_roster("void-session", Seat.C),
        CoreGameConfig.from_json("configs/game/core_v1.json"),
    )
    runtime = session.start_hand("voided-hand")
    runtime.void("void-1", "operator_redeal")
    hand_path = tmp_path / "voided-hand.jsonl"
    with RuntimeEventWriter(hand_path) as hand_writer:
        hand_writer.sync_engine(runtime.engine.log)
    checked = check_runtime_hand_log(
        RuntimeEventLog.from_path(hand_path), allow_voided=True
    )
    assert checked.passed
    session.close_terminal_hand(
        hand_log_path=str(hand_path),
        hand_log_sha256=hashlib.sha256(hand_path.read_bytes()).hexdigest(),
        hand_log_check_passed=True,
    )
    assert session.button is Seat.C
    session.confirm_table_cleared(operator_id="operator-a")
    _run_hand(session, "redeal", tmp_path / "redeal.jsonl")
    session.confirm_table_cleared(operator_id="operator-a")
    session.end_session(operator_id="operator-a", reason="test_complete")
    assert check_session_log(session.log, verify_hand_logs=True).passed
