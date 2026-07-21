from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from poker_dealer.domain import (
    CardIdentity,
    HandPhase,
    ObservationStatus,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    VisionSlot,
)
from poker_dealer.game import (
    ActionRequest,
    EventLog,
    FixedLimitRules,
    HandEngine,
    SimulatedCardPerception,
    SlotLifecycle,
    state_to_contract_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


def _act(engine: HandEngine, action: PlayerActionType, label: str) -> None:
    state = engine.state
    result = engine.apply_action(
        ActionRequest(
            label,
            state.hand_id,
            state.state_version,
            state.acting_seat,  # type: ignore[arg-type]
            action,
        )
    )
    assert result.accepted, result.reason


def _reach_flop_delivery(engine: HandEngine) -> None:
    for action in (
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CHECK,
    ):
        _act(engine, action, f"preflop-{engine.state.state_version}")
    assert engine.state.phase is HandPhase.DEALING_BOARD


def _confirmed(
    simulator: SimulatedCardPerception,
    slot: VisionSlot,
    card: CardIdentity,
    timestamp: int,
):
    return simulator.emit(
        slot,
        ObservationStatus.CONFIRMED,
        card=card,
        confidence=0.999,
        observed_at_ns=timestamp,
    )


def test_board_does_not_advance_until_all_active_slots_are_confirmed() -> None:
    engine = HandEngine.start("scene-gate", Seat.A)
    _reach_flop_delivery(engine)
    with pytest.raises(ValueError, match="not confirmed"):
        engine.confirm_board_dealt("too-early")

    simulator = SimulatedCardPerception()
    original_version = engine.state.state_version
    unknown = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.UNKNOWN,
        observed_at_ns=1,
    )
    result = engine.apply_card_observation(unknown)
    assert not result.accepted and result.reason == "unknown"
    assert engine.state.state_version == original_version
    assert engine.state.phase is HandPhase.DEALING_BOARD

    unconfirmed = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.FACE_UP_UNCONFIRMED,
        observed_at_ns=2,
    )
    assert engine.apply_card_observation(unconfirmed).accepted
    assert (
        engine.state.slot_states[VisionSlot.BOARD_FLOP_1]
        is SlotLifecycle.FACE_UP_UNCONFIRMED
    )
    with pytest.raises(ValueError, match="not confirmed"):
        engine.confirm_board_dealt("still-too-early")

    cards = (
        CardIdentity(Rank.ACE, Suit.SPADES),
        CardIdentity(Rank.KING, Suit.HEARTS),
        CardIdentity(Rank.QUEEN, Suit.DIAMONDS),
    )
    for timestamp, (slot, card) in enumerate(
        zip(
            (
                VisionSlot.BOARD_FLOP_1,
                VisionSlot.BOARD_FLOP_2,
                VisionSlot.BOARD_FLOP_3,
            ),
            cards,
            strict=True,
        ),
        start=3,
    ):
        assert engine.apply_card_observation(
            _confirmed(simulator, slot, card, timestamp)
        ).accepted

    engine.confirm_board_dealt("flop-confirmed")
    assert engine.state.phase is HandPhase.AWAITING_ACTION
    assert engine.state.board == cards


def test_duplicate_card_identity_freezes_hand_and_marks_conflicts() -> None:
    engine = HandEngine.start("duplicate", Seat.A)
    _reach_flop_delivery(engine)
    simulator = SimulatedCardPerception()
    ace = CardIdentity(Rank.ACE, Suit.SPADES)
    assert engine.apply_card_observation(
        _confirmed(simulator, VisionSlot.BOARD_FLOP_1, ace, 1)
    ).accepted
    duplicate = engine.apply_card_observation(
        _confirmed(simulator, VisionSlot.BOARD_FLOP_2, ace, 2)
    )
    assert not duplicate.accepted
    assert duplicate.reason == "duplicate_card_identity"
    assert engine.state.phase is HandPhase.PAUSED_RECOVERY
    assert engine.state.paused_reason == "duplicate_card_identity"
    assert engine.state.slot_states[VisionSlot.BOARD_FLOP_1] is SlotLifecycle.CONFLICT
    assert engine.state.slot_states[VisionSlot.BOARD_FLOP_2] is SlotLifecycle.CONFLICT


def test_contract_snapshot_validates_and_slot_state_recovers_from_log() -> None:
    engine = HandEngine.start("snapshot", Seat.B)
    schema = json.loads(
        (ROOT / "configs/contracts/hand_snapshot.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(
        state_to_contract_snapshot(engine.state)
    )

    _reach_flop_delivery(engine)
    simulator = SimulatedCardPerception()
    observation = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.FACE_UP_UNCONFIRMED,
        observed_at_ns=10,
    )
    assert engine.apply_card_observation(observation).accepted
    recovered = HandEngine.from_log(
        FixedLimitRules(), EventLog.from_jsonl(engine.log.to_jsonl())
    )
    assert (
        recovered.state.slot_states[VisionSlot.BOARD_FLOP_1]
        is SlotLifecycle.FACE_UP_UNCONFIRMED
    )
    Draft202012Validator(schema).validate(
        state_to_contract_snapshot(recovered.state)
    )
