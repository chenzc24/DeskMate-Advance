from __future__ import annotations

import json

import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    HandPhase,
    PlayerActionType,
    SEAT_ORDER,
    Seat,
    Street,
)
from poker_dealer.game import (
    ActionRequest,
    EventLog,
    FixedLimitRules,
    HandEngine,
    OperatorAdjustment,
    SimulatedActionPerception,
    state_to_dict,
)


def request(engine: HandEngine, action: PlayerActionType, action_id: str) -> ActionRequest:
    state = engine.state
    return ActionRequest(
        action_id,
        state.hand_id,
        state.state_version,
        state.acting_seat,  # type: ignore[arg-type]
        action,
    )


@pytest.mark.parametrize("button", list(SEAT_ORDER))
def test_new_hand_posts_blinds_and_focuses_utg(button: Seat) -> None:
    engine = HandEngine.start("hand", button)
    clockwise = tuple(SEAT_ORDER[(SEAT_ORDER.index(button) + i) % 4] for i in (1, 2, 3, 4))
    assert engine.state.small_blind_seat is clockwise[0]
    assert engine.state.big_blind_seat is clockwise[1]
    assert engine.state.acting_seat is clockwise[2]
    assert engine.state.pot_units == 3
    assert engine.state.total_units() == 320


def test_action_evidence_rejection_and_atomic_focus_switch() -> None:
    engine = HandEngine.start("hand", Seat.A)
    simulator = SimulatedActionPerception()
    original = state_to_dict(engine.state)

    non_current = simulator.emit(
        engine.state,
        ActionEvidenceState.CANDIDATE,
        action=PlayerActionType.RAISE,
        focus_seat=Seat.B,
        confidence=0.99,
    )
    result = engine.apply_observation(non_current)
    assert not result.accepted and result.reason == "non_current_seat"
    assert state_to_dict(engine.state) == original

    ambiguous = simulator.emit(engine.state, ActionEvidenceState.AMBIGUOUS)
    result = engine.apply_observation(ambiguous)
    assert not result.accepted and result.reason == "ambiguous"
    assert state_to_dict(engine.state) == original

    accepted = simulator.emit(
        engine.state,
        ActionEvidenceState.CANDIDATE,
        action=PlayerActionType.CALL,
        confidence=0.99,
    )
    result = engine.apply_observation(accepted)
    assert result.accepted
    assert engine.state.state_version == 1
    assert engine.state.pot_units == 5
    assert engine.state.acting_seat is Seat.A
    assert engine.state.players[Seat.D].stack_units == 78


@pytest.mark.parametrize(
    "evidence_state",
    [
        ActionEvidenceState.NO_ACTION,
        ActionEvidenceState.ACTION_START,
        ActionEvidenceState.AMBIGUOUS,
        ActionEvidenceState.OCCLUDED,
        ActionEvidenceState.OUT_OF_ROI,
        ActionEvidenceState.UNKNOWN,
    ],
)
def test_non_candidate_evidence_never_changes_state(
    evidence_state: ActionEvidenceState,
) -> None:
    engine = HandEngine.start("evidence", Seat.A)
    original = state_to_dict(engine.state)
    observation = SimulatedActionPerception().emit(
        engine.state, evidence_state
    )
    result = engine.apply_observation(observation)
    assert not result.accepted and result.reason == evidence_state.value
    assert state_to_dict(engine.state) == original


def test_stale_duplicate_illegal_and_fixed_limit_amount_do_not_mutate() -> None:
    engine = HandEngine.start("hand", Seat.A)
    stale = ActionRequest("stale", "hand", 99, Seat.D, PlayerActionType.CALL)
    assert engine.apply_action(stale).reason == "stale_state_version"
    snapshot = state_to_dict(engine.state)
    assert engine.apply_action(stale).reason == "duplicate_action_id"
    assert state_to_dict(engine.state) == snapshot

    illegal = request(engine, PlayerActionType.CHECK, "illegal-check")
    assert engine.apply_action(illegal).reason == "illegal_action"
    assert state_to_dict(engine.state) == snapshot

    amount = ActionRequest("amount", "hand", 0, Seat.D, PlayerActionType.CALL, 2)
    assert engine.apply_action(amount).reason == "fixed_limit_amount_must_be_null"
    assert state_to_dict(engine.state) == snapshot


def test_candidate_raise_cap_and_street_transition() -> None:
    engine = HandEngine.start("hand", Seat.A)
    for index in range(3):
        result = engine.apply_action(request(engine, PlayerActionType.RAISE, f"raise-{index}"))
        assert result.accepted
    assert PlayerActionType.RAISE not in engine.state.legal_actions
    assert engine.state.full_bets_this_street == 4
    for index in range(3):
        assert engine.apply_action(request(engine, PlayerActionType.CALL, f"call-{index}")).accepted
    assert engine.state.phase is HandPhase.DEALING_BOARD
    assert engine.state.street is Street.FLOP
    assert engine.state.pot_units == 32


def test_short_all_in_raise_does_not_reopen_prior_players() -> None:
    engine = HandEngine.start(
        "short-all-in",
        Seat.A,
        {Seat.A: 80, Seat.B: 3, Seat.C: 80, Seat.D: 80},
    )
    assert engine.apply_action(request(engine, PlayerActionType.CALL, "d-call")).accepted
    assert engine.apply_action(request(engine, PlayerActionType.CALL, "a-call")).accepted
    assert engine.state.acting_seat is Seat.B
    assert engine.apply_action(
        request(engine, PlayerActionType.RAISE, "b-short-raise")
    ).accepted
    assert engine.state.players[Seat.B].all_in
    assert engine.state.current_bet_units == 3
    assert engine.state.full_bets_this_street == 1
    assert engine.apply_action(request(engine, PlayerActionType.CALL, "c-call")).accepted
    assert engine.state.acting_seat is Seat.D
    assert PlayerActionType.RAISE not in engine.state.legal_actions
    assert engine.apply_action(request(engine, PlayerActionType.CALL, "d-call-again")).accepted
    assert engine.state.acting_seat is Seat.A
    assert PlayerActionType.RAISE not in engine.state.legal_actions
    assert engine.apply_action(request(engine, PlayerActionType.CALL, "a-call-again")).accepted
    assert engine.state.phase is HandPhase.DEALING_BOARD
    assert engine.state.pot_units == 12


def test_uncontested_settlement_and_button_rotation() -> None:
    engine = HandEngine.start("hand", Seat.A)
    for index in range(3):
        assert engine.apply_action(request(engine, PlayerActionType.FOLD, f"fold-{index}")).accepted
    assert engine.state.phase is HandPhase.SETTLED
    assert engine.state.awards == {Seat.C: 3}
    assert engine.state.total_units() == 320
    assert engine.next_button() is Seat.B


def test_append_only_log_json_recovery_and_tamper_detection() -> None:
    engine = HandEngine.start("hand", Seat.A)
    engine.apply_action(request(engine, PlayerActionType.CALL, "call"))
    text = engine.log.to_jsonl()
    recovered_log = EventLog.from_jsonl(text)
    recovered = HandEngine.from_log(FixedLimitRules(), recovered_log)
    assert state_to_dict(recovered.state) == state_to_dict(engine.state)
    assert recovered.log.to_jsonl() == text

    lines = text.splitlines()
    tampered = json.loads(lines[-1])
    tampered["state_after"]["pot_units"] = 999
    lines[-1] = json.dumps(tampered)
    with pytest.raises(ValueError, match="hash"):
        EventLog.from_jsonl("\n".join(lines))


def test_void_returns_contributions_and_keeps_button() -> None:
    engine = HandEngine.start("hand", Seat.D)
    engine.void("void", "misdeal")
    assert engine.state.phase is HandPhase.VOIDED
    assert engine.state.total_units() == 320
    assert engine.next_button() is Seat.D


def test_operator_adjustment_is_setup_only_audited_idempotent_and_recoverable() -> None:
    engine = HandEngine.setup_session(
        "setup",
        Seat.A,
        {Seat.A: 0, Seat.B: 80, Seat.C: 80, Seat.D: 80},
    )
    adjustment = OperatorAdjustment(Seat.A, 80, "operator-1", "rebuy")
    engine.apply_operator_adjustment("rebuy-1", adjustment)
    assert engine.state.players[Seat.A].stack_units == 80
    assert engine.state.state_version == 1
    assert engine.log.events[-1].payload["operator_id"] == "operator-1"
    engine.apply_operator_adjustment("rebuy-1", adjustment)
    assert engine.state.players[Seat.A].stack_units == 80
    assert engine.state.state_version == 1

    recovered = HandEngine.from_log(
        FixedLimitRules(), EventLog.from_jsonl(engine.log.to_jsonl())
    )
    recovered.apply_operator_adjustment("rebuy-1", adjustment)
    assert recovered.state.players[Seat.A].stack_units == 80
    assert recovered.state.state_version == 1

    live = HandEngine.start("live", Seat.A)
    with pytest.raises(ValueError, match="only in setup"):
        live.apply_operator_adjustment("forbidden", adjustment)
