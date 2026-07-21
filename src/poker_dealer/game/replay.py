"""Executable adapters for all frozen Stage 0 walkthrough scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    DealerCommand,
    DealerCommandType,
    DealerErrorCode,
    DealerTargetSlot,
    HandPhase,
    ObservationStatus,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    VisionSlot,
)

from .engine import ActionRequest, EventLog, HandEngine
from .evaluator import settle_showdown
from .pots import OperatorAdjustment, Pot, build_pots
from .simulators import (
    DealerFault,
    SimulatedActionPerception,
    SimulatedCardPerception,
    SimulatedDealer,
)
from poker_dealer.domain import DealerAckStatus


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    terminal_phase: str
    winner: str | None
    pause_reason: str | None
    pot_units: int | None
    next_button: str | None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    scenario_id: str
    passed: bool
    outcome: ReplayOutcome | None
    mismatches: tuple[str, ...] = ()


def _act(
    engine: HandEngine, action: PlayerActionType, label: str
) -> None:
    state = engine.state
    result = engine.apply_action(
        ActionRequest(
            action_id=f"{state.hand_id}:{label}:{state.state_version}",
            hand_id=state.hand_id,
            expected_state_version=state.state_version,
            seat=state.acting_seat,  # type: ignore[arg-type]
            action=action,
        )
    )
    if not result.accepted:
        raise AssertionError(f"{label} rejected: {result.reason}")


def _complete_preflop_calls(engine: HandEngine) -> None:
    _act(engine, PlayerActionType.CALL, "utg-call")
    _act(engine, PlayerActionType.CALL, "button-call")
    _act(engine, PlayerActionType.CALL, "sb-call")
    _act(engine, PlayerActionType.CHECK, "bb-check")


def _check_current_street(engine: HandEngine, prefix: str) -> None:
    while engine.state.phase is HandPhase.AWAITING_ACTION:
        _act(engine, PlayerActionType.CHECK, f"{prefix}-check")


def _card(rank: Rank, suit: Suit) -> CardIdentity:
    return CardIdentity(rank, suit)


def _confirm_board_street(
    engine: HandEngine, board: tuple[CardIdentity, ...], label: str
) -> None:
    slots_and_cards = {
        "flop": tuple(
            zip(
                (
                    VisionSlot.BOARD_FLOP_1,
                    VisionSlot.BOARD_FLOP_2,
                    VisionSlot.BOARD_FLOP_3,
                ),
                board[:3],
                strict=True,
            )
        ),
        "turn": ((VisionSlot.BOARD_TURN, board[3]),),
        "river": ((VisionSlot.BOARD_RIVER, board[4]),),
    }[engine.state.street.value]  # type: ignore[union-attr]
    simulator = SimulatedCardPerception()
    for index, (slot, card) in enumerate(slots_and_cards, start=1):
        observation = simulator.emit(
            slot,
            ObservationStatus.CONFIRMED,
            card=card,
            confidence=0.999,
            observed_at_ns=engine.state.state_version * 100 + index,
        )
        result = engine.apply_card_observation(observation)
        if not result.accepted:
            raise AssertionError(f"{label} card rejected: {result.reason}")
    engine.confirm_board_dealt(label)


def _settle_checkdown(engine: HandEngine) -> str:
    board = (
        _card(Rank.TWO, Suit.CLUBS),
        _card(Rank.SEVEN, Suit.DIAMONDS),
        _card(Rank.NINE, Suit.HEARTS),
        _card(Rank.JACK, Suit.SPADES),
        _card(Rank.KING, Suit.CLUBS),
    )
    while engine.state.phase is not HandPhase.SHOWDOWN:
        if engine.state.phase is HandPhase.DEALING_BOARD:
            _confirm_board_street(
                engine,
                board,
                f"{engine.state.hand_id}:confirm:{engine.state.street.value}",
            )
        elif engine.state.phase is HandPhase.AWAITING_ACTION:
            _check_current_street(engine, engine.state.street.value)
        else:
            raise AssertionError(f"unexpected phase {engine.state.phase.value}")
    holes = {
        Seat.A: (_card(Rank.ACE, Suit.DIAMONDS), _card(Rank.THREE, Suit.CLUBS)),
        Seat.B: (_card(Rank.QUEEN, Suit.DIAMONDS), _card(Rank.TEN, Suit.CLUBS)),
        Seat.C: (_card(Rank.NINE, Suit.SPADES), _card(Rank.FOUR, Suit.CLUBS)),
        Seat.D: (_card(Rank.KING, Suit.DIAMONDS), _card(Rank.FIVE, Suit.CLUBS)),
    }
    engine.settle_showdown(f"{engine.state.hand_id}:showdown", board, holes)
    return max(engine.state.awards, key=engine.state.awards.get)  # type: ignore[arg-type,return-value]


def _wt01(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt01", Seat.A)
    _act(engine, PlayerActionType.FOLD, "d-fold")
    _act(engine, PlayerActionType.FOLD, "a-fold")
    _act(engine, PlayerActionType.FOLD, "b-fold")
    winner = next(iter(engine.state.awards)).value
    return ReplayOutcome("settled", winner, None, engine.state.pot_units, engine.next_button().value)


def _wt02(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt02", Seat.A)
    _complete_preflop_calls(engine)
    winner = _settle_checkdown(engine).value
    return ReplayOutcome("settled", winner, None, 0, engine.next_button().value)


def _wt03(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt03", Seat.A)
    _act(engine, PlayerActionType.RAISE, "d-raise")
    _act(engine, PlayerActionType.RAISE, "a-raise")
    _act(engine, PlayerActionType.RAISE, "b-raise")
    state = engine.state
    rejected = engine.apply_action(
        ActionRequest(
            "wt03:c-illegal-raise",
            state.hand_id,
            state.state_version,
            state.acting_seat,  # type: ignore[arg-type]
            PlayerActionType.RAISE,
        )
    )
    if rejected.accepted or rejected.reason != "illegal_action":
        raise AssertionError("raise cap was not enforced")
    _act(engine, PlayerActionType.CALL, "c-call")
    _act(engine, PlayerActionType.CALL, "d-call")
    _act(engine, PlayerActionType.CALL, "a-call")
    return ReplayOutcome("dealing_board", None, None, engine.state.pot_units, None)


def _wt04(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt04", Seat.A)
    _complete_preflop_calls(engine)
    _confirm_board_street(
        engine,
        (
            _card(Rank.TWO, Suit.CLUBS),
            _card(Rank.SEVEN, Suit.DIAMONDS),
            _card(Rank.NINE, Suit.HEARTS),
            _card(Rank.JACK, Suit.SPADES),
            _card(Rank.KING, Suit.CLUBS),
        ),
        "wt04:flop",
    )
    _act(engine, PlayerActionType.BET, "b-bet")
    _act(engine, PlayerActionType.FOLD, "c-fold")
    _act(engine, PlayerActionType.CALL, "d-call")
    _act(engine, PlayerActionType.FOLD, "a-fold")
    return ReplayOutcome("dealing_board", None, None, engine.state.pot_units, None)


def _wt05(_: Mapping[str, Any]) -> ReplayOutcome:
    built = build_pots({Seat.A: 10, Seat.B: 20, Seat.C: 40, Seat.D: 40})
    if [pot.amount_units for pot in built.pots] != [40, 30, 40]:
        raise AssertionError("side-pot layers are incorrect")
    return ReplayOutcome("dealing_board", None, None, built.pot_units, None)


def _wt06(_: Mapping[str, Any]) -> ReplayOutcome:
    board = (
        _card(Rank.TWO, Suit.CLUBS),
        _card(Rank.THREE, Suit.DIAMONDS),
        _card(Rank.FOUR, Suit.HEARTS),
        _card(Rank.FIVE, Suit.SPADES),
        _card(Rank.NINE, Suit.CLUBS),
    )
    holes = {
        Seat.A: (_card(Rank.ACE, Suit.HEARTS), _card(Rank.KING, Suit.DIAMONDS)),
        Seat.B: (_card(Rank.ACE, Suit.SPADES), _card(Rank.QUEEN, Suit.DIAMONDS)),
        Seat.C: (_card(Rank.NINE, Suit.DIAMONDS), _card(Rank.NINE, Suit.HEARTS)),
        Seat.D: (_card(Rank.KING, Suit.CLUBS), _card(Rank.QUEEN, Suit.HEARTS)),
    }
    result = settle_showdown(
        (Pot("main", 40, (Seat.A, Seat.B)), Pot("side_1", 30, (Seat.C, Seat.D))),
        board,
        holes,
        Seat.B,
    )
    if set(result.winners_by_pot["main"]) != {Seat.A, Seat.B}:
        raise AssertionError("main pot tie was not preserved")
    if result.winners_by_pot["side_1"] != (Seat.C,):
        raise AssertionError("side pot winner is incorrect")
    return ReplayOutcome("settled", "multiple_pot_winners", None, 0, Seat.C.value)


def _wt07(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt07", Seat.A)
    cards = SimulatedCardPerception()
    cards.emit(VisionSlot.SEAT_C_HOLE_2, ObservationStatus.UNKNOWN)
    engine.pause("wt07:unknown", "card_unknown")
    return ReplayOutcome("paused_recovery", None, engine.state.paused_reason, None, None)


def _wt08(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt08", Seat.B)
    cards = SimulatedCardPerception()
    ace = _card(Rank.ACE, Suit.SPADES)
    cards.emit(VisionSlot.BOARD_FLOP_1, ObservationStatus.CONFIRMED, card=ace, confidence=0.99)
    cards.emit(VisionSlot.SEAT_A_HOLE_1, ObservationStatus.CONFIRMED, card=ace, confidence=0.99, observed_at_ns=2)
    try:
        cards.confirmed_cards()
    except ValueError:
        engine.pause("wt08:duplicate", "duplicate_card_identity")
    else:
        raise AssertionError("duplicate identity was not rejected")
    return ReplayOutcome("paused_recovery", None, engine.state.paused_reason, None, None)


def _wt09(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt09", Seat.A)
    dealer = SimulatedDealer()
    dealer.inject_fault(
        "jam",
        DealerFault(DealerAckStatus.FAILED, DealerErrorCode.FEED_JAM, "jam"),
    )
    ack = dealer.execute(DealerCommand("jam", 1, DealerCommandType.DISPENSE_ONE))
    if ack.status is not DealerAckStatus.FAILED:
        raise AssertionError("jam did not fail")
    engine.pause("wt09:pause", "feed_jam")
    return ReplayOutcome("paused_recovery", None, "feed_jam", 3, None)


def _wt10(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt10", Seat.B)
    engine.pause("wt10:timeout", "player_action_timeout")
    return ReplayOutcome("paused_recovery", None, "player_action_timeout", 3, None)


def _wt11(_: Mapping[str, Any]) -> ReplayOutcome:
    dealer = SimulatedDealer()
    dealer.execute(DealerCommand("home", 1, DealerCommandType.HOME))
    dealer.execute(DealerCommand("rotate", 2, DealerCommandType.ROTATE_TO, DealerTargetSlot.SEAT_A))
    command = DealerCommand("dispense", 3, DealerCommandType.DISPENSE_ONE)
    first = dealer.execute(command)
    second = dealer.execute(command)
    if first != second or dealer.dispensed_cards != 1:
        raise AssertionError("duplicate ACK changed dispense count")
    return ReplayOutcome("dealing_hole", None, None, 3, None)


def _wt12(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt12", Seat.D)
    engine.void("wt12:void", "misdeal_redeal_required")
    return ReplayOutcome("setup", None, "misdeal_redeal_required", 0, engine.next_button().value)


def _wt13(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt13", Seat.A)
    _act(engine, PlayerActionType.FOLD, "d-fold")
    _act(engine, PlayerActionType.FOLD, "a-fold")
    _act(engine, PlayerActionType.FOLD, "b-fold")
    return ReplayOutcome("setup", None, None, 0, engine.next_button().value)


def _wt14(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt14", Seat.A)
    actions = SimulatedActionPerception()
    observation = actions.emit(
        engine.state,
        ActionEvidenceState.CANDIDATE,
        action=PlayerActionType.RAISE,
        focus_seat=Seat.B,
        confidence=0.99,
    )
    result = engine.apply_observation(observation)
    if result.accepted or result.reason != "non_current_seat":
        raise AssertionError("non-current-seat evidence was not rejected")
    return ReplayOutcome("awaiting_action", None, None, engine.state.pot_units, None)


def _wt15(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt15", Seat.A)
    observation = SimulatedActionPerception().emit(
        engine.state, ActionEvidenceState.AMBIGUOUS
    )
    result = engine.apply_observation(observation)
    if result.accepted or result.reason != "ambiguous":
        raise AssertionError("ambiguous evidence was not held")
    return ReplayOutcome("awaiting_action", None, None, engine.state.pot_units, None)


def _wt16(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt16", Seat.A)
    observation = SimulatedActionPerception().emit(
        engine.state,
        ActionEvidenceState.CANDIDATE,
        action=PlayerActionType.CALL,
        confidence=0.99,
    )
    result = engine.apply_observation(observation)
    if not result.accepted or engine.state.acting_seat is not Seat.A:
        raise AssertionError("accepted evidence did not atomically switch focus")
    return ReplayOutcome("awaiting_action", None, None, engine.state.pot_units, None)


def _wt17(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.start("wt17", Seat.A)
    _complete_preflop_calls(engine)
    dealer = SimulatedDealer()
    dealer.execute(DealerCommand("home", 1, DealerCommandType.HOME))
    dealer.execute(DealerCommand("rotate", 2, DealerCommandType.ROTATE_TO, DealerTargetSlot.BOARD_FLOP_1))
    dealer.execute(DealerCommand("dispense", 3, DealerCommandType.DISPENSE_ONE))
    SimulatedCardPerception().emit(
        VisionSlot.BOARD_FLOP_1, ObservationStatus.FACE_UP_UNCONFIRMED
    )
    if engine.state.phase is not HandPhase.DEALING_BOARD:
        raise AssertionError("ACK without visible identity advanced the board")
    return ReplayOutcome("dealing_board", None, None, engine.state.pot_units, None)


def _wt18(_: Mapping[str, Any]) -> ReplayOutcome:
    engine = HandEngine.setup_session(
        "wt18",
        Seat.A,
        {Seat.A: 0, Seat.B: 80, Seat.C: 80, Seat.D: 80},
    )
    engine.apply_operator_adjustment(
        "wt18:rebuy",
        OperatorAdjustment(Seat.A, 80, "operator-1", "rebuy"),
    )
    if engine.state.players[Seat.A].stack_units != 80:
        raise AssertionError("audited rebuy did not update the digital ledger")
    recovered = HandEngine.from_log(
        engine.rules, EventLog.from_jsonl(engine.log.to_jsonl())
    )
    if recovered.state.state_version != 1:
        raise AssertionError("audited rebuy did not advance/recover state version")
    return ReplayOutcome("setup", None, None, 0, None)


HANDLERS: dict[str, Callable[[Mapping[str, Any]], ReplayOutcome]] = {
    f"WT-{number:02d}": handler
    for number, handler in enumerate(
        (
            _wt01,
            _wt02,
            _wt03,
            _wt04,
            _wt05,
            _wt06,
            _wt07,
            _wt08,
            _wt09,
            _wt10,
            _wt11,
            _wt12,
            _wt13,
            _wt14,
            _wt15,
            _wt16,
            _wt17,
            _wt18,
        ),
        start=1,
    )
}


def _compare(expected: Mapping[str, Any], actual: ReplayOutcome) -> tuple[str, ...]:
    mismatches: list[str] = []
    for field in ("terminal_phase", "pause_reason", "pot_units", "next_button"):
        if expected[field] != getattr(actual, field):
            mismatches.append(
                f"{field}: expected {expected[field]!r}, got {getattr(actual, field)!r}"
            )
    expected_winner = expected["winner"]
    if expected_winner in {"multi_player_evaluator_result", "multiple_pot_winners"}:
        if actual.winner is None:
            mismatches.append("winner: expected an evaluator result, got None")
    elif expected_winner != actual.winner:
        mismatches.append(
            f"winner: expected {expected_winner!r}, got {actual.winner!r}"
        )
    return tuple(mismatches)


def run_walkthroughs(path: str | Path) -> tuple[ReplayResult, ...]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = document["scenarios"]
    configured_prefixes = {scenario["id"].split("-", 2)[0] + "-" + scenario["id"].split("-", 2)[1] for scenario in scenarios}
    if configured_prefixes != set(HANDLERS):
        raise ValueError("walkthrough handlers do not exactly cover configured IDs")
    results: list[ReplayResult] = []
    for scenario in scenarios:
        prefix = "-".join(scenario["id"].split("-")[:2])
        try:
            outcome = HANDLERS[prefix](scenario)
            mismatches = _compare(scenario["expected"], outcome)
            results.append(ReplayResult(scenario["id"], not mismatches, outcome, mismatches))
        except Exception as exc:  # replay evidence must retain the failing case
            results.append(ReplayResult(scenario["id"], False, None, (str(exc),)))
    return tuple(results)
