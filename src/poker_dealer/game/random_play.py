"""Seeded legal-hand generator for Stage 1 property and soak tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_dealer.domain import (
    CardIdentity,
    HandPhase,
    ObservationStatus,
    Rank,
    SEAT_ORDER,
    Seat,
    Street,
    Suit,
    VisionSlot,
)

from .engine import ActionRequest, FixedLimitRules, HandEngine, HandState
from .simulators import SimulatedCardPerception


FULL_DECK = tuple(CardIdentity(rank, suit) for suit in Suit for rank in Rank)


@dataclass(frozen=True, slots=True)
class RandomRunSummary:
    hands: int
    actions: int
    showdowns: int
    folds: int
    seed: int


def _deal_cards(
    rng: random.Random,
) -> tuple[
    tuple[CardIdentity, ...], dict[Seat, tuple[CardIdentity, CardIdentity]]
]:
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    board = tuple(deck[:5])
    cursor = 5
    holes: dict[Seat, tuple[CardIdentity, CardIdentity]] = {}
    for seat in SEAT_ORDER:
        holes[seat] = (deck[cursor], deck[cursor + 1])
        cursor += 2
    return board, holes


def play_random_hand(
    hand_number: int,
    rng: random.Random,
    rules: FixedLimitRules | None = None,
) -> tuple[HandState, int, bool]:
    button = SEAT_ORDER[hand_number % len(SEAT_ORDER)]
    engine = HandEngine.start(
        f"random-{hand_number}", button, rules=rules or FixedLimitRules()
    )
    initial_total = engine.state.total_units()
    board, all_holes = _deal_cards(rng)
    card_simulator = SimulatedCardPerception()
    actions = 0
    showdown = False

    for step in range(200):
        state = engine.state
        if state.phase is HandPhase.AWAITING_ACTION:
            action = rng.choice(state.legal_actions)
            request = ActionRequest(
                action_id=f"random-{hand_number}-action-{actions}",
                hand_id=state.hand_id,
                expected_state_version=state.state_version,
                seat=state.acting_seat,  # type: ignore[arg-type]
                action=action,
            )
            result = engine.apply_action(request)
            if not result.accepted:
                raise AssertionError(f"generated legal action rejected: {result.reason}")
            actions += 1
        elif state.phase is HandPhase.DEALING_BOARD:
            slots_and_cards = {
                Street.FLOP: tuple(zip(
                    (
                        VisionSlot.BOARD_FLOP_1,
                        VisionSlot.BOARD_FLOP_2,
                        VisionSlot.BOARD_FLOP_3,
                    ),
                    board[:3],
                    strict=True,
                )),
                Street.TURN: ((VisionSlot.BOARD_TURN, board[3]),),
                Street.RIVER: ((VisionSlot.BOARD_RIVER, board[4]),),
            }[state.street]  # type: ignore[index]
            for offset, (slot, card) in enumerate(slots_and_cards):
                observation = card_simulator.emit(
                    slot,
                    ObservationStatus.CONFIRMED,
                    card=card,
                    confidence=0.999,
                    observed_at_ns=hand_number * 1_000 + step * 10 + offset + 1,
                )
                result = engine.apply_card_observation(observation)
                if not result.accepted:
                    raise AssertionError(
                        f"generated board observation rejected: {result.reason}"
                    )
            engine.confirm_board_dealt(f"random-{hand_number}-board-{step}")
        elif state.phase is HandPhase.SHOWDOWN:
            holes = {seat: all_holes[seat] for seat in state.live_seats()}
            engine.settle_showdown(
                f"random-{hand_number}-showdown", board, holes
            )
            showdown = True
        elif state.phase is HandPhase.SETTLED:
            break
        else:
            raise AssertionError(f"random hand entered {state.phase.value}")
    else:
        raise AssertionError("random hand exceeded 200 state transitions")

    engine.log.verify()
    final = engine.snapshot()
    if final.total_units() != initial_total:
        raise AssertionError("random hand did not conserve total units")
    if any(player.stack_units < 0 for player in final.players.values()):
        raise AssertionError("random hand produced a negative stack")
    return final, actions, showdown


def run_random_hands(
    count: int = 10_000,
    seed: int = 20260721,
    rules: FixedLimitRules | None = None,
) -> RandomRunSummary:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    actions = 0
    showdowns = 0
    folds = 0
    for hand_number in range(count):
        state, hand_actions, showdown = play_random_hand(
            hand_number, rng, rules
        )
        actions += hand_actions
        showdowns += int(showdown)
        folds += int(not showdown)
        if state.phase is not HandPhase.SETTLED:
            raise AssertionError("random hand did not settle")
    return RandomRunSummary(count, actions, showdowns, folds, seed)
