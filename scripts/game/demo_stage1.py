"""Run one visible four-player Stage 1 hand without camera or robot."""

from __future__ import annotations

import json

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
    HandEngine,
    SimulatedCardPerception,
    state_to_contract_snapshot,
)


def _card(rank: Rank, suit: Suit) -> CardIdentity:
    return CardIdentity(rank, suit)


def main() -> int:
    engine = HandEngine.start("stage1-demo", Seat.A)
    transitions: list[dict[str, object]] = []
    sequence = 0

    def record(trigger: str) -> None:
        state = engine.state
        transitions.append(
            {
                "trigger": trigger,
                "state_version": state.state_version,
                "phase": state.phase.value,
                "street": state.street.value if state.street else None,
                "acting_seat": (
                    state.acting_seat.value if state.acting_seat else None
                ),
                "legal_actions": [action.value for action in state.legal_actions],
                "pot_units": state.pot_units,
            }
        )

    def act(action: PlayerActionType) -> None:
        nonlocal sequence
        sequence += 1
        state = engine.state
        result = engine.apply_action(
            ActionRequest(
                f"demo-action-{sequence}",
                state.hand_id,
                state.state_version,
                state.acting_seat,  # type: ignore[arg-type]
                action,
            )
        )
        if not result.accepted:
            raise RuntimeError(result.reason)
        record(action.value)

    board = (
        _card(Rank.TWO, Suit.CLUBS),
        _card(Rank.SEVEN, Suit.DIAMONDS),
        _card(Rank.NINE, Suit.HEARTS),
        _card(Rank.JACK, Suit.SPADES),
        _card(Rank.KING, Suit.CLUBS),
    )
    holes = {
        Seat.A: (_card(Rank.ACE, Suit.DIAMONDS), _card(Rank.THREE, Suit.CLUBS)),
        Seat.B: (_card(Rank.QUEEN, Suit.DIAMONDS), _card(Rank.TEN, Suit.CLUBS)),
        Seat.C: (_card(Rank.NINE, Suit.SPADES), _card(Rank.FOUR, Suit.CLUBS)),
        Seat.D: (_card(Rank.KING, Suit.DIAMONDS), _card(Rank.FIVE, Suit.CLUBS)),
    }
    street_cards = {
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
    }
    card_simulator = SimulatedCardPerception()

    record("hand_started")
    for action in (
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CHECK,
    ):
        act(action)

    while engine.state.phase is not HandPhase.SHOWDOWN:
        if engine.state.phase is HandPhase.DEALING_BOARD:
            street = engine.state.street.value  # type: ignore[union-attr]
            for slot, card in street_cards[street]:
                sequence += 1
                observation = card_simulator.emit(
                    slot,
                    ObservationStatus.CONFIRMED,
                    card=card,
                    confidence=0.999,
                    observed_at_ns=sequence,
                )
                if not engine.apply_card_observation(observation).accepted:
                    raise RuntimeError("simulated card confirmation failed")
            engine.confirm_board_dealt(f"demo-board-{street}")
            record(f"{street}_confirmed")
        elif engine.state.phase is HandPhase.AWAITING_ACTION:
            act(PlayerActionType.CHECK)
        else:
            raise RuntimeError(f"unexpected phase: {engine.state.phase.value}")

    engine.settle_showdown("demo-showdown", board, holes)
    record("showdown_settled")
    snapshot = state_to_contract_snapshot(engine.state)
    print(
        json.dumps(
            {
                "betting_product_status": engine.rules.product_status,
                "transitions": transitions,
                "final_snapshot": snapshot,
                "awards": {
                    seat.value: amount
                    for seat, amount in engine.state.awards.items()
                },
                "event_count": len(engine.log.events),
                "event_log_tail_hash": engine.log.events[-1].event_hash,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
