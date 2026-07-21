from __future__ import annotations

import pytest

from poker_dealer.domain import CardIdentity, Rank, Seat, Suit
from poker_dealer.game import (
    HandCategory,
    Pot,
    evaluate_best,
    evaluate_five,
    settle_showdown,
)


RANKS = {rank.value: rank for rank in Rank}
SUITS = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}


def cards(spec: str) -> tuple[CardIdentity, ...]:
    return tuple(
        CardIdentity(RANKS[token[:-1]], SUITS[token[-1]])
        for token in spec.split()
    )


@pytest.mark.parametrize(
    ("spec", "category"),
    [
        ("As Ks Qs Js Ts", HandCategory.STRAIGHT_FLUSH),
        ("9s 9h 9d 9c As", HandCategory.FOUR_OF_A_KIND),
        ("Ks Kh Kd 2c 2d", HandCategory.FULL_HOUSE),
        ("As Js 8s 4s 2s", HandCategory.FLUSH),
        ("9s 8h 7d 6c 5s", HandCategory.STRAIGHT),
        ("7s 7h 7d Ac Ks", HandCategory.THREE_OF_A_KIND),
        ("As Ah Kd Kc 2s", HandCategory.TWO_PAIR),
        ("Qs Qh Ad Kc 2s", HandCategory.ONE_PAIR),
        ("As Kh 9d 5c 2s", HandCategory.HIGH_CARD),
    ],
)
def test_all_hand_categories(spec: str, category: HandCategory) -> None:
    assert evaluate_five(cards(spec)).category is category


def test_seven_card_best_hand_wheel_and_kickers() -> None:
    royal = evaluate_best(cards("As Ks Qs Js Ts 2d 3c"))
    assert royal.category is HandCategory.STRAIGHT_FLUSH
    wheel = evaluate_five(cards("As 2h 3d 4c 5s"))
    assert wheel.category is HandCategory.STRAIGHT
    assert wheel.comparison_key == (5,)
    better_pair = evaluate_five(cards("Qs Qh Ad Kc 3s"))
    worse_pair = evaluate_five(cards("Qd Qc Ah Ks 2d"))
    assert better_pair.key > worse_pair.key


def test_board_play_tie_and_per_pot_settlement() -> None:
    board = cards("2c 3d 4h 5s 6c")
    holes = {
        Seat.A: cards("Ah Kd"),
        Seat.B: cards("As Qd"),
    }
    result = settle_showdown((Pot("main", 5, (Seat.A, Seat.B)),), board, holes, Seat.D)
    assert result.winners_by_pot["main"] == (Seat.A, Seat.B)
    assert result.awards == {Seat.A: 3, Seat.B: 2}


def test_duplicate_or_wrong_card_count_is_rejected() -> None:
    duplicate = cards("As As Qs Js Ts")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_five(duplicate)
    with pytest.raises(ValueError, match="five to seven"):
        evaluate_best(cards("As Ks Qs Js"))
