"""Deterministic Texas Hold'em 5-to-7 card evaluator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Iterable, Mapping

from poker_dealer.domain import CardIdentity, Rank, Seat

from .pots import Pot, split_pot


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


RANK_VALUE = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


@dataclass(frozen=True, slots=True)
class HandRank:
    category: HandCategory
    comparison_key: tuple[int, ...]
    best_five: tuple[CardIdentity, ...]

    @property
    def key(self) -> tuple[int, ...]:
        return (int(self.category), *self.comparison_key)


def _straight_high(values: Iterable[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)
    ordered = sorted(unique)
    run = 1
    best: int | None = None
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            run += 1
            if run >= 5:
                best = current
        elif current != previous:
            run = 1
    return best


def evaluate_five(cards: Iterable[CardIdentity]) -> HandRank:
    selected = tuple(cards)
    if len(selected) != 5:
        raise ValueError("evaluate_five requires exactly five cards")
    if len(set(selected)) != 5:
        raise ValueError("duplicate cards are not allowed")

    values = [RANK_VALUE[card.rank] for card in selected]
    counts = Counter(values)
    grouped = sorted(
        ((count, value) for value, count in counts.items()), reverse=True
    )
    flush = len({card.suit for card in selected}) == 1
    straight_high = _straight_high(values)

    if flush and straight_high is not None:
        category = HandCategory.STRAIGHT_FLUSH
        key = (straight_high,)
    elif grouped[0][0] == 4:
        quad = grouped[0][1]
        kicker = max(value for value in values if value != quad)
        category = HandCategory.FOUR_OF_A_KIND
        key = (quad, kicker)
    elif sorted(counts.values()) == [2, 3]:
        triple = max(value for value, count in counts.items() if count == 3)
        pair = max(value for value, count in counts.items() if count == 2)
        category = HandCategory.FULL_HOUSE
        key = (triple, pair)
    elif flush:
        category = HandCategory.FLUSH
        key = tuple(sorted(values, reverse=True))
    elif straight_high is not None:
        category = HandCategory.STRAIGHT
        key = (straight_high,)
    elif grouped[0][0] == 3:
        triple = grouped[0][1]
        kickers = sorted((v for v in values if v != triple), reverse=True)
        category = HandCategory.THREE_OF_A_KIND
        key = (triple, *kickers)
    elif sorted(counts.values()) == [1, 2, 2]:
        pairs = sorted(
            (value for value, count in counts.items() if count == 2),
            reverse=True,
        )
        kicker = next(value for value, count in counts.items() if count == 1)
        category = HandCategory.TWO_PAIR
        key = (*pairs, kicker)
    elif 2 in counts.values():
        pair = max(value for value, count in counts.items() if count == 2)
        kickers = sorted((v for v in values if v != pair), reverse=True)
        category = HandCategory.ONE_PAIR
        key = (pair, *kickers)
    else:
        category = HandCategory.HIGH_CARD
        key = tuple(sorted(values, reverse=True))

    ordered_cards = tuple(
        sorted(selected, key=lambda card: RANK_VALUE[card.rank], reverse=True)
    )
    return HandRank(category, tuple(key), ordered_cards)


def evaluate_best(cards: Iterable[CardIdentity]) -> HandRank:
    selected = tuple(cards)
    if not 5 <= len(selected) <= 7:
        raise ValueError("evaluate_best requires five to seven cards")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate cards are not allowed")
    return max((evaluate_five(combo) for combo in combinations(selected, 5)), key=lambda r: r.key)


@dataclass(frozen=True, slots=True)
class ShowdownResult:
    awards: Mapping[Seat, int]
    ranks: Mapping[Seat, HandRank]
    winners_by_pot: Mapping[str, tuple[Seat, ...]]


def settle_showdown(
    pots: Iterable[Pot],
    board: Iterable[CardIdentity],
    hole_cards: Mapping[Seat, tuple[CardIdentity, CardIdentity]],
    button: Seat,
) -> ShowdownResult:
    board_cards = tuple(board)
    if len(board_cards) != 5:
        raise ValueError("showdown requires five board cards")
    all_cards = list(board_cards)
    for cards in hole_cards.values():
        if len(cards) != 2:
            raise ValueError("each live player requires two hole cards")
        all_cards.extend(cards)
    if len(set(all_cards)) != len(all_cards):
        raise ValueError("showdown contains duplicate cards")

    ranks = {
        seat: evaluate_best((*board_cards, *cards))
        for seat, cards in hole_cards.items()
    }
    awards = {seat: 0 for seat in hole_cards}
    winners_by_pot: dict[str, tuple[Seat, ...]] = {}
    for pot in pots:
        eligible = tuple(seat for seat in pot.eligible_seats if seat in ranks)
        if not eligible:
            raise ValueError(f"{pot.pot_id} has no ranked eligible player")
        best_key = max(ranks[seat].key for seat in eligible)
        winners = tuple(seat for seat in eligible if ranks[seat].key == best_key)
        winners_by_pot[pot.pot_id] = winners
        for seat, amount in split_pot(pot.amount_units, winners, button).items():
            awards[seat] = awards.get(seat, 0) + amount
    return ShowdownResult(awards, ranks, winners_by_pot)
