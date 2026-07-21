"""Pure digital-ledger pot construction and deterministic settlement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from poker_dealer.domain import SEAT_ORDER, Seat, clockwise_order_after


@dataclass(frozen=True, slots=True)
class Pot:
    pot_id: str
    amount_units: int
    eligible_seats: tuple[Seat, ...]

    def __post_init__(self) -> None:
        if not self.pot_id:
            raise ValueError("pot_id must not be empty")
        if self.amount_units <= 0:
            raise ValueError("pot amount must be positive")
        if not self.eligible_seats:
            raise ValueError("pot requires at least one eligible seat")


@dataclass(frozen=True, slots=True)
class PotBuildResult:
    pots: tuple[Pot, ...]
    returned_units: Mapping[Seat, int]

    @property
    def pot_units(self) -> int:
        return sum(pot.amount_units for pot in self.pots)


def build_pots(
    contributions: Mapping[Seat, int],
    folded_seats: set[Seat] | frozenset[Seat] = frozenset(),
) -> PotBuildResult:
    """Layer contributions; return an unmatched single-contributor excess."""

    normalized = {seat: int(contributions.get(seat, 0)) for seat in SEAT_ORDER}
    if any(amount < 0 for amount in normalized.values()):
        raise ValueError("contributions must be non-negative")
    if not folded_seats <= set(SEAT_ORDER):
        raise ValueError("folded_seats contains an unknown seat")

    levels = sorted({amount for amount in normalized.values() if amount > 0})
    previous = 0
    pots: list[Pot] = []
    returned = {seat: 0 for seat in SEAT_ORDER}

    for level in levels:
        contributors = tuple(
            seat for seat in SEAT_ORDER if normalized[seat] >= level
        )
        layer = (level - previous) * len(contributors)
        previous = level
        if layer == 0:
            continue
        if len(contributors) == 1:
            returned[contributors[0]] += layer
            continue
        eligible = tuple(seat for seat in contributors if seat not in folded_seats)
        if not eligible:
            raise ValueError("a pot layer has no eligible player")
        pot_id = "main" if not pots else f"side_{len(pots)}"
        pots.append(Pot(pot_id, layer, eligible))

    if sum(normalized.values()) != sum(p.amount_units for p in pots) + sum(
        returned.values()
    ):
        raise AssertionError("pot construction did not conserve contributions")
    return PotBuildResult(tuple(pots), returned)


def split_pot(
    amount_units: int,
    winners: tuple[Seat, ...],
    button: Seat,
) -> dict[Seat, int]:
    """Split a pot; odd units go clockwise to the first winners left of Button."""

    if amount_units <= 0:
        raise ValueError("amount_units must be positive")
    unique_winners = tuple(dict.fromkeys(winners))
    if not unique_winners:
        raise ValueError("at least one winner is required")
    base, remainder = divmod(amount_units, len(unique_winners))
    awards = {seat: base for seat in unique_winners}
    winner_set = set(unique_winners)
    odd_order = tuple(
        seat for seat in clockwise_order_after(button) if seat in winner_set
    )
    for seat in odd_order[:remainder]:
        awards[seat] += 1
    if sum(awards.values()) != amount_units:
        raise AssertionError("pot split did not conserve units")
    return awards


@dataclass(frozen=True, slots=True)
class OperatorAdjustment:
    seat: Seat
    amount_units: int
    operator_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ValueError("operator_id is required")
        if not self.reason.strip():
            raise ValueError("reason is required")
        if self.amount_units == 0:
            raise ValueError("adjustment must change the balance")


def apply_operator_adjustment(
    balances: Mapping[Seat, int], adjustment: OperatorAdjustment
) -> dict[Seat, int]:
    updated = {seat: int(balances.get(seat, 0)) for seat in SEAT_ORDER}
    result = updated[adjustment.seat] + adjustment.amount_units
    if result < 0:
        raise ValueError("operator adjustment cannot create a negative balance")
    updated[adjustment.seat] = result
    return updated
