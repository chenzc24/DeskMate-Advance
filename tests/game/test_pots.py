from __future__ import annotations

from itertools import product

import pytest

from poker_dealer.domain import Seat
from poker_dealer.game import (
    OperatorAdjustment,
    apply_operator_adjustment,
    build_pots,
    split_pot,
)


def test_three_layer_side_pots_and_eligibility() -> None:
    result = build_pots(
        {Seat.A: 10, Seat.B: 20, Seat.C: 40, Seat.D: 40}
    )
    assert [pot.amount_units for pot in result.pots] == [40, 30, 40]
    assert result.pots[0].eligible_seats == (Seat.A, Seat.B, Seat.C, Seat.D)
    assert result.pots[1].eligible_seats == (Seat.B, Seat.C, Seat.D)
    assert result.pots[2].eligible_seats == (Seat.C, Seat.D)
    assert result.pot_units == 110
    assert sum(result.returned_units.values()) == 0


def test_folded_money_stays_but_folded_seat_is_never_eligible() -> None:
    result = build_pots(
        {Seat.A: 10, Seat.B: 20, Seat.C: 40, Seat.D: 40}, {Seat.B}
    )
    assert [pot.amount_units for pot in result.pots] == [40, 30, 40]
    assert all(Seat.B not in pot.eligible_seats for pot in result.pots)


def test_unmatched_excess_is_returned_and_conserved() -> None:
    result = build_pots(
        {Seat.A: 100, Seat.B: 40, Seat.C: 40, Seat.D: 0}
    )
    assert [pot.amount_units for pot in result.pots] == [120]
    assert result.returned_units[Seat.A] == 60
    assert result.pot_units + sum(result.returned_units.values()) == 180


def test_odd_unit_goes_clockwise_left_of_button() -> None:
    assert split_pot(5, (Seat.B, Seat.D), Seat.A) == {Seat.B: 3, Seat.D: 2}


def test_operator_adjustment_requires_audit_and_no_negative_balance() -> None:
    balances = {Seat.A: 0, Seat.B: 80, Seat.C: 80, Seat.D: 80}
    updated = apply_operator_adjustment(
        balances, OperatorAdjustment(Seat.A, 80, "operator-1", "rebuy")
    )
    assert updated[Seat.A] == 80
    assert balances[Seat.A] == 0
    with pytest.raises(ValueError, match="operator_id"):
        OperatorAdjustment(Seat.A, 80, "", "rebuy")
    with pytest.raises(ValueError, match="negative"):
        apply_operator_adjustment(
            updated, OperatorAdjustment(Seat.A, -81, "operator-1", "correction")
        )


def test_exhaustive_small_contribution_layers_conserve_and_filter_folds() -> None:
    seats = (Seat.A, Seat.B, Seat.C, Seat.D)
    checked = 0
    for amounts in product(range(6), repeat=4):
        if sum(amounts) == 0:
            continue
        contributions = dict(zip(seats, amounts, strict=True))
        contributors = {seat for seat, amount in contributions.items() if amount}
        for folded_bits in product((False, True), repeat=4):
            folded = {
                seat
                for seat, is_folded in zip(seats, folded_bits, strict=True)
                if is_folded
            }
            if contributors <= folded:
                continue
            maximum = max(amounts)
            highest_contributors = {
                seat
                for seat, amount in contributions.items()
                if amount == maximum
            }
            if highest_contributors <= folded:
                continue
            result = build_pots(contributions, folded)
            assert result.pot_units + sum(result.returned_units.values()) == sum(
                amounts
            )
            assert all(pot.amount_units > 0 for pot in result.pots)
            assert all(
                not (set(pot.eligible_seats) & folded) for pot in result.pots
            )
            assert len({pot.pot_id for pot in result.pots}) == len(result.pots)
            checked += 1
    assert checked > 10_000


def test_unreachable_layer_with_no_eligible_player_is_rejected() -> None:
    with pytest.raises(ValueError, match="no eligible"):
        build_pots(
            {Seat.A: 0, Seat.B: 1, Seat.C: 2, Seat.D: 2},
            {Seat.C, Seat.D},
        )
