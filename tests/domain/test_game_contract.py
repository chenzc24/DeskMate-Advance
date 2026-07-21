from __future__ import annotations

import pytest

from poker_dealer.domain import (
    SEAT_ORDER,
    DealerTargetSlot,
    Seat,
    Street,
    big_blind_seat,
    board_deal_targets,
    clockwise_order_after,
    first_to_act,
    hole_deal_targets,
    next_button,
    small_blind_seat,
)


@pytest.mark.parametrize("button", list(Seat))
def test_four_player_roles_and_hole_deal_order(button: Seat) -> None:
    clockwise = clockwise_order_after(button)
    assert small_blind_seat(button) is clockwise[0]
    assert big_blind_seat(button) is clockwise[1]
    assert next_button(button) is clockwise[0]
    expected_round = tuple(DealerTargetSlot(seat.value) for seat in clockwise)
    assert hole_deal_targets(button) == expected_round + expected_round
    assert hole_deal_targets(button)[3] is DealerTargetSlot(button.value)
    assert hole_deal_targets(button)[7] is DealerTargetSlot(button.value)


@pytest.mark.parametrize("button", list(Seat))
def test_four_player_preflop_and_postflop_action_order(button: Seat) -> None:
    clockwise = clockwise_order_after(button)
    assert first_to_act(button, Street.PREFLOP) is clockwise[2]
    assert first_to_act(button, Street.FLOP) is clockwise[0]
    assert first_to_act(button, Street.TURN) is clockwise[0]
    assert first_to_act(button, Street.RIVER) is clockwise[0]
    with pytest.raises(ValueError, match="no betting actor"):
        first_to_act(button, Street.SHOWDOWN)


def test_action_order_skips_folded_and_all_in_seats() -> None:
    # Button A -> SB B -> BB C -> UTG D. D and A cannot act, so B is next.
    assert first_to_act(
        Seat.A,
        Street.PREFLOP,
        actionable_seats=(Seat.B, Seat.C),
    ) is Seat.B
    # Post-flop starts left of Button; skip B and C, so D acts first.
    assert first_to_act(
        Seat.A,
        Street.FLOP,
        actionable_seats=(Seat.A, Seat.D),
    ) is Seat.D


def test_every_seat_is_present_exactly_once_per_clockwise_round() -> None:
    for button in SEAT_ORDER:
        order = clockwise_order_after(button)
        assert len(order) == 4
        assert set(order) == set(SEAT_ORDER)
        assert order[-1] is button


def test_board_deal_includes_one_burn_before_each_street() -> None:
    assert board_deal_targets(Street.FLOP) == (
        DealerTargetSlot.BURN_TRAY,
        DealerTargetSlot.BOARD_FLOP_1,
        DealerTargetSlot.BOARD_FLOP_2,
        DealerTargetSlot.BOARD_FLOP_3,
    )
    assert board_deal_targets(Street.TURN) == (
        DealerTargetSlot.BURN_TRAY,
        DealerTargetSlot.BOARD_TURN,
    )
    assert board_deal_targets(Street.RIVER) == (
        DealerTargetSlot.BURN_TRAY,
        DealerTargetSlot.BOARD_RIVER,
    )
