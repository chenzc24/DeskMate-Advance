"""Stage 0 four-player game vocabulary and ordering contracts.

This module freezes table position and dealing order.  It deliberately contains
no betting reducer, side-pot builder or hand evaluator; those belong to Stage 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from .dealer import DealerTargetSlot


class Seat(StrEnum):
    A = "seat_a"
    B = "seat_b"
    C = "seat_c"
    D = "seat_d"


SEAT_ORDER: tuple[Seat, ...] = (Seat.A, Seat.B, Seat.C, Seat.D)


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class HandPhase(StrEnum):
    SETUP = "setup"
    POSTING_BLINDS = "posting_blinds"
    DEALING_HOLE = "dealing_hole"
    AWAITING_ACTION = "awaiting_action"
    DEALING_BOARD = "dealing_board"
    SHOWDOWN = "showdown"
    SETTLED = "settled"
    PAUSED_RECOVERY = "paused_recovery"
    VOIDED = "voided"


class PlayerActionType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class TableRole(StrEnum):
    """Current-hand player roles; unlike seats, these rotate with Button."""

    BUTTON = "button"
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"
    UNDER_THE_GUN = "under_the_gun"


def _validated_seats(seats: Iterable[Seat]) -> tuple[Seat, ...]:
    selected = tuple(dict.fromkeys(seats))
    if not selected:
        raise ValueError("at least one seat is required")
    if any(seat not in SEAT_ORDER for seat in selected):
        raise ValueError("unknown seat")
    return selected


def clockwise_order_after(
    anchor: Seat,
    included_seats: Iterable[Seat] = SEAT_ORDER,
) -> tuple[Seat, ...]:
    """Return included seats clockwise after anchor, with anchor last if included."""

    included = set(_validated_seats(included_seats))
    start = SEAT_ORDER.index(anchor)
    rotated = SEAT_ORDER[start + 1 :] + SEAT_ORDER[: start + 1]
    return tuple(seat for seat in rotated if seat in included)


def next_button(button: Seat, active_seats: Iterable[Seat] = SEAT_ORDER) -> Seat:
    return clockwise_order_after(button, active_seats)[0]


def small_blind_seat(
    button: Seat, active_seats: Iterable[Seat] = SEAT_ORDER
) -> Seat:
    return clockwise_order_after(button, active_seats)[0]


def big_blind_seat(
    button: Seat, active_seats: Iterable[Seat] = SEAT_ORDER
) -> Seat:
    active = _validated_seats(active_seats)
    small_blind = small_blind_seat(button, active)
    return clockwise_order_after(small_blind, active)[0]


def role_seats(button: Seat) -> dict[TableRole, Seat]:
    """Resolve the four current-hand roles onto fixed physical seats."""

    small_blind = small_blind_seat(button)
    big_blind = big_blind_seat(button)
    under_the_gun = clockwise_order_after(big_blind)[0]
    return {
        TableRole.BUTTON: button,
        TableRole.SMALL_BLIND: small_blind,
        TableRole.BIG_BLIND: big_blind,
        TableRole.UNDER_THE_GUN: under_the_gun,
    }


def role_for_seat(button: Seat, seat: Seat) -> TableRole:
    """Return the public role label for one internal physical seat."""

    return next(role for role, assigned in role_seats(button).items() if assigned is seat)


def _dealer_target(seat: Seat) -> DealerTargetSlot:
    return DealerTargetSlot(seat.value)


def hole_deal_targets(
    button: Seat, active_seats: Iterable[Seat] = SEAT_ORDER
) -> tuple[DealerTargetSlot, ...]:
    """Deal two clockwise rounds starting left of Button; Button receives last."""

    active = _validated_seats(active_seats)
    one_round = clockwise_order_after(button, active)
    targets = tuple(_dealer_target(seat) for seat in one_round)
    return targets + targets


def first_to_act(
    button: Seat,
    street: Street,
    actionable_seats: Iterable[Seat] = SEAT_ORDER,
    active_seats: Iterable[Seat] = SEAT_ORDER,
) -> Seat:
    """Resolve first non-folded, non-all-in actor for a betting street."""

    actionable = _validated_seats(actionable_seats)
    active = _validated_seats(active_seats)
    if not set(actionable) <= set(active):
        raise ValueError("actionable seats must be active")
    if street is Street.PREFLOP:
        anchor = big_blind_seat(button, active)
    elif street in {Street.FLOP, Street.TURN, Street.RIVER}:
        anchor = button
    else:
        raise ValueError("showdown has no betting actor")
    return clockwise_order_after(anchor, actionable)[0]


def board_deal_targets(street: Street) -> tuple[DealerTargetSlot, ...]:
    if street is Street.FLOP:
        return (
            DealerTargetSlot.BOARD_FLOP_1,
            DealerTargetSlot.BOARD_FLOP_2,
            DealerTargetSlot.BOARD_FLOP_3,
        )
    if street is Street.TURN:
        return (DealerTargetSlot.BOARD_TURN,)
    if street is Street.RIVER:
        return (DealerTargetSlot.BOARD_RIVER,)
    raise ValueError(f"street {street.value} has no board deal")
