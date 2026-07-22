from __future__ import annotations

from poker_dealer.domain import (
    ControlIntent,
    ControlSource,
    LaptopControlAdapter,
    RobotButtonAdapter,
    Seat,
    TableRole,
    role_for_seat,
    role_seats,
)


def test_roles_rotate_while_physical_seats_stay_fixed() -> None:
    first = role_seats(Seat.A)
    assert first == {
        TableRole.BUTTON: Seat.A,
        TableRole.SMALL_BLIND: Seat.B,
        TableRole.BIG_BLIND: Seat.C,
        TableRole.UNDER_THE_GUN: Seat.D,
    }
    second = role_seats(Seat.B)
    assert second[TableRole.BUTTON] is Seat.B
    assert second[TableRole.UNDER_THE_GUN] is Seat.A
    assert role_for_seat(Seat.B, Seat.D) is TableRole.BIG_BLIND


def test_laptop_and_robot_controls_share_semantic_contract() -> None:
    laptop = LaptopControlAdapter().process_key(ord("e"), 100)
    robot = RobotButtonAdapter().process_press(
        event_id="robot:1",
        button_id="main",
        observed_at_ns=100,
        device_state_version=7,
    )
    assert laptop is not None
    assert laptop.intent is robot.intent is ControlIntent.CONFIRM
    assert laptop.source is ControlSource.LAPTOP_KEYBOARD
    assert robot.source is ControlSource.ROBOT_BUTTON


def test_robot_long_press_is_cancel() -> None:
    result = RobotButtonAdapter().process_press(
        event_id="robot:2",
        button_id="main",
        observed_at_ns=200,
        device_state_version=8,
        long_press=True,
    )
    assert result.intent is ControlIntent.CANCEL
