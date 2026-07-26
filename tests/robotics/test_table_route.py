from __future__ import annotations

import pytest

from poker_dealer.domain import RobotPoseNode, Seat
from poker_dealer.robotics.navigation import (
    CarApiAction,
    TableRoutePlanner,
    TurnDirection,
    hole_deal_pose,
    player_pose,
)


def test_role_and_hole_deal_poses_follow_button_mapping() -> None:
    assert player_pose(Seat.A, Seat.A) is RobotPoseNode.INIT_BUTTON
    assert player_pose(Seat.A, Seat.B) is RobotPoseNode.END_SMALL_BLIND
    assert player_pose(Seat.A, Seat.C) is RobotPoseNode.END_BIG_BLIND
    assert player_pose(Seat.A, Seat.D) is RobotPoseNode.INIT_UTG
    assert hole_deal_pose(Seat.A, Seat.A) is RobotPoseNode.INIT_TO_END
    assert hole_deal_pose(Seat.A, Seat.B) is RobotPoseNode.END_TO_END
    assert hole_deal_pose(Seat.A, Seat.C) is RobotPoseNode.END_TO_INIT
    assert hole_deal_pose(Seat.A, Seat.D) is RobotPoseNode.INIT_TO_INIT


def test_route_uses_only_allowed_right_180_shortcuts() -> None:
    planner = TableRoutePlanner()
    sb_to_bb = planner.plan(
        RobotPoseNode.END_SMALL_BLIND, RobotPoseNode.END_BIG_BLIND
    )
    assert len(sb_to_bb) == 1
    assert sb_to_bb[0].action is CarApiAction.PRESET_TURN
    assert sb_to_bb[0].direction is TurnDirection.RIGHT
    assert sb_to_bb[0].degrees == 180

    utg_to_button = planner.plan(
        RobotPoseNode.INIT_UTG, RobotPoseNode.INIT_BUTTON
    )
    assert len(utg_to_button) == 1
    assert utg_to_button[0].action is CarApiAction.PRESET_TURN
    assert utg_to_button[0].direction is TurnDirection.RIGHT
    assert utg_to_button[0].degrees == 180


def test_postflop_board_to_small_blind_route_continues_to_end() -> None:
    route = TableRoutePlanner().plan(
        RobotPoseNode.BOARD_TO_END, RobotPoseNode.END_SMALL_BLIND
    )
    assert [item.action for item in route] == [
        CarApiAction.FOLLOW_LINE_TO_END,
        CarApiAction.FACE_TURN,
    ]
    assert route[-1].direction is TurnDirection.LEFT


def test_hole_route_turns_right_twice_after_small_blind_delivery() -> None:
    route = TableRoutePlanner().plan(
        RobotPoseNode.END_TO_END, RobotPoseNode.END_TO_INIT
    )
    assert len(route) == 1
    assert route[0].action is CarApiAction.PRESET_TURN
    assert route[0].direction is TurnDirection.RIGHT
    assert route[0].degrees == 180


@pytest.mark.parametrize(
    "target",
    (
        RobotPoseNode.INIT_BUTTON,
        RobotPoseNode.INIT_UTG,
        RobotPoseNode.END_SMALL_BLIND,
        RobotPoseNode.END_BIG_BLIND,
    ),
)
def test_every_player_target_is_reachable_from_initial_pose(
    target: RobotPoseNode,
) -> None:
    assert TableRoutePlanner().plan(RobotPoseNode.INIT_TO_END, target)


def test_board_route_is_explicitly_represented_for_fail_closed_api_mapping() -> None:
    route = TableRoutePlanner().plan(
        RobotPoseNode.INIT_TO_END, RobotPoseNode.BOARD_TO_END
    )
    assert len(route) == 1
    assert route[0].action is CarApiAction.FOLLOW_LINE_TO_BOARD
