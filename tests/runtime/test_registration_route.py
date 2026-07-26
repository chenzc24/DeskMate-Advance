from __future__ import annotations

from poker_dealer.domain import RobotPoseNode, Seat, TableRole
from poker_dealer.robotics.navigation import SimulatedNavigationAdapter
from poker_dealer.runtime import (
    RegistrationNavigationCoordinator,
    RegistrationRuntime,
)


def test_registration_route_visits_button_sb_bb_utg_then_normalizes() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    navigation = SimulatedNavigationAdapter(
        initial_pose=RobotPoseNode.INIT_TO_END
    )
    navigation.open()
    route = RegistrationNavigationCoordinator(runtime, navigation)

    expected = (
        (TableRole.BUTTON, RobotPoseNode.INIT_BUTTON),
        (TableRole.SMALL_BLIND, RobotPoseNode.END_SMALL_BLIND),
        (TableRole.BIG_BLIND, RobotPoseNode.END_BIG_BLIND),
        (TableRole.UNDER_THE_GUN, RobotPoseNode.INIT_UTG),
    )
    for role, pose in expected:
        runtime.select_role(role)
        ack = route.align_focus()
        assert ack is not None
        assert ack.actual_pose is pose

    ack = route.normalize_to_init()
    assert ack is not None
    assert ack.actual_pose is RobotPoseNode.INIT_TO_END


def test_registration_focus_alignment_is_idempotent_for_same_seat() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    navigation = SimulatedNavigationAdapter(
        initial_pose=RobotPoseNode.INIT_TO_END
    )
    navigation.open()
    route = RegistrationNavigationCoordinator(runtime, navigation)
    assert route.align_focus() is not None
    version = navigation.health().pose_version
    assert route.align_focus() is None
    assert navigation.health().pose_version == version
