from __future__ import annotations

import pytest

from poker_dealer.domain import (
    ChipAmountScope,
    ChipCount,
    ChipObservation,
    ChipObservationStatus,
    DealerTargetSlot,
    FaceEnrollmentObservation,
    FaceEnrollmentStatus,
    NavigationAck,
    NavigationAckStatus,
    NavigationAction,
    NavigationCommand,
    RobotPoseNode,
    Seat,
)


def test_navigation_contract_carries_game_pose_and_target_versions() -> None:
    command = NavigationCommand(
        command_id="nav-1",
        session_id="session",
        hand_id="hand",
        expected_state_version=7,
        expected_pose_version=3,
        issued_at_ns=10,
        action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
        start_pose=RobotPoseNode.INIT_TO_END,
        target_pose=RobotPoseNode.UNKNOWN,
        target_slot=DealerTargetSlot.SEAT_B,
    )
    assert command.inter_motion_delay_ms == 2500
    ack = NavigationAck(
        command_id=command.command_id,
        session_id=command.session_id,
        hand_id=command.hand_id,
        expected_state_version=command.expected_state_version,
        action=command.action,
        target_slot=command.target_slot,
        status=NavigationAckStatus.SUCCEEDED,
        observed_at_ns=20,
        actual_pose=RobotPoseNode.END_SMALL_BLIND,
        pose_version=4,
        pose_confidence=0.99,
        line_locked=False,
        endpoint_confirmed=True,
        target_aligned=True,
        stable_frames=5,
        face_center_error_px=2.0,
    )
    assert ack.target_slot is DealerTargetSlot.SEAT_B
    assert ack.pose_version > command.expected_pose_version


def test_navigation_command_rejects_negative_inter_motion_delay() -> None:
    with pytest.raises(ValueError, match="inter_motion_delay_ms"):
        NavigationCommand(
            command_id="nav-negative-delay",
            session_id="session",
            hand_id="hand",
            expected_state_version=0,
            expected_pose_version=0,
            issued_at_ns=0,
            action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
            start_pose=RobotPoseNode.INIT_TO_END,
            target_pose=RobotPoseNode.UNKNOWN,
            target_slot=DealerTargetSlot.SEAT_A,
            inter_motion_delay_ms=-1,
        )


def test_successful_target_navigation_requires_alignment() -> None:
    with pytest.raises(ValueError, match="requires target and alignment"):
        NavigationAck(
            command_id="nav-1",
            session_id="session",
            hand_id="hand",
            expected_state_version=7,
            action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
            target_slot=DealerTargetSlot.SEAT_A,
            status=NavigationAckStatus.SUCCEEDED,
            observed_at_ns=20,
            actual_pose=RobotPoseNode.INIT_BUTTON,
            pose_version=4,
            pose_confidence=0.99,
            line_locked=False,
            endpoint_confirmed=True,
            target_aligned=False,
            stable_frames=5,
        )


def test_confirmed_chip_counts_must_equal_total() -> None:
    observation = ChipObservation(
        observation_id="chips-1",
        hand_id="hand",
        expected_state_version=4,
        focus_seat=Seat.D,
        observed_at_ns=50,
        status=ChipObservationStatus.CONFIRMED,
        amount_scope=ChipAmountScope.NEW_CONTRIBUTION,
        chip_counts=(ChipCount(2, 2),),
        total_units=4,
        confidence=0.98,
        stable_frames=5,
        model_version="chip@test",
        calibration_version="table@test",
    )
    assert observation.total_units == 4

    with pytest.raises(ValueError, match="sum to total_units"):
        ChipObservation(
            observation_id="chips-2",
            hand_id="hand",
            expected_state_version=4,
            focus_seat=Seat.D,
            observed_at_ns=50,
            status=ChipObservationStatus.CONFIRMED,
            amount_scope=ChipAmountScope.NEW_CONTRIBUTION,
            chip_counts=(ChipCount(2, 1),),
            total_units=4,
            confidence=0.98,
            stable_frames=5,
            model_version="chip@test",
            calibration_version="table@test",
        )


def test_face_enrollment_contract_does_not_carry_biometrics() -> None:
    observation = FaceEnrollmentObservation(
        observation_id="face-enrollment-1",
        session_id="session",
        expected_roster_version=0,
        focus_seat=Seat.A,
        observed_at_ns=50,
        status=FaceEnrollmentStatus.CONFIRMED,
        sample_count=8,
        stable_frames=8,
        model_version="face@test",
        quality_flags=("consent_confirmed",),
    )

    assert observation.sample_count == 8
    assert not hasattr(observation, "embedding")
    assert not hasattr(observation, "image")
