"""Safe navigation adapters; only the simulator is enabled here."""

from __future__ import annotations

import time

from poker_dealer.domain import (
    DealerTargetSlot,
    NavigationAck,
    NavigationAckStatus,
    NavigationCommand,
    NavigationErrorCode,
    RobotPoseNode,
)

from .port import NavigationHealth, NavigationUnavailableError


_LINE_POSES = {
    RobotPoseNode.INIT_TO_END,
    RobotPoseNode.INIT_TO_INIT,
    RobotPoseNode.BOARD_TO_END,
    RobotPoseNode.END_TO_END,
    RobotPoseNode.END_TO_INIT,
}
_ENDPOINT_POSES = {
    RobotPoseNode.INIT_TO_END,
    RobotPoseNode.INIT_TO_INIT,
    RobotPoseNode.INIT_BUTTON,
    RobotPoseNode.INIT_UTG,
    RobotPoseNode.END_TO_END,
    RobotPoseNode.END_TO_INIT,
    RobotPoseNode.END_SMALL_BLIND,
    RobotPoseNode.END_BIG_BLIND,
}
_DEFAULT_TARGET_POSES = {
    DealerTargetSlot.SEAT_A: RobotPoseNode.INIT_BUTTON,
    DealerTargetSlot.SEAT_B: RobotPoseNode.END_SMALL_BLIND,
    DealerTargetSlot.SEAT_C: RobotPoseNode.END_BIG_BLIND,
    DealerTargetSlot.SEAT_D: RobotPoseNode.INIT_UTG,
    DealerTargetSlot.BOARD_FLOP_1: RobotPoseNode.BOARD_TO_END,
    DealerTargetSlot.BOARD_FLOP_2: RobotPoseNode.BOARD_TO_END,
    DealerTargetSlot.BOARD_FLOP_3: RobotPoseNode.BOARD_TO_END,
    DealerTargetSlot.BOARD_TURN: RobotPoseNode.BOARD_TO_END,
    DealerTargetSlot.BOARD_RIVER: RobotPoseNode.BOARD_TO_END,
}


class SimulatedNavigationAdapter:
    """Deterministic no-motion adapter for state-machine and replay tests."""

    physical_motion = False

    def __init__(
        self,
        device_id: str = "simulated-navigation",
        *,
        initial_pose: RobotPoseNode = RobotPoseNode.INIT_TO_END,
        initial_pose_version: int = 0,
        target_poses: dict[DealerTargetSlot, RobotPoseNode] | None = None,
    ) -> None:
        if initial_pose is RobotPoseNode.UNKNOWN:
            raise ValueError("simulated navigation requires a known initial pose")
        if initial_pose_version < 0:
            raise ValueError("initial_pose_version must be non-negative")
        self._device_id = device_id
        self._pose = initial_pose
        self._pose_version = initial_pose_version
        self._opened = False
        self._acks: dict[str, NavigationAck] = {}
        self._target_poses = dict(target_poses or _DEFAULT_TARGET_POSES)

    @property
    def device_id(self) -> str:
        return self._device_id

    def open(self) -> None:
        self._opened = True

    def execute(
        self, command: NavigationCommand, observed_at_ns: int | None = None
    ) -> NavigationAck:
        if not self._opened:
            raise NavigationUnavailableError("simulated navigation is not open")
        if command.command_id in self._acks:
            return self._acks[command.command_id]
        observed = time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        if (
            command.expected_pose_version != self._pose_version
            or command.start_pose is not self._pose
        ):
            ack = NavigationAck(
                command_id=command.command_id,
                session_id=command.session_id,
                hand_id=command.hand_id,
                expected_state_version=command.expected_state_version,
                action=command.action,
                target_slot=command.target_slot,
                status=NavigationAckStatus.REJECTED,
                observed_at_ns=observed,
                actual_pose=self._pose,
                pose_version=self._pose_version,
                pose_confidence=1.0,
                line_locked=self._pose in _LINE_POSES,
                endpoint_confirmed=self._pose in _ENDPOINT_POSES,
                target_aligned=False,
                stable_frames=1,
                error_code=NavigationErrorCode.POSE_UNKNOWN,
                reason="expected pose/version does not match simulator pose",
            )
            self._acks[command.command_id] = ack
            return ack
        target_pose = command.target_pose
        if target_pose is RobotPoseNode.UNKNOWN and command.target_slot is not None:
            target_pose = self._target_poses.get(
                command.target_slot, RobotPoseNode.UNKNOWN
            )
        if target_pose is RobotPoseNode.UNKNOWN:
            ack = NavigationAck(
                command_id=command.command_id,
                session_id=command.session_id,
                hand_id=command.hand_id,
                expected_state_version=command.expected_state_version,
                action=command.action,
                target_slot=command.target_slot,
                status=NavigationAckStatus.REJECTED,
                observed_at_ns=observed,
                actual_pose=self._pose,
                pose_version=self._pose_version,
                pose_confidence=1.0,
                line_locked=self._pose in _LINE_POSES,
                endpoint_confirmed=self._pose in _ENDPOINT_POSES,
                target_aligned=False,
                stable_frames=1,
                error_code=NavigationErrorCode.TARGET_NOT_FOUND,
                reason="target slot has no configured simulator pose",
            )
            self._acks[command.command_id] = ack
            return ack
        self._pose = target_pose
        self._pose_version += 1
        ack = NavigationAck(
            command_id=command.command_id,
            session_id=command.session_id,
            hand_id=command.hand_id,
            expected_state_version=command.expected_state_version,
            action=command.action,
            target_slot=command.target_slot,
            status=NavigationAckStatus.SUCCEEDED,
            observed_at_ns=observed,
            actual_pose=self._pose,
            pose_version=self._pose_version,
            pose_confidence=1.0,
            line_locked=self._pose in _LINE_POSES,
            endpoint_confirmed=self._pose in _ENDPOINT_POSES,
            target_aligned=True,
            stable_frames=3,
            face_center_error_px=0.0,
        )
        self._acks[command.command_id] = ack
        return ack

    def health(self) -> NavigationHealth:
        return NavigationHealth(
            device_id=self.device_id,
            available=True,
            opened=self._opened,
            physical_motion=False,
            pose=self._pose,
            pose_version=self._pose_version,
        )

    def close(self) -> None:
        self._opened = False


class UnavailableNavigationAdapter:
    """Fail-closed placeholder for future Raspberry Pi/MCU navigation."""

    physical_motion = True

    def __init__(self, device_id: str, reason: str) -> None:
        self._device_id = device_id
        self._reason = reason

    @property
    def device_id(self) -> str:
        return self._device_id

    def open(self) -> None:
        raise NavigationUnavailableError(self._reason)

    def execute(
        self, command: NavigationCommand, observed_at_ns: int | None = None
    ) -> NavigationAck:
        del command, observed_at_ns
        raise NavigationUnavailableError(self._reason)

    def health(self) -> NavigationHealth:
        return NavigationHealth(
            device_id=self.device_id,
            available=False,
            opened=False,
            physical_motion=True,
            pose=RobotPoseNode.UNKNOWN,
            pose_version=0,
            reason=self._reason,
        )

    def close(self) -> None:
        return None


__all__ = ["SimulatedNavigationAdapter", "UnavailableNavigationAdapter"]
