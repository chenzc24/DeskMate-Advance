"""Navigation gate for the fixed Button→SB→BB→UTG enrollment route."""

from __future__ import annotations

import time

from poker_dealer.domain import (
    DealerTargetSlot,
    NavigationAckStatus,
    NavigationAction,
    NavigationCommand,
    RobotPoseNode,
)
from poker_dealer.robotics.navigation import NavigationPort, player_pose

from .registration import RegistrationRuntime


class RegistrationNavigationCoordinator:
    """Issue correlated navigation commands around the registration runtime."""

    def __init__(
        self,
        runtime: RegistrationRuntime,
        port: NavigationPort,
        *,
        command_timeout_ms: int = 20_000,
        inter_motion_delay_ms: int = 2500,
    ) -> None:
        if command_timeout_ms <= 0 or inter_motion_delay_ms < 0:
            raise ValueError("registration navigation timing is invalid")
        self.runtime = runtime
        self.port = port
        self.command_timeout_ms = command_timeout_ms
        self.inter_motion_delay_ms = inter_motion_delay_ms
        self._sequence = 0
        self._aligned_seat = None

    @property
    def aligned_seat(self):
        return self._aligned_seat

    def align_focus(self, observed_at_ns: int | None = None):
        """Move to and face the current role before enrollment can begin."""

        seat = self.runtime.focus_seat
        if self._aligned_seat is seat:
            return None
        target = player_pose(self.runtime.button, seat)
        command = self._command(
            action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
            target_pose=target,
            target_slot=DealerTargetSlot(seat.value),
            observed_at_ns=observed_at_ns,
            label=f"align:{self.runtime.focus_role.value}",
        )
        ack = self.port.execute(command)
        self._require_success(ack, command)
        self._aligned_seat = seat
        return ack

    def normalize_to_init(self, observed_at_ns: int | None = None):
        """Return from the final UTG enrollment to I_E before dealing starts."""

        health = self.port.health()
        if health.pose is RobotPoseNode.INIT_TO_END:
            return None
        command = self._command(
            action=NavigationAction.RETURN_TO_LINE,
            target_pose=RobotPoseNode.INIT_TO_END,
            target_slot=None,
            observed_at_ns=observed_at_ns,
            label="normalize:init_to_end",
        )
        ack = self.port.execute(command)
        self._require_success(ack, command)
        self._aligned_seat = None
        return ack

    def invalidate_alignment(self) -> None:
        self._aligned_seat = None

    def _command(
        self,
        *,
        action: NavigationAction,
        target_pose: RobotPoseNode,
        target_slot: DealerTargetSlot | None,
        observed_at_ns: int | None,
        label: str,
    ) -> NavigationCommand:
        health = self.port.health()
        if not health.available or not health.opened:
            raise RuntimeError(health.reason or "registration navigation is unavailable")
        if health.pose is RobotPoseNode.UNKNOWN:
            raise RuntimeError("registration navigation pose is unknown")
        self._sequence += 1
        issued_at_ns = (
            time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        )
        return NavigationCommand(
            command_id=(
                f"registration:{self.runtime.session_id}:"
                f"{self.runtime.roster_version}:{self._sequence}:{label}"
            ),
            session_id=self.runtime.session_id,
            hand_id="registration",
            expected_state_version=self.runtime.roster_version,
            expected_pose_version=health.pose_version,
            issued_at_ns=issued_at_ns,
            action=action,
            start_pose=health.pose,
            target_pose=target_pose,
            target_slot=target_slot,
            timeout_ms=self.command_timeout_ms,
            inter_motion_delay_ms=self.inter_motion_delay_ms,
        )

    @staticmethod
    def _require_success(ack, command: NavigationCommand) -> None:
        if (
            ack.command_id != command.command_id
            or ack.session_id != command.session_id
            or ack.hand_id != command.hand_id
            or ack.expected_state_version != command.expected_state_version
            or ack.status is not NavigationAckStatus.SUCCEEDED
            or ack.actual_pose is not command.target_pose
            or not ack.target_aligned
            or ack.pose_version <= command.expected_pose_version
        ):
            raise RuntimeError(
                f"registration navigation failed: {ack.status.value}:"
                f"{ack.reason or 'invalid acknowledgement'}"
            )


__all__ = ["RegistrationNavigationCoordinator"]
