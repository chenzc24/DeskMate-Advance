"""Deterministic gates for the sequential Stage 2A player-action loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import (
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerTargetSlot,
    HandPhase,
    PlayerActionObservation,
    Seat,
)
from poker_dealer.game import ActionResult, HandEngine
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState


class PartAPhase(StrEnum):
    WAITING_ROTATION_ACK = "waiting_rotation_ack"
    VERIFYING_IDENTITY = "verifying_identity"
    WAITING_PLAYER_ACTION = "waiting_player_action"
    ROUND_COMPLETE = "round_complete"
    RECOVERY_REQUIRED = "recovery_required"


@dataclass(frozen=True, slots=True)
class CoordinatorActionOutcome:
    accepted: bool
    reason: str
    game_result: ActionResult | None
    next_seat: Seat | None


class SequentialPartACoordinator:
    """Gate action evidence behind rotation ACK and session identity."""

    def __init__(self, engine: HandEngine, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        if (
            engine.state.phase is not HandPhase.AWAITING_ACTION
            or engine.state.acting_seat is None
        ):
            raise ValueError("coordinator requires an awaiting-action game state")
        self.engine = engine
        self.session_id = session_id
        self.phase = PartAPhase.WAITING_ROTATION_ACK
        self.pending_rotation: DealerCommand | None = None
        self.verified_player_id: str | None = None
        self.last_reason = "rotation_not_requested"
        self._command_sequence = 0

    @property
    def focus_seat(self) -> Seat | None:
        return self.engine.state.acting_seat

    def request_rotation(self, issued_at_ns: int) -> DealerCommand:
        if self.phase is not PartAPhase.WAITING_ROTATION_ACK:
            raise ValueError("rotation can only be requested at a turn boundary")
        if self.pending_rotation is not None:
            return self.pending_rotation
        seat = self.focus_seat
        if seat is None:
            raise ValueError("cannot rotate without an acting seat")
        self._command_sequence += 1
        command = DealerCommand(
            command_id=(
                f"part-a:{self.engine.state.hand_id}:{self.engine.state.state_version}:"
                f"rotate:{self._command_sequence}"
            ),
            issued_at_ns=issued_at_ns,
            command=DealerCommandType.ROTATE_TO,
            target_slot=DealerTargetSlot(seat.value),
        )
        self.pending_rotation = command
        self.last_reason = "waiting_for_matching_rotation_ack"
        return command

    def accept_rotation_ack(self, ack: DealerAck) -> bool:
        command = self.pending_rotation
        if self.phase is not PartAPhase.WAITING_ROTATION_ACK or command is None:
            raise ValueError("no rotation acknowledgement is expected")
        if (
            ack.command_id != command.command_id
            or ack.command is not command.command
            or ack.target_slot is not command.target_slot
        ):
            self._enter_recovery("rotation_ack_command_or_target_mismatch")
            return False
        if ack.status is not DealerAckStatus.SUCCEEDED:
            self._enter_recovery(f"rotation_ack_{ack.status.value}")
            return False
        if ack.sensor_evidence.at_target is not True:
            self._enter_recovery("rotation_ack_missing_at_target_evidence")
            return False
        self.pending_rotation = None
        self.phase = PartAPhase.VERIFYING_IDENTITY
        self.last_reason = "rotation_confirmed_verify_identity"
        return True

    def accept_identity(self, observation: FaceIdentityObservation) -> bool:
        if self.phase is not PartAPhase.VERIFYING_IDENTITY:
            raise ValueError("identity evidence is outside the verification window")
        seat = self.focus_seat
        if (
            observation.session_id != self.session_id
            or observation.expected_state_version != self.engine.state.state_version
            or observation.focus_seat is not seat
        ):
            self.last_reason = "stale_or_wrong_identity_context"
            return False
        if observation.identity_state is not FaceIdentityState.MATCHED:
            self.last_reason = f"identity_{observation.identity_state.value}"
            return False
        if observation.registered_seat is not seat or observation.player_id is None:
            self.last_reason = "identity_registered_seat_mismatch"
            return False
        self.verified_player_id = observation.player_id
        self.phase = PartAPhase.WAITING_PLAYER_ACTION
        self.last_reason = "identity_verified_action_window_open"
        return True

    def accept_action(
        self, observation: PlayerActionObservation
    ) -> CoordinatorActionOutcome:
        if self.phase is not PartAPhase.WAITING_PLAYER_ACTION:
            return CoordinatorActionOutcome(
                False, "identity_not_verified", None, self.focus_seat
            )
        result = self.engine.apply_observation(observation)
        if not result.accepted:
            self.last_reason = f"action_rejected:{result.reason}"
            return CoordinatorActionOutcome(
                False, result.reason, result, self.focus_seat
            )

        self.verified_player_id = None
        self.pending_rotation = None
        if (
            self.engine.state.phase is HandPhase.AWAITING_ACTION
            and self.engine.state.acting_seat is not None
        ):
            self.phase = PartAPhase.WAITING_ROTATION_ACK
            self.last_reason = "action_committed_rotate_to_next_seat"
        else:
            self.phase = PartAPhase.ROUND_COMPLETE
            self.last_reason = f"part_a_boundary:{self.engine.state.phase.value}"
        return CoordinatorActionOutcome(
            True, result.reason, result, self.focus_seat
        )

    def revoke_identity(self, reason: str) -> None:
        """Close an open action window without changing game or ledger state."""

        if self.phase is not PartAPhase.WAITING_PLAYER_ACTION:
            raise ValueError("identity can only be revoked from an open action window")
        if not reason.strip():
            raise ValueError("identity revocation reason is required")
        self.verified_player_id = None
        self.phase = PartAPhase.VERIFYING_IDENTITY
        self.last_reason = f"identity_revoked:{reason}"

    def complete_pilot(self, reason: str) -> None:
        """Stop at an explicit non-product pilot boundary."""

        if not reason.strip():
            raise ValueError("pilot completion reason is required")
        self.pending_rotation = None
        self.verified_player_id = None
        self.phase = PartAPhase.ROUND_COMPLETE
        self.last_reason = f"pilot_complete:{reason}"

    def _enter_recovery(self, reason: str) -> None:
        self.phase = PartAPhase.RECOVERY_REQUIRED
        self.verified_player_id = None
        self.last_reason = reason
