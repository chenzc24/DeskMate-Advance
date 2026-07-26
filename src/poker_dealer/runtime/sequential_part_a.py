"""Deterministic gates for the sequential Stage 2A player-action loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    ChipAmountScope,
    ChipObservation,
    ChipObservationStatus,
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerTargetSlot,
    HandPhase,
    NavigationAck,
    NavigationAckStatus,
    NavigationAction,
    NavigationCommand,
    PlayerActionObservation,
    PlayerActionType,
    RobotPoseNode,
    Seat,
)
from poker_dealer.game import ActionResult, HandEngine
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState
from poker_dealer.perception.attribution import ActorBinding, AttributedActionCandidate


class PartAPhase(StrEnum):
    WAITING_ROTATION_ACK = "waiting_rotation_ack"
    WAITING_NAVIGATION_ACK = "waiting_navigation_ack"
    WAITING_VISUAL_SETTLE = "waiting_visual_settle"
    VERIFYING_IDENTITY = "verifying_identity"
    WAITING_PLAYER_ACTION = "waiting_player_action"
    WAITING_CHIP_OBSERVATION = "waiting_chip_observation"
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

    def __init__(
        self,
        engine: HandEngine,
        session_id: str,
        *,
        require_actor_binding: bool = True,
        require_visual_settle: bool = True,
        require_chip_observation: bool = False,
        visual_settle_timeout_ms: int = 5000,
        expected_player_by_seat: Mapping[Seat, str] | None = None,
        minimum_attribution_confidence: float = 0.35,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        if visual_settle_timeout_ms <= 0:
            raise ValueError("visual settle timeout must be positive")
        if not 0.0 <= minimum_attribution_confidence <= 1.0:
            raise ValueError("minimum attribution confidence must be in [0, 1]")
        if (
            engine.state.phase is not HandPhase.AWAITING_ACTION
            or engine.state.acting_seat is None
        ):
            raise ValueError("coordinator requires an awaiting-action game state")
        self.engine = engine
        self.session_id = session_id
        self.require_actor_binding = require_actor_binding
        self.require_visual_settle = require_visual_settle
        self.require_chip_observation = require_chip_observation
        self.visual_settle_timeout_ms = visual_settle_timeout_ms
        self.expected_player_by_seat = dict(expected_player_by_seat or {})
        self.minimum_attribution_confidence = minimum_attribution_confidence
        self.phase = PartAPhase.WAITING_ROTATION_ACK
        self.pending_rotation: DealerCommand | None = None
        self.pending_navigation: NavigationCommand | None = None
        self.pending_action_observation: PlayerActionObservation | None = None
        self.verified_player_id: str | None = None
        self.active_actor_binding: ActorBinding | None = None
        self.last_reason = "rotation_not_requested"
        self._command_sequence = 0
        self._action_window_opened_at_ns: int | None = None
        self._visual_settle_opened_at_ns: int | None = None
        self._attention_window_opened_at_ns: int | None = None
        self._accepted_rotation_ack_ids: set[str] = set()
        self._accepted_navigation_ack_ids: set[str] = set()

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
        self.engine.record_dealer_command(command)
        self.last_reason = "waiting_for_matching_rotation_ack"
        return command

    def request_navigation(
        self,
        issued_at_ns: int,
        *,
        start_pose: RobotPoseNode,
        expected_pose_version: int,
        target_pose: RobotPoseNode = RobotPoseNode.UNKNOWN,
        inter_motion_delay_ms: int = 2500,
    ) -> NavigationCommand:
        """Request mobile navigation instead of the legacy fixed-base rotation."""

        if self.phase not in {
            PartAPhase.WAITING_ROTATION_ACK,
            PartAPhase.WAITING_NAVIGATION_ACK,
        }:
            raise ValueError("navigation can only be requested at a turn boundary")
        if self.pending_navigation is not None:
            return self.pending_navigation
        seat = self.focus_seat
        if seat is None:
            raise ValueError("cannot navigate without an acting seat")
        self._command_sequence += 1
        command = NavigationCommand(
            command_id=(
                f"part-a:{self.engine.state.hand_id}:{self.engine.state.state_version}:"
                f"navigate:{self._command_sequence}"
            ),
            session_id=self.session_id,
            hand_id=self.engine.state.hand_id,
            expected_state_version=self.engine.state.state_version,
            expected_pose_version=expected_pose_version,
            issued_at_ns=issued_at_ns,
            action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
            start_pose=start_pose,
            target_pose=target_pose,
            target_slot=DealerTargetSlot(seat.value),
            inter_motion_delay_ms=inter_motion_delay_ms,
        )
        self.pending_navigation = command
        self.phase = PartAPhase.WAITING_NAVIGATION_ACK
        self.last_reason = "waiting_for_matching_navigation_ack"
        return command

    def accept_navigation_ack(self, ack: NavigationAck) -> bool:
        if ack.command_id in self._accepted_navigation_ack_ids:
            return True
        command = self.pending_navigation
        if self.phase is not PartAPhase.WAITING_NAVIGATION_ACK or command is None:
            raise ValueError("no navigation acknowledgement is expected")
        if (
            ack.command_id != command.command_id
            or ack.session_id != command.session_id
            or ack.hand_id != command.hand_id
            or ack.expected_state_version != command.expected_state_version
            or ack.action is not command.action
            or ack.target_slot is not command.target_slot
        ):
            self._enter_recovery("navigation_ack_context_or_target_mismatch")
            return False
        if ack.status is not NavigationAckStatus.SUCCEEDED:
            self._enter_recovery(f"navigation_ack_{ack.status.value}")
            return False
        if (
            not ack.target_aligned
            or ack.pose_version <= command.expected_pose_version
        ):
            self._enter_recovery("navigation_ack_missing_alignment_or_pose_advance")
            return False
        self._accepted_navigation_ack_ids.add(ack.command_id)
        self.pending_navigation = None
        self._target_confirmed(ack.observed_at_ns, "navigation_confirmed")
        return True

    def accept_rotation_ack(self, ack: DealerAck) -> bool:
        if ack.command_id in self._accepted_rotation_ack_ids:
            return True
        self.engine.record_dealer_ack(ack)
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
        try:
            self.engine.record_dealer_completion(ack)
        except ValueError as exc:
            self._enter_recovery(f"rotation_ack_not_committed:{exc}")
            return False
        self._accepted_rotation_ack_ids.add(ack.command_id)
        self.pending_rotation = None
        self._target_confirmed(ack.observed_at_ns, "rotation_confirmed")
        return True

    def _target_confirmed(self, observed_at_ns: int, source: str) -> None:
        self._attention_window_opened_at_ns = observed_at_ns
        if self.require_visual_settle:
            self._visual_settle_opened_at_ns = observed_at_ns
            self.phase = PartAPhase.WAITING_VISUAL_SETTLE
            self.last_reason = f"{source}_wait_visual_settle"
        else:
            self.phase = PartAPhase.VERIFYING_IDENTITY
            self.last_reason = f"{source}_verify_identity"

    def accept_visual_settle(self) -> None:
        if self.phase is not PartAPhase.WAITING_VISUAL_SETTLE:
            raise ValueError("visual settle is outside the post-rotation window")
        self._visual_settle_opened_at_ns = None
        self.phase = PartAPhase.VERIFYING_IDENTITY
        self.last_reason = "visual_settled_verify_identity"

    def fail_visual_settle(self, reason: str) -> None:
        if self.phase is not PartAPhase.WAITING_VISUAL_SETTLE:
            raise ValueError("visual settle failure is outside its window")
        if not reason.strip():
            raise ValueError("visual settle failure reason is required")
        self._enter_recovery(reason)

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
        expected_player = self.expected_player_by_seat.get(seat)
        if expected_player is not None and observation.player_id != expected_player:
            self.last_reason = "identity_player_not_in_frozen_roster"
            return False
        self.verified_player_id = observation.player_id
        self.active_actor_binding = None
        if self._action_window_opened_at_ns is None:
            self._action_window_opened_at_ns = observation.observed_at_ns
        self._attention_window_opened_at_ns = None
        self.phase = PartAPhase.WAITING_PLAYER_ACTION
        self.last_reason = "identity_verified_action_window_open"
        return True

    def bind_actor(self, binding: ActorBinding) -> None:
        """Attach body/face attribution to the already state-selected player."""

        if self.phase is not PartAPhase.WAITING_PLAYER_ACTION:
            raise ValueError("actor binding requires an open action window")
        if self.verified_player_id is None:
            raise ValueError("actor binding requires a verified player")
        if (
            binding.session_id != self.session_id
            or binding.hand_id != self.engine.state.hand_id
            or binding.expected_state_version != self.engine.state.state_version
            or binding.focus_seat is not self.focus_seat
            or binding.player_id != self.verified_player_id
        ):
            raise ValueError("actor binding does not match runtime context")
        self.active_actor_binding = binding
        self.last_reason = "actor_binding_attached_action_window_open"

    def accept_attributed_action(
        self, candidate: AttributedActionCandidate
    ) -> CoordinatorActionOutcome:
        binding = self.active_actor_binding
        if binding is None:
            return CoordinatorActionOutcome(
                False, "actor_binding_required", None, self.focus_seat
            )
        if candidate.binding.binding_id != binding.binding_id:
            return CoordinatorActionOutcome(
                False, "actor_binding_mismatch", None, self.focus_seat
            )
        if candidate.attribution_confidence < self.minimum_attribution_confidence:
            return CoordinatorActionOutcome(
                False, "attribution_confidence_below_threshold", None, self.focus_seat
            )
        if not binding.is_valid_at(candidate.observation.observed_at_ns):
            return CoordinatorActionOutcome(
                False, "actor_binding_expired", None, self.focus_seat
            )
        return self._accept_action(candidate.observation)

    def accept_action(
        self, observation: PlayerActionObservation
    ) -> CoordinatorActionOutcome:
        if self.require_actor_binding:
            return CoordinatorActionOutcome(
                False, "attributed_action_required", None, self.focus_seat
            )
        return self._accept_action(observation)

    def _accept_action(
        self, observation: PlayerActionObservation
    ) -> CoordinatorActionOutcome:
        if self.phase is not PartAPhase.WAITING_PLAYER_ACTION:
            return CoordinatorActionOutcome(
                False, "identity_not_verified", None, self.focus_seat
            )
        if (
            self.require_chip_observation
            and observation.evidence_state is ActionEvidenceState.CANDIDATE
            and observation.candidate_action
            in {PlayerActionType.BET, PlayerActionType.RAISE}
        ):
            reason = self._chip_gate_candidate_reason(observation)
            if reason is not None:
                self.last_reason = f"action_rejected:{reason}"
                return CoordinatorActionOutcome(
                    False, reason, None, self.focus_seat
                )
            self.pending_action_observation = observation
            self.phase = PartAPhase.WAITING_CHIP_OBSERVATION
            self.last_reason = "bet_or_raise_waiting_chip_observation"
            return CoordinatorActionOutcome(
                False,
                "chip_observation_required",
                None,
                self.focus_seat,
            )
        return self._commit_action(observation)

    def _chip_gate_candidate_reason(
        self, observation: PlayerActionObservation
    ) -> str | None:
        state = self.engine.state
        policy = self.engine.promoter.policy
        if observation.hand_id != state.hand_id:
            return "wrong_hand"
        if observation.expected_state_version != state.state_version:
            return "stale_state_version"
        if observation.focus_seat is not state.acting_seat:
            return "non_current_seat"
        if observation.candidate_action not in state.legal_actions:
            return "illegal_action"
        if observation.confidence is None or observation.confidence < policy.minimum_confidence:
            return "low_confidence"
        if observation.stable_frames < policy.minimum_stable_frames:
            return "insufficient_stable_frames"
        if observation.stable_duration_ms < policy.minimum_stable_duration_ms:
            return "insufficient_stable_duration"
        return None

    def accept_chip_observation(
        self, observation: ChipObservation
    ) -> CoordinatorActionOutcome:
        if self.phase is not PartAPhase.WAITING_CHIP_OBSERVATION:
            return CoordinatorActionOutcome(
                False, "chip_window_not_open", None, self.focus_seat
            )
        pending = self.pending_action_observation
        if pending is None:
            self._enter_recovery("chip_window_without_pending_action")
            return CoordinatorActionOutcome(
                False, "chip_window_without_pending_action", None, self.focus_seat
            )
        if (
            observation.hand_id != self.engine.state.hand_id
            or observation.expected_state_version != self.engine.state.state_version
            or observation.focus_seat is not self.focus_seat
            or observation.observed_at_ns < pending.observed_at_ns
        ):
            self.last_reason = "stale_or_wrong_chip_context"
            return CoordinatorActionOutcome(
                False, "stale_or_wrong_chip_context", None, self.focus_seat
            )
        if observation.status is not ChipObservationStatus.CONFIRMED:
            self.last_reason = f"chip_{observation.status.value}"
            return CoordinatorActionOutcome(
                False, self.last_reason, None, self.focus_seat
            )
        expected = self._expected_chip_units(observation.amount_scope)
        if observation.total_units != expected:
            self.last_reason = (
                f"chip_amount_mismatch:expected={expected}:"
                f"observed={observation.total_units}"
            )
            return CoordinatorActionOutcome(
                False, "chip_amount_mismatch", None, self.focus_seat
            )
        self.pending_action_observation = None
        self.phase = PartAPhase.WAITING_PLAYER_ACTION
        self.last_reason = "chip_observation_confirmed_commit_action"
        return self._commit_action(pending)

    def _expected_chip_units(self, scope: ChipAmountScope) -> int:
        observation = self.pending_action_observation
        assert observation is not None and observation.candidate_action is not None
        state = self.engine.state
        seat = observation.focus_seat
        player = state.players[seat]
        bet_size = self.engine.rules.bet_size(state.street)  # type: ignore[arg-type]
        target = (
            bet_size
            if observation.candidate_action is PlayerActionType.BET
            else state.current_bet_units + bet_size
        )
        if scope is ChipAmountScope.STREET_TOTAL:
            return target
        return target - player.street_commit_units

    def _commit_action(
        self, observation: PlayerActionObservation
    ) -> CoordinatorActionOutcome:
        result = self.engine.apply_observation(observation)
        if not result.accepted:
            self.last_reason = f"action_rejected:{result.reason}"
            return CoordinatorActionOutcome(
                False, result.reason, result, self.focus_seat
            )

        self.verified_player_id = None
        self.active_actor_binding = None
        self._action_window_opened_at_ns = None
        self._attention_window_opened_at_ns = None
        self.pending_rotation = None
        self.pending_navigation = None
        self.pending_action_observation = None
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
        self.active_actor_binding = None
        self.phase = PartAPhase.VERIFYING_IDENTITY
        self.last_reason = f"identity_revoked:{reason}"

    def complete_pilot(self, reason: str) -> None:
        """Stop at an explicit non-product pilot boundary."""

        if not reason.strip():
            raise ValueError("pilot completion reason is required")
        self.pending_rotation = None
        self.pending_navigation = None
        self.pending_action_observation = None
        self.verified_player_id = None
        self.active_actor_binding = None
        self._action_window_opened_at_ns = None
        self._attention_window_opened_at_ns = None
        self.phase = PartAPhase.ROUND_COMPLETE
        self.last_reason = f"pilot_complete:{reason}"

    def check_timeout(self, now_ns: int) -> bool:
        """Pause the authoritative hand when an open action window expires."""

        if now_ns < 0:
            raise ValueError("timeout clock must be non-negative")
        if self.pending_rotation is not None:
            deadline = (
                self.pending_rotation.issued_at_ns
                + self.pending_rotation.timeout_ms * 1_000_000
            )
            if now_ns >= deadline:
                self._enter_recovery("rotation_ack_timeout")
                return True
        if self.pending_navigation is not None:
            deadline = (
                self.pending_navigation.issued_at_ns
                + self.pending_navigation.timeout_ms * 1_000_000
            )
            if now_ns >= deadline:
                self._enter_recovery("navigation_ack_timeout")
                return True
        if (
            self.phase is PartAPhase.WAITING_VISUAL_SETTLE
            and self._visual_settle_opened_at_ns is not None
            and now_ns
            >= self._visual_settle_opened_at_ns
            + self.visual_settle_timeout_ms * 1_000_000
        ):
            self._enter_recovery("visual_settle_timeout")
            return True
        action_deadline_reached = (
            self.phase
            in {
                PartAPhase.VERIFYING_IDENTITY,
                PartAPhase.WAITING_PLAYER_ACTION,
                PartAPhase.WAITING_CHIP_OBSERVATION,
            }
            and self._action_window_opened_at_ns is not None
            and now_ns
            >= self._action_window_opened_at_ns
            + self.engine.rules.action_timeout_seconds * 1_000_000_000
        )
        if action_deadline_reached:
            self._enter_recovery("player_action_timeout")
            return True
        attention_deadline_reached = (
            self.phase
            in {
                PartAPhase.WAITING_VISUAL_SETTLE,
                PartAPhase.VERIFYING_IDENTITY,
            }
            and self._attention_window_opened_at_ns is not None
            and now_ns
            >= self._attention_window_opened_at_ns
            + self.engine.rules.action_timeout_seconds * 1_000_000_000
        )
        if attention_deadline_reached:
            self._enter_recovery("attention_window_timeout")
            return True
        return False

    def _enter_recovery(self, reason: str) -> None:
        self.phase = PartAPhase.RECOVERY_REQUIRED
        self.pending_rotation = None
        self.pending_navigation = None
        self.pending_action_observation = None
        self.verified_player_id = None
        self.active_actor_binding = None
        self._action_window_opened_at_ns = None
        self._visual_settle_opened_at_ns = None
        self._attention_window_opened_at_ns = None
        self.last_reason = reason
        if self.engine.state.phase is not HandPhase.PAUSED_RECOVERY:
            self.engine.pause(
                f"part-a:{self.engine.state.hand_id}:pause:{self.engine.state.state_version}",
                reason,
            )
