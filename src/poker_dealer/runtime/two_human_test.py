"""Development-only action sources for a two-human four-seat live test."""

from __future__ import annotations

from typing import Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    FramePacket,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
)
from poker_dealer.perception.attribution import ActorBinding
from poker_dealer.perception.identity import (
    FaceIdentityObservation,
    FaceIdentityState,
)
from poker_dealer.game import PromotionPolicy

from .ports import ActionEvidence, RuntimeObservationContext


class TwoHumanAutoFoldSource:
    """Delegate human seats while simulators traverse normal runtime gates."""

    MODEL_VERSION = "development-two-human-auto-fold@1"
    POLICY_VERSION = "development-two-human-ad@1"

    def __init__(
        self,
        delegate: object,
        simulated_players: Mapping[Seat, str],
        promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        if not simulated_players:
            raise ValueError("at least one simulated player is required")
        if any(not player_id.strip() for player_id in simulated_players.values()):
            raise ValueError("simulated player IDs must be non-empty")
        if len(set(simulated_players.values())) != len(simulated_players):
            raise ValueError("simulated player IDs must be unique")
        self.delegate = delegate
        self.simulated_players = dict(simulated_players)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.sequence = 0

    def observe_identity(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> FaceIdentityObservation | None:
        player_id = self._simulated_player(context)
        if player_id is None:
            return self.delegate.observe_identity(frame, context, observed_at_ns)
        self.sequence += 1
        assert context.focus_seat is not None
        return FaceIdentityObservation(
            observation_id=self._observation_id("identity", context),
            session_id=context.session_id,
            expected_state_version=context.state_version,
            observed_at_ns=observed_at_ns,
            focus_seat=context.focus_seat,
            identity_state=FaceIdentityState.MATCHED,
            player_id=player_id,
            registered_seat=context.focus_seat,
            similarity=1.0,
            second_best_similarity=None,
            stable_frames=1,
            stable_duration_ms=0,
            model_version=self.MODEL_VERSION,
            policy_version=self.POLICY_VERSION,
            quality_flags=(
                "development_test_scenario",
                "simulated_identity",
            ),
        )

    def observe_action(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> ActionEvidence | None:
        player_id = self._simulated_player(context)
        if player_id is None:
            return self.delegate.observe_action(frame, context, observed_at_ns)
        if PlayerActionType.FOLD not in context.legal_actions:
            raise RuntimeError("auto-fold is not legal in the current engine state")
        self.sequence += 1
        assert context.focus_seat is not None
        observation_id = self._observation_id("action", context)
        stable_duration_ms = self.promotion_policy.minimum_stable_duration_ms
        observation = PlayerActionObservation(
            observation_id=observation_id,
            hand_id=context.hand_id,
            expected_state_version=context.state_version,
            window_started_at_ns=max(
                0,
                observed_at_ns - stable_duration_ms * 1_000_000,
            ),
            observed_at_ns=observed_at_ns,
            focus_seat=context.focus_seat,
            evidence_state=ActionEvidenceState.CANDIDATE,
            candidate_action=PlayerActionType.FOLD,
            confidence=1.0,
            stable_duration_ms=stable_duration_ms,
            stable_frames=self.promotion_policy.minimum_stable_frames,
            model_version=self.MODEL_VERSION,
            calibration_version=self.POLICY_VERSION,
            quality_flags=(
                "development_test_scenario",
                "simulated_player",
                "auto_fold",
            ),
        )
        binding = ActorBinding(
            binding_id=f"{observation_id}:binding",
            session_id=context.session_id,
            hand_id=context.hand_id,
            expected_state_version=context.state_version,
            focus_seat=context.focus_seat,
            player_id=player_id,
            person_track_id=f"simulated-track:{context.focus_seat.value}",
            verified_at_ns=observed_at_ns,
            valid_until_ns=observed_at_ns + 1_000_000_000,
            identity_confidence=1.0,
            camera_epoch=context.camera_epoch,
        )
        return ActionEvidence(
            observation=observation,
            actor_binding=binding,
            attribution_source="development_simulator",
            attribution_confidence=1.0,
            quality_flags=(
                "development_test_scenario",
                "simulated_player",
                "auto_fold",
            ),
        )

    def _simulated_player(
        self,
        context: RuntimeObservationContext,
    ) -> str | None:
        if context.focus_seat is None:
            return None
        return self.simulated_players.get(context.focus_seat)

    def _observation_id(
        self,
        kind: str,
        context: RuntimeObservationContext,
    ) -> str:
        assert context.focus_seat is not None
        return (
            f"development-two-human:{kind}:{context.hand_id}:"
            f"{context.state_version}:{context.focus_seat.value}:{self.sequence}"
        )
