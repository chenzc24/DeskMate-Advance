"""Typed speech intents and actor-bound spoken confirmation state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from poker_dealer.domain import ActionEvidenceState, PlayerActionObservation, PlayerActionType
from poker_dealer.perception.attribution.domain import ActorBinding

from .speech import SpeechPilotConfig, SpeechUtteranceEvidence


class SpeechIntentKind(StrEnum):
    ACTION = "action"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SpeechIntentObservation:
    kind: SpeechIntentKind
    transcript: str
    confidence: float | None
    window_started_at_ns: int
    observed_at_ns: int
    action: PlayerActionType | None = None

    def __post_init__(self) -> None:
        if self.observed_at_ns < self.window_started_at_ns:
            raise ValueError("speech intent timestamps are invalid")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speech confidence must be in [0, 1]")
        if (self.kind is SpeechIntentKind.ACTION) != (self.action is not None):
            raise ValueError("only action speech intents carry a poker action")


def classify_speech_intent(
    evidence: SpeechUtteranceEvidence, config: SpeechPilotConfig
) -> SpeechIntentObservation:
    transcript = evidence.canonical_transcript
    confidence_ok = (
        evidence.confidence is not None
        and evidence.confidence >= config.minimum_confidence
    )
    if not evidence.is_final or not confidence_ok:
        kind = SpeechIntentKind.UNKNOWN
        action = None
    elif transcript in config.command_to_action:
        kind = SpeechIntentKind.ACTION
        action = config.command_to_action[transcript]
    elif transcript == "confirm":
        kind = SpeechIntentKind.CONFIRM
        action = None
    elif transcript == "cancel":
        kind = SpeechIntentKind.CANCEL
        action = None
    else:
        kind = SpeechIntentKind.UNKNOWN
        action = None
    return SpeechIntentObservation(
        kind=kind,
        transcript=transcript,
        confidence=evidence.confidence,
        window_started_at_ns=evidence.window_started_at_ns,
        observed_at_ns=evidence.observed_at_ns,
        action=action,
    )


class SpeechConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NO_PENDING = "no_pending"


@dataclass(frozen=True, slots=True)
class PendingSpeechAction:
    observation: PlayerActionObservation
    binding_id: str
    player_id: str
    expires_at_ns: int
    speaker_player_id: str | None


@dataclass(frozen=True, slots=True)
class SpeechConfirmationOutcome:
    status: SpeechConfirmationStatus
    reason: str
    observation: PlayerActionObservation | None = None


class SpeechConfirmationController:
    """Hold speech commands until a same-actor spoken confirmation arrives."""

    def __init__(
        self,
        *,
        confirmation_timeout_ms: int = 8000,
        require_speaker_match: bool = True,
    ) -> None:
        if confirmation_timeout_ms <= 0:
            raise ValueError("speech confirmation timeout must be positive")
        self.confirmation_timeout_ms = confirmation_timeout_ms
        self.require_speaker_match = require_speaker_match
        self._pending: PendingSpeechAction | None = None

    @property
    def pending(self) -> PendingSpeechAction | None:
        return self._pending

    def offer_action(
        self,
        observation: PlayerActionObservation,
        binding: ActorBinding,
        *,
        speaker_player_id: str | None,
    ) -> SpeechConfirmationOutcome:
        if (
            observation.evidence_state is not ActionEvidenceState.CANDIDATE
            or observation.candidate_action is None
        ):
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_action_candidate_required"
            )
        if not binding.matches_observation(observation) or not binding.is_valid_at(
            observation.observed_at_ns
        ):
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_binding_context_mismatch"
            )
        if self.require_speaker_match and speaker_player_id != binding.player_id:
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_speaker_not_verified"
            )
        self._pending = PendingSpeechAction(
            observation=observation,
            binding_id=binding.binding_id,
            player_id=binding.player_id,
            expires_at_ns=(
                observation.observed_at_ns
                + self.confirmation_timeout_ms * 1_000_000
            ),
            speaker_player_id=speaker_player_id,
        )
        return SpeechConfirmationOutcome(
            SpeechConfirmationStatus.PENDING, "speech_action_waiting_for_confirm"
        )

    def handle_control(
        self,
        intent: SpeechIntentObservation,
        binding: ActorBinding,
        *,
        speaker_player_id: str | None,
    ) -> SpeechConfirmationOutcome:
        pending = self._pending
        if pending is None:
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.NO_PENDING, "no_pending_speech_action"
            )
        if intent.observed_at_ns > pending.expires_at_ns:
            self._pending = None
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.EXPIRED, "pending_speech_action_expired"
            )
        if (
            pending.binding_id != binding.binding_id
            or pending.player_id != binding.player_id
            or not binding.is_valid_at(intent.observed_at_ns)
        ):
            self._pending = None
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_control_binding_mismatch"
            )
        if self.require_speaker_match and speaker_player_id != binding.player_id:
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_control_speaker_not_verified"
            )
        if (
            pending.speaker_player_id is not None
            and speaker_player_id != pending.speaker_player_id
        ):
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "speech_command_control_speaker_mismatch"
            )
        if intent.kind is SpeechIntentKind.CANCEL:
            self._pending = None
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.CANCELLED, "pending_speech_action_cancelled"
            )
        if intent.kind is not SpeechIntentKind.CONFIRM:
            return SpeechConfirmationOutcome(
                SpeechConfirmationStatus.REJECTED, "confirm_or_cancel_required"
            )
        confirmed = replace(
            pending.observation,
            observation_id=f"{pending.observation.observation_id}:spoken-confirmed",
            observed_at_ns=intent.observed_at_ns,
            confidence=min(
                pending.observation.confidence or 0.0, intent.confidence or 0.0
            ),
            stable_duration_ms=max(
                pending.observation.stable_duration_ms,
                (intent.observed_at_ns - pending.observation.window_started_at_ns)
                // 1_000_000,
            ),
            quality_flags=tuple(
                dict.fromkeys(
                    pending.observation.quality_flags
                    + (
                        "speech_spoken_confirmed",
                        "speaker_verified_same_actor",
                        "fusion_sources:speech_verified",
                    )
                )
            ),
        )
        self._pending = None
        return SpeechConfirmationOutcome(
            SpeechConfirmationStatus.CONFIRMED,
            "speech_action_spoken_confirmed",
            confirmed,
        )

    def expire(self, observed_at_ns: int) -> bool:
        if self._pending is None or observed_at_ns <= self._pending.expires_at_ns:
            return False
        self._pending = None
        return True

    def clear(self) -> None:
        self._pending = None
