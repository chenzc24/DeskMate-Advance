"""Session-only actor attribution contracts for player action evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import PlayerActionObservation, Seat


class ActorBindingState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class ActorBinding:
    """An expiring link from state-owned focus to a verified session player."""

    binding_id: str
    session_id: str
    hand_id: str
    expected_state_version: int
    focus_seat: Seat
    player_id: str
    person_track_id: str
    verified_at_ns: int
    valid_until_ns: int
    identity_confidence: float
    camera_epoch: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.binding_id,
            self.session_id,
            self.hand_id,
            self.player_id,
            self.person_track_id,
        ):
            if not value.strip():
                raise ValueError("actor binding IDs must be non-empty")
        if self.expected_state_version < 0 or self.camera_epoch < 0:
            raise ValueError("actor binding versions must be non-negative")
        if self.verified_at_ns < 0 or self.valid_until_ns <= self.verified_at_ns:
            raise ValueError("actor binding validity window is invalid")
        if not 0.0 <= self.identity_confidence <= 1.0:
            raise ValueError("identity confidence must be in [0, 1]")

    def is_valid_at(self, observed_at_ns: int) -> bool:
        return self.verified_at_ns <= observed_at_ns <= self.valid_until_ns

    def matches_observation(self, observation: PlayerActionObservation) -> bool:
        return (
            observation.hand_id == self.hand_id
            and observation.expected_state_version == self.expected_state_version
            and observation.focus_seat is self.focus_seat
        )


@dataclass(frozen=True, slots=True)
class AttributedActionCandidate:
    """Action evidence accompanied by the actor binding that authorized it."""

    observation: PlayerActionObservation
    binding: ActorBinding
    attribution_source: str
    attribution_confidence: float
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.attribution_source.strip():
            raise ValueError("attribution source is required")
        if not 0.0 <= self.attribution_confidence <= 1.0:
            raise ValueError("attribution confidence must be in [0, 1]")
        if not self.binding.matches_observation(self.observation):
            raise ValueError("action observation does not match actor binding context")
        if not self.binding.is_valid_at(self.observation.observed_at_ns):
            raise ValueError("action observation is outside actor binding validity")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("attribution quality flags cannot be empty")


def actor_binding_to_dict(binding: ActorBinding) -> dict[str, object]:
    """Serialize metadata only; no face, voice or body embedding is exposed."""

    return {
        "schema_version": "1.0",
        "binding_id": binding.binding_id,
        "session_id": binding.session_id,
        "hand_id": binding.hand_id,
        "expected_state_version": binding.expected_state_version,
        "focus_seat": binding.focus_seat.value,
        "player_id": binding.player_id,
        "person_track_id": binding.person_track_id,
        "verified_at_ns": binding.verified_at_ns,
        "valid_until_ns": binding.valid_until_ns,
        "identity_confidence": binding.identity_confidence,
        "camera_epoch": binding.camera_epoch,
    }
