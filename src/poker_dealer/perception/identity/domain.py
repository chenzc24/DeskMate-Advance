"""Owned, serializable session identity evidence without face embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import Seat


class FaceIdentityState(StrEnum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    ENROLLMENT_REQUIRED = "enrollment_required"
    IDENTITY_START = "identity_start"
    MATCHED = "matched"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    LOW_QUALITY = "low_quality"
    SEAT_MISMATCH = "seat_mismatch"


@dataclass(frozen=True, slots=True)
class FaceIdentityContext:
    session_id: str
    expected_state_version: int
    focus_seat: Seat

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if self.expected_state_version < 0:
            raise ValueError("state version must be non-negative")


@dataclass(frozen=True, slots=True)
class FaceIdentityObservation:
    observation_id: str
    session_id: str
    expected_state_version: int
    observed_at_ns: int
    focus_seat: Seat
    identity_state: FaceIdentityState
    player_id: str | None
    registered_seat: Seat | None
    similarity: float | None
    second_best_similarity: float | None
    stable_frames: int
    stable_duration_ms: int
    model_version: str
    policy_version: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.session_id.strip():
            raise ValueError("identity and session IDs are required")
        if self.expected_state_version < 0 or self.observed_at_ns < 0:
            raise ValueError("identity versions/timestamps must be non-negative")
        if self.stable_frames <= 0 or self.stable_duration_ms < 0:
            raise ValueError("identity temporal evidence is invalid")
        identified = self.identity_state in {
            FaceIdentityState.MATCHED,
            FaceIdentityState.SEAT_MISMATCH,
        }
        if identified and (
            self.player_id is None
            or self.registered_seat is None
            or self.similarity is None
        ):
            raise ValueError("identified observations require player, seat and score")
        if not identified and (self.player_id is not None or self.registered_seat is not None):
            raise ValueError("unidentified observations cannot expose player identity")
        for score in (self.similarity, self.second_best_similarity):
            if score is not None and not -1.0 <= score <= 1.0:
                raise ValueError("cosine similarity must be in [-1, 1]")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("identity quality flags cannot be empty")


def identity_observation_to_dict(
    observation: FaceIdentityObservation,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "observation_id": observation.observation_id,
        "session_id": observation.session_id,
        "expected_state_version": observation.expected_state_version,
        "observed_at_ns": observation.observed_at_ns,
        "focus_seat": observation.focus_seat.value,
        "identity_state": observation.identity_state.value,
        "player_id": observation.player_id,
        "registered_seat": (
            observation.registered_seat.value
            if observation.registered_seat is not None
            else None
        ),
        "similarity": observation.similarity,
        "second_best_similarity": observation.second_best_similarity,
        "stable_frames": observation.stable_frames,
        "stable_duration_ms": observation.stable_duration_ms,
        "model_version": observation.model_version,
        "policy_version": observation.policy_version,
        "quality_flags": list(observation.quality_flags),
    }
