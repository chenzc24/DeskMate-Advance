"""Model-neutral evidence for player behaviour at the focused seat."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .game import PlayerActionType, Seat


class ActionEvidenceState(StrEnum):
    NO_ACTION = "no_action"
    ACTION_START = "action_start"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"
    OCCLUDED = "occluded"
    OUT_OF_ROI = "out_of_roi"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PlayerActionObservation:
    """Temporal evidence only; it never mutates game state by itself."""

    observation_id: str
    hand_id: str
    expected_state_version: int
    window_started_at_ns: int
    observed_at_ns: int
    focus_seat: Seat
    evidence_state: ActionEvidenceState
    candidate_action: PlayerActionType | None
    confidence: float | None
    stable_duration_ms: int
    stable_frames: int
    model_version: str
    calibration_version: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty")
        if not self.hand_id.strip():
            raise ValueError("hand_id must not be empty")
        if self.expected_state_version < 0:
            raise ValueError("expected_state_version must be non-negative")
        if self.window_started_at_ns < 0 or self.observed_at_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if self.observed_at_ns < self.window_started_at_ns:
            raise ValueError("observed_at_ns must not precede the evidence window")
        if self.evidence_state is ActionEvidenceState.CANDIDATE:
            if self.candidate_action is None:
                raise ValueError("candidate evidence requires an action")
            if self.confidence is None:
                raise ValueError("candidate evidence requires confidence")
        elif self.candidate_action is not None:
            raise ValueError("only candidate evidence may carry an action")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.stable_duration_ms < 0:
            raise ValueError("stable_duration_ms must be non-negative")
        if self.stable_frames <= 0:
            raise ValueError("stable_frames must be positive")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not self.calibration_version.strip():
            raise ValueError("calibration_version must not be empty")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("quality_flags cannot contain empty values")
