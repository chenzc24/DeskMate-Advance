"""Temporal confirmation from owned gesture evidence to action observations."""

from __future__ import annotations

from dataclasses import dataclass

from poker_dealer.domain import (
    ActionEvidenceState,
    PlayerActionObservation,
    Seat,
)

from .config import GesturePilotConfig


@dataclass(frozen=True, slots=True)
class GestureFrameEvidence:
    observed_at_ns: int
    hand_present: bool
    hand_in_focus_roi: bool
    gesture_label: str | None
    gesture_score: float | None
    centroid_x: float | None = None
    centroid_y: float | None = None
    handedness: str | None = None
    inference_latency_ms: float | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be non-negative")
        if self.gesture_score is not None and not 0.0 <= self.gesture_score <= 1.0:
            raise ValueError("gesture_score must be in [0, 1]")
        if not self.hand_present and any(
            value is not None
            for value in (
                self.gesture_label,
                self.gesture_score,
                self.centroid_x,
                self.centroid_y,
                self.handedness,
            )
        ):
            raise ValueError("missing-hand evidence cannot carry hand outputs")
        if self.hand_present and not self.hand_in_focus_roi and (
            self.centroid_x is None or self.centroid_y is None
        ):
            raise ValueError("out-of-ROI evidence requires a hand centroid")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("gesture evidence quality flags cannot be empty")


@dataclass(frozen=True, slots=True)
class ActionObservationContext:
    hand_id: str
    expected_state_version: int
    focus_seat: Seat

    def __post_init__(self) -> None:
        if not self.hand_id.strip():
            raise ValueError("hand_id is required")
        if self.expected_state_version < 0:
            raise ValueError("expected_state_version must be non-negative")


class GestureTemporalAdapter:
    """Emit one candidate per stable gesture hold and require release to rearm."""

    def __init__(self, config: GesturePilotConfig) -> None:
        self.config = config
        self._last_timestamp_ns: int | None = None
        self._pending_label: str | None = None
        self._pending_started_at_ns: int | None = None
        self._pending_frames = 0
        self._release_frames = 0
        self._latched_label: str | None = None
        self._last_candidate_at_ns: int | None = None
        self._sequence = 0

    def process(
        self,
        evidence: GestureFrameEvidence,
        context: ActionObservationContext,
    ) -> PlayerActionObservation:
        if (
            self._last_timestamp_ns is not None
            and evidence.observed_at_ns < self._last_timestamp_ns
        ):
            raise ValueError("gesture evidence timestamps must be monotonic")
        self._last_timestamp_ns = evidence.observed_at_ns
        self._sequence += 1

        if not evidence.hand_present:
            self._note_release()
            self._reset_pending()
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.NO_ACTION,
                quality_flags=("hand_not_detected",),
            )
        if not evidence.hand_in_focus_roi:
            self._note_release()
            self._reset_pending()
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.OUT_OF_ROI,
                quality_flags=("hand_centroid_outside_focus_roi",),
            )

        label = evidence.gesture_label
        score = evidence.gesture_score
        if (
            label is None
            or label in self.config.ignored_gestures
            or label not in self.config.gesture_to_action
        ):
            self._note_release()
            self._reset_pending()
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.UNKNOWN,
                quality_flags=("unmapped_gesture",),
            )
        if score is None or score < self.config.confirmation.minimum_score:
            self._note_release()
            self._reset_pending()
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.UNKNOWN,
                quality_flags=("gesture_below_score_threshold",),
            )

        self._release_frames = 0
        if label != self._pending_label:
            self._pending_label = label
            self._pending_started_at_ns = evidence.observed_at_ns
            self._pending_frames = 1
        else:
            self._pending_frames += 1
        started_at_ns = (
            evidence.observed_at_ns
            if self._pending_started_at_ns is None
            else self._pending_started_at_ns
        )
        stable_duration_ms = max(
            0, (evidence.observed_at_ns - started_at_ns) // 1_000_000
        )

        if self._latched_label == label:
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.NO_ACTION,
                window_started_at_ns=started_at_ns,
                stable_frames=self._pending_frames,
                stable_duration_ms=stable_duration_ms,
                quality_flags=("gesture_latched_until_release",),
            )

        cooldown_elapsed = (
            self._last_candidate_at_ns is None
            or evidence.observed_at_ns - self._last_candidate_at_ns
            >= self.config.confirmation.cooldown_ms * 1_000_000
        )
        confirmed = (
            self._pending_frames
            >= self.config.confirmation.minimum_stable_frames
            and stable_duration_ms
            >= self.config.confirmation.minimum_stable_duration_ms
            and cooldown_elapsed
        )
        if confirmed:
            self._latched_label = label
            self._last_candidate_at_ns = evidence.observed_at_ns
            return self._observation(
                evidence,
                context,
                ActionEvidenceState.CANDIDATE,
                window_started_at_ns=started_at_ns,
                stable_frames=self._pending_frames,
                stable_duration_ms=stable_duration_ms,
            )

        flags = () if cooldown_elapsed else ("candidate_cooldown_active",)
        return self._observation(
            evidence,
            context,
            ActionEvidenceState.ACTION_START,
            window_started_at_ns=started_at_ns,
            stable_frames=self._pending_frames,
            stable_duration_ms=stable_duration_ms,
            quality_flags=flags,
        )

    def _note_release(self) -> None:
        self._release_frames += 1
        if self._release_frames >= self.config.confirmation.release_frames:
            self._latched_label = None

    def _reset_pending(self) -> None:
        self._pending_label = None
        self._pending_started_at_ns = None
        self._pending_frames = 0

    def _observation(
        self,
        evidence: GestureFrameEvidence,
        context: ActionObservationContext,
        state: ActionEvidenceState,
        *,
        window_started_at_ns: int | None = None,
        stable_frames: int | None = None,
        stable_duration_ms: int | None = None,
        quality_flags: tuple[str, ...] = (),
    ) -> PlayerActionObservation:
        candidate_action = None
        confidence = evidence.gesture_score
        if state is ActionEvidenceState.CANDIDATE:
            assert evidence.gesture_label is not None
            candidate_action = self.config.gesture_to_action[evidence.gesture_label]
        return PlayerActionObservation(
            observation_id=(
                f"gesture-pilot:{context.hand_id}:{self._sequence}:"
                f"{evidence.observed_at_ns}"
            ),
            hand_id=context.hand_id,
            expected_state_version=context.expected_state_version,
            window_started_at_ns=(
                evidence.observed_at_ns
                if window_started_at_ns is None
                else window_started_at_ns
            ),
            observed_at_ns=evidence.observed_at_ns,
            focus_seat=context.focus_seat,
            evidence_state=state,
            candidate_action=candidate_action,
            confidence=confidence,
            stable_duration_ms=(
                0 if stable_duration_ms is None else stable_duration_ms
            ),
            stable_frames=1 if stable_frames is None else stable_frames,
            model_version=(
                f"{self.config.model.model_id}@{self.config.model.version}"
            ),
            calibration_version=self.config.calibration_version,
            quality_flags=tuple(dict.fromkeys(evidence.quality_flags + quality_flags)),
        )


def observation_to_dict(observation: PlayerActionObservation) -> dict[str, object]:
    """Serialize the owned record to the frozen schema 1.0 shape."""

    return {
        "schema_version": "1.0",
        "observation_id": observation.observation_id,
        "hand_id": observation.hand_id,
        "expected_state_version": observation.expected_state_version,
        "window_started_at_ns": observation.window_started_at_ns,
        "observed_at_ns": observation.observed_at_ns,
        "focus_seat": observation.focus_seat.value,
        "evidence_state": observation.evidence_state.value,
        "candidate_action": (
            observation.candidate_action.value
            if observation.candidate_action is not None
            else None
        ),
        "confidence": observation.confidence,
        "stable_duration_ms": observation.stable_duration_ms,
        "stable_frames": observation.stable_frames,
        "model_version": observation.model_version,
        "calibration_version": observation.calibration_version,
        "quality_flags": list(observation.quality_flags),
    }
