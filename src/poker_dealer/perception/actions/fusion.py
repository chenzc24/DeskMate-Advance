"""Conservative fusion of model-neutral player-action observations."""

from __future__ import annotations

from collections.abc import Sequence

from poker_dealer.domain import ActionEvidenceState, PlayerActionObservation


def action_observation_source(observation: PlayerActionObservation) -> str:
    if observation.model_version.startswith("player-action-mediapipe"):
        return "gesture"
    if observation.model_version.startswith("player-action-vosk"):
        return "speech"
    return "unknown"


def fuse_action_observations(
    observations: Sequence[PlayerActionObservation],
    *,
    max_skew_ms: int = 1500,
) -> PlayerActionObservation:
    """Fuse contemporaneous evidence without applying a game transition."""

    if not observations:
        raise ValueError("at least one action observation is required")
    if max_skew_ms < 0:
        raise ValueError("fusion skew must be non-negative")
    anchor = observations[0]
    for observation in observations[1:]:
        if (
            observation.hand_id != anchor.hand_id
            or observation.expected_state_version != anchor.expected_state_version
            or observation.focus_seat is not anchor.focus_seat
        ):
            raise ValueError("fusion inputs must share hand, state version and focus seat")

    latest_ns = max(item.observed_at_ns for item in observations)
    fresh = [
        item
        for item in observations
        if latest_ns - item.observed_at_ns <= max_skew_ms * 1_000_000
    ]
    candidates = [
        item
        for item in fresh
        if item.evidence_state is ActionEvidenceState.CANDIDATE
    ]
    actions = {item.candidate_action for item in candidates}
    sources = sorted({action_observation_source(item) for item in fresh})
    flags = [f"fusion_sources:{','.join(sources)}"]
    candidate_action = None
    confidence = max(
        (item.confidence for item in fresh if item.confidence is not None),
        default=None,
    )

    if len(actions) > 1:
        state = ActionEvidenceState.AMBIGUOUS
        flags.append("modality_action_conflict")
    elif candidates:
        state = ActionEvidenceState.CANDIDATE
        candidate_action = candidates[0].candidate_action
        flags.append(
            "multimodal_agreement" if len(candidates) > 1 else "single_modality_candidate"
        )
    elif any(
        item.evidence_state is ActionEvidenceState.AMBIGUOUS for item in fresh
    ):
        state = ActionEvidenceState.AMBIGUOUS
    elif any(
        item.evidence_state is ActionEvidenceState.ACTION_START for item in fresh
    ):
        state = ActionEvidenceState.ACTION_START
    elif any(item.evidence_state is ActionEvidenceState.UNKNOWN for item in fresh):
        state = ActionEvidenceState.UNKNOWN
    else:
        state = ActionEvidenceState.NO_ACTION

    return PlayerActionObservation(
        observation_id=f"multimodal:{anchor.hand_id}:{latest_ns}",
        hand_id=anchor.hand_id,
        expected_state_version=anchor.expected_state_version,
        window_started_at_ns=min(item.window_started_at_ns for item in fresh),
        observed_at_ns=latest_ns,
        focus_seat=anchor.focus_seat,
        evidence_state=state,
        candidate_action=candidate_action,
        confidence=confidence,
        stable_duration_ms=max(item.stable_duration_ms for item in fresh),
        stable_frames=max(item.stable_frames for item in fresh),
        model_version="multimodal-action-fusion@1.0-development",
        calibration_version="multimodal-pilot-unfrozen-v1",
        quality_flags=tuple(flags),
    )
