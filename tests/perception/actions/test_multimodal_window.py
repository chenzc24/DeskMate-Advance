from __future__ import annotations

from dataclasses import replace

import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
)
from poker_dealer.perception.actions import MultimodalActionWindow


def candidate(
    source: str,
    action: PlayerActionType,
    timestamp_ms: int,
    *,
    state_version: int = 3,
) -> PlayerActionObservation:
    model = (
        "player-action-vosk@test"
        if source == "speech"
        else "player-action-mediapipe@test"
    )
    return PlayerActionObservation(
        observation_id=f"{source}:{timestamp_ms}",
        hand_id="hand",
        expected_state_version=state_version,
        window_started_at_ns=(timestamp_ms - 300) * 1_000_000,
        observed_at_ns=timestamp_ms * 1_000_000,
        focus_seat=Seat.B,
        evidence_state=ActionEvidenceState.CANDIDATE,
        candidate_action=action,
        confidence=0.97,
        stable_duration_ms=300,
        stable_frames=5,
        model_version=model,
        calibration_version="test",
    )


def test_agreement_emits_immediately_and_clears() -> None:
    window = MultimodalActionWindow(decision_wait_ms=500)
    assert window.add(candidate("gesture", PlayerActionType.CALL, 1000)) is None
    fused = window.add(candidate("speech", PlayerActionType.CALL, 1200))
    assert fused is not None
    assert fused.evidence_state is ActionEvidenceState.CANDIDATE
    assert fused.candidate_action is PlayerActionType.CALL
    assert "multimodal_agreement" in fused.quality_flags
    assert window.poll(2000 * 1_000_000) is None


def test_conflict_is_ambiguous_and_never_carries_action() -> None:
    window = MultimodalActionWindow(decision_wait_ms=500)
    window.add(candidate("gesture", PlayerActionType.CALL, 1000))
    fused = window.add(candidate("speech", PlayerActionType.RAISE, 1200))
    assert fused is not None
    assert fused.evidence_state is ActionEvidenceState.AMBIGUOUS
    assert fused.candidate_action is None


def test_single_modality_waits_then_emits() -> None:
    window = MultimodalActionWindow(decision_wait_ms=500)
    item = candidate("speech", PlayerActionType.FOLD, 1000)
    assert window.add(item) is None
    assert window.poll(1499 * 1_000_000) is None
    fused = window.poll(1500 * 1_000_000)
    assert fused is not None
    assert fused.candidate_action is PlayerActionType.FOLD
    assert "single_modality_candidate" in fused.quality_flags


def test_context_change_requires_explicit_reset() -> None:
    window = MultimodalActionWindow()
    window.add(candidate("speech", PlayerActionType.CALL, 1000))
    with pytest.raises(ValueError, match="context changed"):
        window.add(
            replace(
                candidate("gesture", PlayerActionType.CALL, 1100),
                expected_state_version=4,
            )
        )


def test_non_candidate_does_not_start_decision_timer() -> None:
    window = MultimodalActionWindow()
    item = replace(
        candidate("gesture", PlayerActionType.CALL, 1000),
        evidence_state=ActionEvidenceState.NO_ACTION,
        candidate_action=None,
    )
    assert window.add(item) is None
    assert window.poll(5000 * 1_000_000) is None


def test_four_player_policy_holds_speech_until_ui_or_gesture_confirmation() -> None:
    window = MultimodalActionWindow(
        decision_wait_ms=500,
        max_skew_ms=3000,
        allow_speech_single_source=False,
    )
    speech = candidate("speech", PlayerActionType.CALL, 1000)
    assert window.add(speech) is None
    assert window.poll(1700 * 1_000_000) is None
    assert window.pending_sources == ("speech",)
    confirmed = window.confirm_pending_speech(1800 * 1_000_000)
    assert confirmed is not None
    assert confirmed.candidate_action is PlayerActionType.CALL
    assert "speech_ui_confirmed" in confirmed.quality_flags


def test_four_player_policy_accepts_matching_gesture_confirmation() -> None:
    window = MultimodalActionWindow(
        max_skew_ms=3000, allow_speech_single_source=False
    )
    assert window.add(candidate("speech", PlayerActionType.RAISE, 1000)) is None
    agreed = window.add(candidate("gesture", PlayerActionType.RAISE, 2500))
    assert agreed is not None
    assert agreed.candidate_action is PlayerActionType.RAISE
    assert "multimodal_agreement" in agreed.quality_flags


def test_cancel_pending_speech_does_not_remove_gesture() -> None:
    window = MultimodalActionWindow(max_skew_ms=3000)
    window.add(candidate("gesture", PlayerActionType.CALL, 1000))
    window.add(candidate("speech", PlayerActionType.CALL, 1100))
    # The matching pair emits and clears immediately; repopulate different sources.
    window.add(candidate("gesture", PlayerActionType.CALL, 1200))
    assert not window.cancel_pending_speech()
    assert window.pending_sources == ("gesture",)
    window.clear()
    window.add(candidate("speech", PlayerActionType.CALL, 1300))
    assert window.cancel_pending_speech()
    assert window.pending_sources == ()
