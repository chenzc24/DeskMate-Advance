from __future__ import annotations

from pathlib import Path

from poker_dealer.domain import Seat
from poker_dealer.perception.actions import (
    ActionObservationContext,
    SpeechConfirmationController,
    SpeechConfirmationStatus,
    SpeechIntentKind,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeechUtteranceEvidence,
    classify_speech_intent,
)
from poker_dealer.perception.attribution import ActorBindingLease
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState


ROOT = Path(__file__).resolve().parents[3]
CONFIG = SpeechPilotConfig.from_json(
    ROOT / "configs/perception/actions_speech_pilot.json"
)


def utterance(word: str, timestamp_ms: int, confidence: float = 0.98):
    return SpeechUtteranceEvidence(
        window_started_at_ns=(timestamp_ms - 400) * 1_000_000,
        observed_at_ns=timestamp_ms * 1_000_000,
        transcript=word,
        confidence=confidence,
        is_final=True,
        supporting_blocks=5,
    )


def binding():
    matched = FaceIdentityObservation(
        observation_id="face",
        session_id="session",
        expected_state_version=0,
        observed_at_ns=1_000_000_000,
        focus_seat=Seat.D,
        identity_state=FaceIdentityState.MATCHED,
        player_id="player_d",
        registered_seat=Seat.D,
        similarity=0.8,
        second_best_similarity=None,
        stable_frames=5,
        stable_duration_ms=300,
        model_version="face@test",
        policy_version="test",
    )
    return ActorBindingLease(lease_ms=5000).open(
        matched, hand_id="hand", person_track_id="person:1"
    )


def test_classifies_actions_and_spoken_controls() -> None:
    assert classify_speech_intent(utterance("call", 1500), CONFIG).kind is SpeechIntentKind.ACTION
    assert classify_speech_intent(utterance("confirm", 1600), CONFIG).kind is SpeechIntentKind.CONFIRM
    assert classify_speech_intent(utterance("cancel", 1700), CONFIG).kind is SpeechIntentKind.CANCEL
    assert classify_speech_intent(utterance("call", 1800, 0.2), CONFIG).kind is SpeechIntentKind.UNKNOWN


def test_same_speaker_command_and_confirm_emits_confirmed_observation() -> None:
    actor = binding()
    context = ActionObservationContext("hand", 0, Seat.D)
    command_evidence = utterance("call", 1500)
    observation = SpeechObservationAdapter(CONFIG).process(command_evidence, context)
    controller = SpeechConfirmationController(require_speaker_match=True)
    offered = controller.offer_action(
        observation, actor, speaker_player_id="player_d"
    )
    assert offered.status is SpeechConfirmationStatus.PENDING

    confirm = classify_speech_intent(utterance("confirm", 2000), CONFIG)
    outcome = controller.handle_control(
        confirm, actor, speaker_player_id="player_d"
    )
    assert outcome.status is SpeechConfirmationStatus.CONFIRMED
    assert outcome.observation is not None
    assert outcome.observation.candidate_action is observation.candidate_action
    assert "speech_spoken_confirmed" in outcome.observation.quality_flags


def test_different_speaker_cannot_confirm_or_cancel() -> None:
    actor = binding()
    context = ActionObservationContext("hand", 0, Seat.D)
    observation = SpeechObservationAdapter(CONFIG).process(
        utterance("raise", 1500), context
    )
    controller = SpeechConfirmationController(require_speaker_match=True)
    assert controller.offer_action(
        observation, actor, speaker_player_id="player_d"
    ).status is SpeechConfirmationStatus.PENDING
    rejected = controller.handle_control(
        classify_speech_intent(utterance("confirm", 2000), CONFIG),
        actor,
        speaker_player_id="player_a",
    )
    assert rejected.status is SpeechConfirmationStatus.REJECTED
    assert controller.pending is not None


def test_same_speaker_cancel_clears_without_action() -> None:
    actor = binding()
    context = ActionObservationContext("hand", 0, Seat.D)
    observation = SpeechObservationAdapter(CONFIG).process(
        utterance("fold", 1500), context
    )
    controller = SpeechConfirmationController(require_speaker_match=True)
    controller.offer_action(observation, actor, speaker_player_id="player_d")
    outcome = controller.handle_control(
        classify_speech_intent(utterance("cancel", 1800), CONFIG),
        actor,
        speaker_player_id="player_d",
    )
    assert outcome.status is SpeechConfirmationStatus.CANCELLED
    assert outcome.observation is None
    assert controller.pending is None


def test_unverified_speaker_fails_closed() -> None:
    actor = binding()
    context = ActionObservationContext("hand", 0, Seat.D)
    observation = SpeechObservationAdapter(CONFIG).process(
        utterance("call", 1500), context
    )
    outcome = SpeechConfirmationController(require_speaker_match=True).offer_action(
        observation, actor, speaker_player_id=None
    )
    assert outcome.status is SpeechConfirmationStatus.REJECTED
    assert outcome.reason == "speech_speaker_not_verified"
