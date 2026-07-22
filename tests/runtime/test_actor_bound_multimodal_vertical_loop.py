from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from poker_dealer.domain import ActionEvidenceState, PlayerActionType, Seat
from poker_dealer.game import HandEngine, SimulatedDealer
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GestureFrameEvidence,
    GesturePilotConfig,
    GestureTemporalAdapter,
    SpeechConfirmationController,
    SpeechConfirmationStatus,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeechUtteranceEvidence,
    classify_speech_intent,
)
from poker_dealer.perception.attribution import (
    ActorAttributionConfig,
    ActorBinding,
    ActorBindingLease,
    AttributedActionCandidate,
    HandAttributionState,
    LandmarkPoint,
    PersonPoseEvidence,
    SessionSpeakerGallery,
    SpeakerVerificationState,
    attribute_hands_to_target,
)
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState
from poker_dealer.runtime import SequentialPartACoordinator


ROOT = Path(__file__).resolve().parents[2]
SPEECH_CONFIG = SpeechPilotConfig.from_json(
    ROOT / "configs/perception/actions_speech_pilot.json"
)
GESTURE_CONFIG = GesturePilotConfig.from_json(
    ROOT / "configs/perception/actions_laptop_pilot.json"
)
ATTRIBUTION_CONFIG = ActorAttributionConfig.from_json(
    ROOT / "configs/perception/actor_binding_session.json"
)


def matched_identity(coordinator: SequentialPartACoordinator) -> FaceIdentityObservation:
    assert coordinator.focus_seat is Seat.D
    return FaceIdentityObservation(
        observation_id="face:player-d",
        session_id=coordinator.session_id,
        expected_state_version=coordinator.engine.state.state_version,
        observed_at_ns=1_000_000_000,
        focus_seat=Seat.D,
        identity_state=FaceIdentityState.MATCHED,
        player_id="player_d",
        registered_seat=Seat.D,
        similarity=0.92,
        second_best_similarity=0.20,
        stable_frames=5,
        stable_duration_ms=400,
        model_version="face@test",
        policy_version="test",
    )


def ready_strict_loop() -> tuple[SequentialPartACoordinator, ActorBinding]:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("actor-bound-e2e", Seat.A),
        "session",
        require_actor_binding=True,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(command, 2))
    identity = matched_identity(coordinator)
    assert coordinator.accept_identity(identity)
    binding = ActorBindingLease(lease_ms=10_000).open(
        identity,
        hand_id=coordinator.engine.state.hand_id,
        person_track_id="person:target",
    )
    coordinator.bind_actor(binding)
    return coordinator, binding


def utterance(word: str, observed_at_ns: int) -> SpeechUtteranceEvidence:
    return SpeechUtteranceEvidence(
        window_started_at_ns=observed_at_ns - 400_000_000,
        observed_at_ns=observed_at_ns,
        transcript=word,
        confidence=0.98,
        is_final=True,
        supporting_blocks=5,
    )


def test_speaker_gallery_to_same_speaker_confirm_to_state_commit() -> None:
    coordinator, binding = ready_strict_loop()
    gallery = SessionSpeakerGallery(
        "session", minimum_samples=3, minimum_similarity=0.70, minimum_margin=0.08
    )
    gallery.enroll(
        "player_d",
        [
            np.array((1.0, 0.01, 0.0), dtype=np.float32),
            np.array((0.99, 0.02, 0.0), dtype=np.float32),
            np.array((1.0, 0.0, 0.01), dtype=np.float32),
        ],
    )
    gallery.enroll(
        "player_a",
        [
            np.array((0.0, 1.0, 0.01), dtype=np.float32),
            np.array((0.01, 0.99, 0.0), dtype=np.float32),
            np.array((0.0, 1.0, 0.0), dtype=np.float32),
        ],
    )
    context = ActionObservationContext(
        coordinator.engine.state.hand_id,
        coordinator.engine.state.state_version,
        Seat.D,
    )
    command_evidence = utterance("call", 1_500_000_000)
    command = SpeechObservationAdapter(SPEECH_CONFIG).process(
        command_evidence, context
    )
    speaker = gallery.match(
        np.array((0.98, 0.03, 0.0), dtype=np.float32), speaker_frames=80
    )
    assert speaker.state is SpeakerVerificationState.MATCHED
    assert speaker.player_id == binding.player_id

    confirmation = SpeechConfirmationController(require_speaker_match=True)
    assert confirmation.offer_action(
        command, binding, speaker_player_id=speaker.player_id
    ).status is SpeechConfirmationStatus.PENDING

    wrong_speaker = gallery.match(
        np.array((0.02, 0.98, 0.0), dtype=np.float32), speaker_frames=80
    )
    rejected = confirmation.handle_control(
        classify_speech_intent(utterance("confirm", 1_800_000_000), SPEECH_CONFIG),
        binding,
        speaker_player_id=wrong_speaker.player_id,
    )
    assert rejected.status is SpeechConfirmationStatus.REJECTED
    assert coordinator.engine.state.state_version == 0

    confirmed = confirmation.handle_control(
        classify_speech_intent(utterance("confirm", 2_000_000_000), SPEECH_CONFIG),
        binding,
        speaker_player_id=speaker.player_id,
    )
    assert confirmed.status is SpeechConfirmationStatus.CONFIRMED
    assert confirmed.observation is not None
    outcome = coordinator.accept_attributed_action(
        AttributedActionCandidate(
            confirmed.observation,
            binding,
            "session_speaker_verification",
            min(speaker.similarity or 0.0, confirmed.observation.confidence or 0.0),
        )
    )
    assert outcome.accepted
    assert outcome.next_seat is Seat.A
    assert coordinator.engine.state.state_version == 1
    gallery.clear()


def point(x: float, y: float) -> LandmarkPoint:
    return LandmarkPoint(x, y, 0.99, 0.99)


def pose(index: int, x: float) -> PersonPoseEvidence:
    return PersonPoseEvidence(
        detector_index=index,
        nose=point(x, 0.2),
        left_shoulder=point(x - 0.05, 0.4),
        right_shoulder=point(x + 0.05, 0.4),
        left_wrist=point(x - 0.1, 0.65),
        right_wrist=point(x + 0.1, 0.65),
        bbox_xyxy=(x - 0.15, 0.15, x + 0.15, 0.9),
    )


def hand(index: int, x: float, label: str, observed_at_ns: int) -> GestureFrameEvidence:
    return GestureFrameEvidence(
        observed_at_ns=observed_at_ns,
        hand_present=True,
        hand_in_focus_roi=True,
        gesture_label=label,
        gesture_score=0.96,
        centroid_x=x,
        centroid_y=0.60,
        wrist_x=x,
        wrist_y=0.65,
        handedness="Right",
        detector_index=index,
    )


def test_neighbor_hand_rejection_to_target_gesture_state_commit() -> None:
    coordinator, binding = ready_strict_loop()
    context = ActionObservationContext(
        coordinator.engine.state.hand_id,
        coordinator.engine.state.state_version,
        Seat.D,
    )
    adapter = GestureTemporalAdapter(GESTURE_CONFIG)
    observation = None
    for index in range(8):
        observed_at_ns = 1_500_000_000 + index * 100_000_000
        attribution = attribute_hands_to_target(
            (
                hand(0, 0.40, "Thumb_Up", observed_at_ns),
                hand(1, 0.65, "Open_Palm", observed_at_ns),
            ),
            (pose(0, 0.30), pose(1, 0.75)),
            target_pose_detector_index=0,
            config=ATTRIBUTION_CONFIG,
        )
        assert attribution.state is HandAttributionState.BOUND
        assert attribution.selected_hand is not None
        assert attribution.selected_hand.gesture_label == "Thumb_Up"
        assert attribution.rejected_hand_count == 1
        current = adapter.process(attribution.selected_hand, context)
        if current.evidence_state is ActionEvidenceState.CANDIDATE:
            observation = current
            break

    assert observation is not None
    assert observation.evidence_state is ActionEvidenceState.CANDIDATE
    assert observation.candidate_action is PlayerActionType.CALL
    outcome = coordinator.accept_attributed_action(
        AttributedActionCandidate(
            observation,
            binding,
            "face_pose_wrist",
            0.96,
            ("non_target_hands_rejected",),
        )
    )
    assert outcome.accepted
    assert outcome.next_seat is Seat.A
    assert coordinator.engine.state.state_version == 1
