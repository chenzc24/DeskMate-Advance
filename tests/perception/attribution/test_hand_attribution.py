from __future__ import annotations

from pathlib import Path

from poker_dealer.perception.actions import GestureFrameEvidence
from poker_dealer.perception.attribution import (
    ActorAttributionConfig,
    HandAttributionState,
    LandmarkPoint,
    PersonPoseEvidence,
    TargetPersonTracker,
    attribute_hands_to_target,
)


def config() -> ActorAttributionConfig:
    return ActorAttributionConfig(
        pose_model_id="pose",
        pose_model_version="test",
        pose_asset_path=Path("unused"),
        pose_asset_sha256="0" * 64,
        num_poses=4,
        minimum_pose_confidence=0.5,
        maximum_face_pose_distance=0.22,
        maximum_track_jump=0.25,
        maximum_hand_wrist_distance=0.18,
        minimum_assignment_margin=0.03,
        actor_lease_ms=2500,
        max_hands=4,
    )


def point(x: float, y: float) -> LandmarkPoint:
    return LandmarkPoint(x, y, 0.99, 0.99)


def pose(index: int, x: float) -> PersonPoseEvidence:
    return PersonPoseEvidence(
        detector_index=index,
        nose=point(x, 0.2),
        left_shoulder=point(x - 0.05, 0.4),
        right_shoulder=point(x + 0.05, 0.4),
        left_wrist=point(x - 0.10, 0.65),
        right_wrist=point(x + 0.10, 0.65),
        bbox_xyxy=(x - 0.15, 0.15, x + 0.15, 0.9),
    )


def hand(index: int, x: float, label: str) -> GestureFrameEvidence:
    return GestureFrameEvidence(
        observed_at_ns=1_000_000_000,
        hand_present=True,
        hand_in_focus_roi=True,
        gesture_label=label,
        gesture_score=0.9,
        centroid_x=x,
        centroid_y=0.60,
        wrist_x=x,
        wrist_y=0.65,
        handedness="Right",
        detector_index=index,
    )


def test_neighbor_hand_is_rejected_and_target_hand_selected() -> None:
    target = pose(0, 0.30)
    neighbor = pose(1, 0.75)
    result = attribute_hands_to_target(
        (hand(0, 0.40, "Thumb_Up"), hand(1, 0.65, "Open_Palm")),
        (target, neighbor),
        target_pose_detector_index=0,
        config=config(),
    )
    assert result.state is HandAttributionState.BOUND
    assert result.selected_hand is not None
    assert result.selected_hand.gesture_label == "Thumb_Up"
    assert result.target_hand_count == 1
    assert result.rejected_hand_count == 1
    assert "non_target_hands_rejected" in result.quality_flags


def test_two_target_hands_conflict_is_ambiguous() -> None:
    target = pose(0, 0.50)
    result = attribute_hands_to_target(
        (hand(0, 0.40, "Thumb_Up"), hand(1, 0.60, "Open_Palm")),
        (target,),
        target_pose_detector_index=0,
        config=config(),
    )
    assert result.state is HandAttributionState.AMBIGUOUS
    assert result.selected_hand is None
    assert "target_player_hands_gesture_conflict" in result.quality_flags


def test_two_target_hands_same_action_choose_strongest() -> None:
    target = pose(0, 0.50)
    result = attribute_hands_to_target(
        (hand(0, 0.40, "Thumb_Up"), hand(1, 0.60, "Thumb_Up")),
        (target,),
        target_pose_detector_index=0,
        config=config(),
    )
    assert result.state is HandAttributionState.BOUND
    assert result.target_hand_count == 2
    assert "multiple_target_hands_agree" in result.quality_flags


def test_person_tracker_does_not_trust_detector_order() -> None:
    tracker = TargetPersonTracker(config())
    first = tracker.acquire(
        (pose(0, 0.30), pose(1, 0.75)),
        face_bbox_xywh=(160, 60, 64, 96),
        frame_width=640,
        frame_height=480,
        observed_at_ns=1,
    )
    assert first is not None
    assert first.pose.body_anchor[0] == 0.30

    # Detector indices/order swap; spatial continuity keeps the target person.
    updated = tracker.update(
        (pose(0, 0.76), pose(1, 0.31)), observed_at_ns=2
    )
    assert updated is not None
    assert updated.track_id == first.track_id
    assert updated.pose.body_anchor[0] == 0.31
