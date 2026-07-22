"""Associate detected hands with the face-selected person's pose wrists."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math

from poker_dealer.perception.actions.temporal import GestureFrameEvidence

from .config import ActorAttributionConfig
from .pose import PersonPoseEvidence


class HandAttributionState(StrEnum):
    BOUND = "bound"
    NO_HAND = "no_hand"
    NO_TARGET_HAND = "no_target_hand"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class HandAttributionResult:
    state: HandAttributionState
    selected_hand: GestureFrameEvidence | None
    target_hand_count: int
    rejected_hand_count: int
    attribution_confidence: float | None
    quality_flags: tuple[str, ...]

    def temporal_evidence(self, observed_at_ns: int) -> GestureFrameEvidence:
        if self.selected_hand is not None:
            return replace(
                self.selected_hand,
                quality_flags=tuple(
                    dict.fromkeys(self.selected_hand.quality_flags + self.quality_flags)
                ),
            )
        return GestureFrameEvidence(
            observed_at_ns=observed_at_ns,
            hand_present=False,
            hand_in_focus_roi=False,
            gesture_label=None,
            gesture_score=None,
            quality_flags=self.quality_flags,
        )


def _pose_wrist_distance(hand: GestureFrameEvidence, pose: PersonPoseEvidence) -> float:
    assert hand.wrist_x is not None and hand.wrist_y is not None
    wrist = (hand.wrist_x, hand.wrist_y)
    return min(
        math.dist(wrist, (pose.left_wrist.x, pose.left_wrist.y)),
        math.dist(wrist, (pose.right_wrist.x, pose.right_wrist.y)),
    )


def attribute_hands_to_target(
    hands: tuple[GestureFrameEvidence, ...],
    poses: tuple[PersonPoseEvidence, ...],
    *,
    target_pose_detector_index: int,
    config: ActorAttributionConfig,
) -> HandAttributionResult:
    """Return one uniquely target-owned gesture or a fail-closed result."""

    if not hands:
        return HandAttributionResult(
            HandAttributionState.NO_HAND,
            None,
            0,
            0,
            None,
            ("actor_binding_no_hand",),
        )
    pose_by_index = {pose.detector_index: pose for pose in poses}
    if target_pose_detector_index not in pose_by_index:
        return HandAttributionResult(
            HandAttributionState.AMBIGUOUS,
            None,
            0,
            len(hands),
            None,
            ("target_pose_missing",),
        )

    assigned: list[tuple[GestureFrameEvidence, float]] = []
    rejected = 0
    ambiguous_assignment = False
    for hand in hands:
        if hand.wrist_x is None or hand.wrist_y is None:
            rejected += 1
            continue
        ranked = sorted(
            ((_pose_wrist_distance(hand, pose), pose.detector_index) for pose in poses),
            key=lambda item: item[0],
        )
        if not ranked or ranked[0][0] > config.maximum_hand_wrist_distance:
            rejected += 1
            continue
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < config.minimum_assignment_margin
        ):
            rejected += 1
            if ranked[0][1] == target_pose_detector_index or ranked[1][1] == target_pose_detector_index:
                ambiguous_assignment = True
            continue
        if ranked[0][1] == target_pose_detector_index:
            assigned.append((hand, ranked[0][0]))
        else:
            rejected += 1

    if ambiguous_assignment:
        return HandAttributionResult(
            HandAttributionState.AMBIGUOUS,
            None,
            len(assigned),
            rejected,
            None,
            ("hand_owner_assignment_ambiguous",),
        )
    if not assigned:
        return HandAttributionResult(
            HandAttributionState.NO_TARGET_HAND,
            None,
            0,
            rejected,
            None,
            ("no_hand_bound_to_target_player",),
        )

    meaningful = [
        item
        for item in assigned
        if item[0].gesture_label not in {None, "None"}
    ]
    labels = {item[0].gesture_label for item in meaningful}
    if len(labels) > 1:
        return HandAttributionResult(
            HandAttributionState.AMBIGUOUS,
            None,
            len(assigned),
            rejected,
            None,
            ("target_player_hands_gesture_conflict",),
        )
    candidates = meaningful or assigned
    selected, distance = max(
        candidates,
        key=lambda item: item[0].gesture_score if item[0].gesture_score is not None else -1.0,
    )
    confidence = max(
        0.0, 1.0 - distance / max(config.maximum_hand_wrist_distance, 1e-9)
    )
    flags = ["hand_bound_to_target_pose"]
    if len(assigned) > 1:
        flags.append("multiple_target_hands_agree")
    if rejected:
        flags.append("non_target_hands_rejected")
    return HandAttributionResult(
        HandAttributionState.BOUND,
        selected,
        len(assigned),
        rejected,
        confidence,
        tuple(flags),
    )
