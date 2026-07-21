from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poker_dealer.domain import ActionEvidenceState, ColorSpace, FramePacket, Seat
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GestureFrameEvidence,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    MultiSeatGesturePilotConfig,
    SeatRoiRouter,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/perception/actions_multiseat_laptop_pilot.json"
MODEL_PATH = ROOT / "models/assets/gesture_recognizer.task"


def load_config() -> MultiSeatGesturePilotConfig:
    return MultiSeatGesturePilotConfig.from_json(CONFIG_PATH)


def hand(
    x: float,
    y: float,
    *,
    timestamp_ms: int = 0,
    label: str = "Thumb_Up",
    score: float = 0.95,
) -> GestureFrameEvidence:
    return GestureFrameEvidence(
        observed_at_ns=timestamp_ms * 1_000_000,
        hand_present=True,
        hand_in_focus_roi=True,
        gesture_label=label,
        gesture_score=score,
        centroid_x=x,
        centroid_y=y,
        handedness="Right",
        inference_latency_ms=14.0,
    )


def test_multiseat_config_has_four_nonoverlapping_pilot_rois() -> None:
    config = load_config()
    assert set(config.seat_rois) == set(Seat)
    assert config.max_hands == 4
    assert config.gesture.model.num_hands == 4
    assert config.initial_focus_seat is Seat.A
    assert config.layout_status == "laptop_quadrant_pilot_not_target_geometry"


def test_router_assigns_clockwise_quadrants_and_rejects_center_gap() -> None:
    router = SeatRoiRouter(load_config().seat_rois)
    routed = router.route(
        (
            hand(0.25, 0.75),
            hand(0.25, 0.25),
            hand(0.75, 0.25),
            hand(0.75, 0.75),
            hand(0.50, 0.50),
        )
    )
    assert len(routed.assignments[Seat.A]) == 1
    assert len(routed.assignments[Seat.B]) == 1
    assert len(routed.assignments[Seat.C]) == 1
    assert len(routed.assignments[Seat.D]) == 1
    assert len(routed.unassigned) == 1
    assert not routed.ambiguous


def test_only_focused_seat_evidence_can_reach_temporal_candidate() -> None:
    config = load_config()
    router = SeatRoiRouter(config.seat_rois)
    adapter = GestureTemporalAdapter(config.gesture)
    context = ActionObservationContext("seat-hand", 8, Seat.A)
    observations = []
    for timestamp in (0, 70, 140, 210, 280):
        routed = router.route(
            (
                hand(0.25, 0.25, timestamp_ms=timestamp, label="Thumb_Down"),
                hand(0.25, 0.75, timestamp_ms=timestamp, label="Thumb_Up"),
            )
        )
        focused = router.focus_evidence(
            routed,
            Seat.A,
            observed_at_ns=timestamp * 1_000_000,
            inference_latency_ms=14.0,
        )
        observations.append(adapter.process(focused, context))
    assert observations[-1].evidence_state is ActionEvidenceState.CANDIDATE
    assert observations[-1].candidate_action.value == "call"  # type: ignore[union-attr]
    assert observations[-1].focus_seat is Seat.A


def test_hand_only_in_nonfocused_seat_is_no_action() -> None:
    config = load_config()
    router = SeatRoiRouter(config.seat_rois)
    routed = router.route((hand(0.25, 0.25),))
    focused = router.focus_evidence(
        routed,
        Seat.A,
        observed_at_ns=0,
        inference_latency_ms=14.0,
    )
    observation = GestureTemporalAdapter(config.gesture).process(
        focused, ActionObservationContext("seat-hand", 0, Seat.A)
    )
    assert observation.evidence_state is ActionEvidenceState.NO_ACTION
    assert observation.candidate_action is None
    assert "focus_seat_no_hand" in observation.quality_flags


def test_multiple_hands_in_one_seat_are_rejected_as_unknown() -> None:
    config = load_config()
    router = SeatRoiRouter(config.seat_rois)
    routed = router.route((hand(0.20, 0.70), hand(0.35, 0.80)))
    focused = router.focus_evidence(
        routed,
        Seat.A,
        observed_at_ns=0,
        inference_latency_ms=14.0,
    )
    observation = GestureTemporalAdapter(config.gesture).process(
        focused, ActionObservationContext("seat-hand", 0, Seat.A)
    )
    assert observation.evidence_state is ActionEvidenceState.UNKNOWN
    assert observation.candidate_action is None
    assert "multiple_hands_in_focus_seat" in observation.quality_flags


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="ignored model unavailable")
def test_four_hand_model_configuration_loads_and_blank_returns_no_hands() -> None:
    config = load_config()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = FramePacket(
        0,
        1_000_000,
        "blank-multiseat",
        0,
        320,
        240,
        ColorSpace.BGR,
        30.0,
        0,
        image,
    )
    with MediaPipeGestureAdapter(config.gesture) as adapter:
        assert adapter.recognize_all(frame) == ()
