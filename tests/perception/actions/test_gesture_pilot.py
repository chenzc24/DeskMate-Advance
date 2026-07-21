from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from poker_dealer.domain import (
    ActionEvidenceState,
    ColorSpace,
    FramePacket,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import HandEngine
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GestureFrameEvidence,
    GesturePilotConfig,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/perception/actions_laptop_pilot.json"
MODEL_PATH = ROOT / "models/assets/gesture_recognizer.task"


def load_config() -> GesturePilotConfig:
    return GesturePilotConfig.from_json(CONFIG_PATH)


def frame_evidence(
    timestamp_ms: int,
    *,
    label: str | None = "Thumb_Up",
    score: float | None = 0.95,
    hand_present: bool = True,
    in_roi: bool = True,
) -> GestureFrameEvidence:
    return GestureFrameEvidence(
        observed_at_ns=timestamp_ms * 1_000_000,
        hand_present=hand_present,
        hand_in_focus_roi=in_roi,
        gesture_label=label if hand_present else None,
        gesture_score=score if hand_present else None,
        centroid_x=0.5 if hand_present else None,
        centroid_y=0.5 if hand_present else None,
        handedness="Right" if hand_present else None,
        inference_latency_ms=12.5,
    )


def test_config_covers_five_actions_and_keeps_asset_offline() -> None:
    config = load_config()
    assert set(config.gesture_to_action.values()) == set(PlayerActionType)
    assert config.model.asset_path == MODEL_PATH
    assert not config.save_frames
    assert config.pilot_status == "development_feasibility_only"


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="ignored model asset unavailable")
def test_official_model_hash_and_blank_frame_inference() -> None:
    config = load_config()
    assert config.verify_model_asset() == config.model.sha256
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = FramePacket(
        0,
        1_000_000,
        "blank",
        0,
        320,
        240,
        ColorSpace.BGR,
        30.0,
        0,
        image,
    )
    with MediaPipeGestureAdapter(config) as adapter:
        result = adapter.recognize(frame)
    assert not result.hand_present
    assert result.gesture_label is None
    assert result.inference_latency_ms is not None


def test_unknown_no_hand_and_out_of_roi_never_become_candidates() -> None:
    adapter = GestureTemporalAdapter(load_config())
    context = ActionObservationContext("hand", 7, Seat.C)
    no_hand = adapter.process(
        frame_evidence(0, hand_present=False), context
    )
    outside = adapter.process(frame_evidence(10, in_roi=False), context)
    ignored = adapter.process(frame_evidence(20, label="Closed_Fist"), context)
    low_score = adapter.process(frame_evidence(30, score=0.2), context)
    assert no_hand.evidence_state is ActionEvidenceState.NO_ACTION
    assert outside.evidence_state is ActionEvidenceState.OUT_OF_ROI
    assert ignored.evidence_state is ActionEvidenceState.UNKNOWN
    assert low_score.evidence_state is ActionEvidenceState.UNKNOWN
    assert all(
        item.candidate_action is None
        for item in (no_hand, outside, ignored, low_score)
    )


@pytest.mark.parametrize(
    ("label", "action"),
    [
        ("Thumb_Down", PlayerActionType.FOLD),
        ("Open_Palm", PlayerActionType.CHECK),
        ("Thumb_Up", PlayerActionType.CALL),
        ("Victory", PlayerActionType.BET),
        ("Pointing_Up", PlayerActionType.RAISE),
    ],
)
def test_each_stable_label_maps_to_exactly_one_poker_action(
    label: str, action: PlayerActionType
) -> None:
    adapter = GestureTemporalAdapter(load_config())
    context = ActionObservationContext("hand", 0, Seat.A)
    observations = [
        adapter.process(frame_evidence(timestamp, label=label), context)
        for timestamp in (0, 70, 140, 210, 280)
    ]
    assert all(
        item.evidence_state is ActionEvidenceState.ACTION_START
        for item in observations[:-1]
    )
    candidate = observations[-1]
    assert candidate.evidence_state is ActionEvidenceState.CANDIDATE
    assert candidate.candidate_action is action
    assert candidate.stable_frames == 5
    assert candidate.stable_duration_ms == 280


def test_held_gesture_emits_once_until_release_and_cooldown() -> None:
    adapter = GestureTemporalAdapter(load_config())
    context = ActionObservationContext("hand", 0, Seat.A)
    first = None
    for timestamp in (0, 70, 140, 210, 280):
        first = adapter.process(frame_evidence(timestamp), context)
    assert first is not None
    assert first.evidence_state is ActionEvidenceState.CANDIDATE
    held = adapter.process(frame_evidence(350), context)
    assert held.evidence_state is ActionEvidenceState.NO_ACTION

    for timestamp in (400, 470, 540):
        adapter.process(frame_evidence(timestamp, hand_present=False), context)
    second = None
    for timestamp in (1400, 1470, 1540, 1610, 1680):
        second = adapter.process(frame_evidence(timestamp), context)
    assert second is not None
    assert second.evidence_state is ActionEvidenceState.CANDIDATE


def test_candidate_serializes_to_schema_and_game_retains_final_authority() -> None:
    config = load_config()
    adapter = GestureTemporalAdapter(config)
    engine = HandEngine.start("gesture-hand", Seat.A)
    context = ActionObservationContext(
        engine.state.hand_id,
        engine.state.state_version,
        engine.state.acting_seat,  # type: ignore[arg-type]
    )
    candidate = None
    for timestamp in (0, 70, 140, 210, 280):
        candidate = adapter.process(frame_evidence(timestamp), context)
    assert candidate is not None
    schema = json.loads(
        (ROOT / "configs/contracts/action_observation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(observation_to_dict(candidate))
    result = engine.apply_observation(candidate)
    assert result.accepted
    assert engine.state.pot_units == 5
    assert engine.state.acting_seat is Seat.A


def test_non_monotonic_evidence_is_rejected() -> None:
    adapter = GestureTemporalAdapter(load_config())
    context = ActionObservationContext("hand", 0, Seat.A)
    adapter.process(frame_evidence(100), context)
    with pytest.raises(ValueError, match="monotonic"):
        adapter.process(frame_evidence(99), context)
