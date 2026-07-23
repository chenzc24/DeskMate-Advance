from __future__ import annotations

import numpy as np
import pytest

import chip_live_value
import live_chip_yolo11
from chip_best_frame import BestFrameCandidate
from chip_live_value import (
    ChipValueObservation,
    _fuse_raw_color_and_digit,
    _raw_outer_ring_signature,
    _resolve_one_five,
    recognize_chip_value,
)
from chip_template_matcher import ColorMatch, TemplateMatch
from live_chip_yolo11 import _attach_values, _run_value_batch
from rectify_chip_images import EllipseEvidence


class _UnusedMatcher:
    def match_normalized_chip(self, image):  # pragma: no cover
        raise AssertionError("matcher must not run for an invalid crop")


def _ellipse(axes=(80.0, 60.0), aspect=0.75):
    return EllipseEvidence(
        center_xy=(50.0, 50.0),
        axes_wh=axes,
        angle_degrees=0.0,
        aspect_ratio=aspect,
        contour_fill_ratio=0.95,
        center_offset_ratio=0.01,
        quality=0.90,
    )


def _accepted_match(denomination):
    scores = {1: 0.70, 5: 0.15, 10: 0.10, 20: 0.05}
    if denomination == 5:
        scores = {1: 0.15, 5: 0.70, 10: 0.10, 20: 0.05}
    return TemplateMatch(
        denomination=denomination,
        accepted=True,
        best_score=0.70,
        margin=0.55,
        scores=scores,
        source_id="synthetic",
        rotation_degrees=0,
        latency_ms=1.0,
    )


def test_tiny_detection_remains_unknown():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = recognize_chip_value(_UnusedMatcher(), image, (10, 10, 18, 18))

    assert not result.accepted
    assert result.denomination is None
    assert result.rejection_reason == "crop_too_small"


def test_red_yellow_five_ring_corrects_digit_one_candidate():
    normalized = np.full((384, 384, 3), 220, dtype=np.uint8)
    cv_center = (192, 192)
    import cv2

    cv2.circle(normalized, cv_center, 178, (0, 0, 255), 48)
    cv2.ellipse(
        normalized,
        cv_center,
        (178, 178),
        0,
        0,
        180,
        (0, 255, 255),
        48,
    )

    denomination, accepted, _, decision, rejection = _resolve_one_five(
        _accepted_match(1), normalized
    )

    assert accepted
    assert denomination == 5
    assert decision == "five_ring_override"
    assert rejection is None


def test_five_without_fixed_red_yellow_ring_is_rejected():
    normalized = np.full((384, 384, 3), (160, 160, 120), dtype=np.uint8)

    denomination, accepted, _, decision, rejection = _resolve_one_five(
        _accepted_match(5), normalized
    )

    assert not accepted
    assert denomination is None
    assert decision is None
    assert rejection == "five_ring_conflict"


def test_small_projected_minor_axis_is_reported_as_too_far(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(
        chip_live_value,
        "_grabcut_chip_mask",
        lambda crop: np.full(crop.shape[:2], 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        chip_live_value, "_fit_top_ellipse", lambda mask: (_ellipse(), None)
    )
    monkeypatch.setattr(
        chip_live_value,
        "_derive_top_ellipse_from_inlay",
        lambda crop, preliminary: (_ellipse((60.0, 30.0), 0.50), None),
    )

    result = recognize_chip_value(
        _UnusedMatcher(),
        image,
        (20, 20, 100, 100),
        minimum_minor_axis_px=42.0,
    )

    assert not result.accepted
    assert result.rejection_reason == "too_far"
    assert result.ellipse_minor_axis_px == 30.0


def test_flat_projection_is_rejected_before_template_match(monkeypatch):
    image = np.zeros((140, 140, 3), dtype=np.uint8)
    monkeypatch.setattr(
        chip_live_value,
        "_grabcut_chip_mask",
        lambda crop: np.full(crop.shape[:2], 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        chip_live_value, "_fit_top_ellipse", lambda mask: (_ellipse(), None)
    )
    monkeypatch.setattr(
        chip_live_value,
        "_derive_top_ellipse_from_inlay",
        lambda crop, preliminary: (_ellipse((100.0, 35.0), 0.35), None),
    )

    result = recognize_chip_value(
        _UnusedMatcher(),
        image,
        (20, 20, 120, 120),
        minimum_minor_axis_px=30.0,
        minimum_aspect_ratio=0.38,
    )

    assert not result.accepted
    assert result.rejection_reason == "too_flat"
    assert result.ellipse_aspect_ratio == 0.35


@pytest.mark.parametrize(
    ("image", "bbox"),
    [
        (np.empty((0, 0, 3), dtype=np.uint8), (0, 0, 20, 20)),
        (np.zeros((40, 40), dtype=np.uint8), (0, 0, 20, 20)),
        (np.zeros((40, 40, 3), dtype=np.uint8), (0, 0, 20)),
    ],
)
def test_invalid_inputs_are_rejected(image, bbox):
    with pytest.raises(ValueError):
        recognize_chip_value(_UnusedMatcher(), image, bbox)


def test_live_detection_receives_matching_value_evidence():
    detections = [{"bbox_xyxy": [20, 20, 80, 80]}]
    observation = ChipValueObservation(
        bbox_xyxy=(22, 22, 78, 78),
        denomination=20,
        accepted=True,
        score=0.91,
        margin=0.40,
        ellipse_quality=0.88,
        ellipse_aspect_ratio=0.75,
        ellipse_minor_axis_px=70.0,
        latency_ms=70.0,
        decision_reason="five_ring_confirmed",
        rejection_reason=None,
    )

    _attach_values(detections, (observation,))

    assert detections[0]["denomination"] == 20
    assert detections[0]["value_score"] == 0.91
    assert detections[0]["value_margin"] == 0.40
    assert detections[0]["value_decision_reason"] == "five_ring_confirmed"
    assert detections[0]["value_rejection_reason"] is None


def test_live_detection_without_matching_evidence_stays_unknown():
    detections = [{"bbox_xyxy": [20, 20, 80, 80]}]

    _attach_values(detections, ())

    assert detections[0]["denomination"] is None
    assert detections[0]["value_rejection_reason"] == "no_fresh_match"


def test_raw_ring_signature_is_measured_on_ellipse_before_warp():
    crop = np.full((120, 140, 3), (40, 50, 210), dtype=np.uint8)
    signature = _raw_outer_ring_signature(
        crop,
        _ellipse(axes=(100.0, 60.0), aspect=0.60),
    )

    assert signature.shape == (12,)
    assert np.all(np.isfinite(signature))
    assert signature[1] > 100


def test_raw_ring_resolves_one_five_digit_conflict_in_favour_of_five():
    color = ColorMatch(
        denomination=5,
        accepted=True,
        best_score=0.82,
        margin=0.55,
        scores={1: 0.12, 5: 0.82, 10: 0.03, 20: 0.03},
        distances={1: 2.0, 5: 0.1, 10: 4.0, 20: 4.0},
    )
    digit = _accepted_match(1)

    value, accepted, _, _, reason, rejection = _fuse_raw_color_and_digit(
        color, digit
    )

    assert accepted
    assert value == 5
    assert reason == "raw_ring_one_five_override"
    assert rejection is None


def test_value_batch_preserves_track_and_selected_source_frame(monkeypatch):
    observation = ChipValueObservation(
        bbox_xyxy=(5, 5, 45, 45),
        denomination=5,
        accepted=True,
        score=0.9,
        margin=0.4,
        ellipse_quality=0.9,
        ellipse_aspect_ratio=0.8,
        ellipse_minor_axis_px=55.0,
        latency_ms=2.0,
        decision_reason="colour_digit_agree",
        rejection_reason=None,
    )
    monkeypatch.setattr(
        live_chip_yolo11,
        "recognize_chip_value",
        lambda *args, **kwargs: observation,
    )
    candidate = BestFrameCandidate(
        track_id=7,
        source_frame=12,
        image=np.zeros((60, 60, 3), dtype=np.uint8),
        local_bbox_xyxy=(5, 5, 45, 45),
        source_bbox_xyxy=(100, 110, 140, 150),
        quality_score=0.88,
        sharpness_score=0.75,
        glare_ratio=0.01,
    )

    results, _ = _run_value_batch(
        object(),
        (candidate,),
        minimum_minor_axis_px=42.0,
        minimum_aspect_ratio=0.38,
    )

    assert results[0].track_id == 7
    assert results[0].source_frame == 12
    assert results[0].bbox_xyxy == (100, 110, 140, 150)
    assert results[0].best_frame_quality == 0.88


def test_track_id_attaches_value_even_after_box_moves():
    detections = [{"track_id": 7, "bbox_xyxy": [100, 100, 150, 150]}]
    observation = ChipValueObservation(
        bbox_xyxy=(0, 0, 40, 40),
        denomination=5,
        accepted=True,
        score=0.91,
        margin=0.40,
        ellipse_quality=0.88,
        ellipse_aspect_ratio=0.75,
        ellipse_minor_axis_px=70.0,
        latency_ms=70.0,
        decision_reason="colour_digit_agree",
        rejection_reason=None,
        track_id=7,
        source_frame=22,
    )

    _attach_values(detections, (observation,))

    assert detections[0]["denomination"] == 5
    assert detections[0]["value_source_frame"] == 22
