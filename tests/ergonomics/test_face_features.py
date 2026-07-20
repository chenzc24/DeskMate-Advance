import math

import numpy as np
import pytest

from deskmate_advance.features.ergonomics import FaceFeatureExtractor
from deskmate_advance.perception.ergonomics import (
    BlendshapeScore,
    FaceObservation,
    Landmark3D,
    ObservationContext,
    ObservationState,
)


def _context() -> ObservationContext:
    return ObservationContext(
        source_id="fixture",
        sequence_id=3,
        captured_at_ns=100_000_000,
        model_id="face",
        inference_ms=2.0,
    )


def _valid_observation() -> FaceObservation:
    points = [Landmark3D(0.2, 0.3, 0.0) for _ in range(478)]
    points[1] = Landmark3D(0.8, 0.9, 0.0)
    angle = math.radians(30)
    matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0, 1],
            [math.sin(angle), math.cos(angle), 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return FaceObservation(
        context=_context(),
        state=ObservationState.VALID,
        landmarks=tuple(points),
        blendshapes=(
            BlendshapeScore("eyeBlinkLeft", 0.8),
            BlendshapeScore("eye_blink_right", 0.6),
        ),
        transformation_matrix=tuple(tuple(row) for row in matrix),
    )


def test_face_features_extract_proxy_geometry_rotation_and_blinks() -> None:
    features = FaceFeatureExtractor().extract(_valid_observation())

    assert features.state is ObservationState.VALID
    assert features.face_center_xy == pytest.approx((0.5, 0.6))
    assert features.face_bbox_width_ratio == pytest.approx(0.6)
    assert features.face_bbox_height_ratio == pytest.approx(0.6)
    assert features.face_bbox_area_ratio == pytest.approx(0.36)
    assert features.raw_rotation_xyz_deg == pytest.approx((0.0, 0.0, 30.0))
    assert features.raw_translation_xyz == pytest.approx((1.0, 2.0, 3.0))
    assert features.eye_blink_left == pytest.approx(0.8)
    assert features.eye_blink_right == pytest.approx(0.6)
    assert features.eye_blink_mean == pytest.approx(0.7)


def test_face_features_preserve_missing_state() -> None:
    observation = FaceObservation(
        context=_context(),
        state=ObservationState.MISSING,
        reason="face_not_detected",
    )

    features = FaceFeatureExtractor().extract(observation)

    assert features.state is ObservationState.MISSING
    assert features.face_bbox_area_ratio is None
    assert features.eye_blink_mean is None
    assert features.reason == "face_not_detected"


def test_face_features_do_not_claim_rotation_for_invalid_matrix() -> None:
    observation = _valid_observation()
    observation = FaceObservation(
        context=observation.context,
        state=observation.state,
        landmarks=observation.landmarks,
        blendshapes=observation.blendshapes,
        transformation_matrix=((1.0, 0.0),),
    )

    features = FaceFeatureExtractor().extract(observation)

    assert features.state is ObservationState.VALID
    assert features.raw_rotation_xyz_deg is None
    assert features.raw_translation_xyz is None
    assert features.rotation_state is ObservationState.ERROR
    assert features.rotation_reason == "invalid_transformation_matrix"


def test_face_features_track_valid_eye_time_and_partial_blink_missing() -> None:
    extractor = FaceFeatureExtractor()
    first = extractor.extract(_valid_observation())
    source = _valid_observation()
    second = extractor.extract(
        FaceObservation(
            context=ObservationContext(
                source_id=source.context.source_id,
                sequence_id=4,
                captured_at_ns=150_000_000,
                model_id=source.context.model_id,
                inference_ms=2.0,
            ),
            state=source.state,
            landmarks=source.landmarks,
            blendshapes=(BlendshapeScore("eyeBlinkLeft", 0.9),),
            transformation_matrix=source.transformation_matrix,
        )
    )

    assert first.blink_state is ObservationState.VALID
    assert first.valid_eye_dt_ns is None
    assert second.dt_ns == 50_000_000
    assert second.blink_state is ObservationState.MISSING
    assert second.eye_blink_left == pytest.approx(0.9)
    assert second.eye_blink_right is None
    assert second.eye_blink_mean is None
    assert second.valid_eye_dt_ns is None
