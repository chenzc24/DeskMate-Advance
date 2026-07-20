import math

import pytest

from deskmate_advance.features.ergonomics import (
    PoseFeatureConfig,
    PoseFeatureExtractor,
)
from deskmate_advance.perception.ergonomics import (
    Landmark3D,
    ObservationContext,
    ObservationState,
    PoseObservation,
)


def _observation(
    *,
    captured_at_ns: int = 100_000_000,
    wrist_delta_x: float = 0.0,
    low_visibility_index: int | None = None,
    dropped_before: int = 0,
) -> PoseObservation:
    points = [Landmark3D(0.5, 0.5, 0.0, 1.0, 1.0) for _ in range(33)]
    points[0] = Landmark3D(0.5, 0.2, -0.1, 1.0, 1.0)
    points[11] = Landmark3D(0.4, 0.4, 0.0, 1.0, 1.0)
    points[12] = Landmark3D(0.6, 0.4, 0.0, 1.0, 1.0)
    points[15] = Landmark3D(0.3 + wrist_delta_x, 0.6, 0.0, 1.0, 1.0)
    points[23] = Landmark3D(0.45, 0.7, 0.0, 1.0, 1.0)
    points[24] = Landmark3D(0.55, 0.7, 0.0, 1.0, 1.0)
    if low_visibility_index is not None:
        old = points[low_visibility_index]
        points[low_visibility_index] = Landmark3D(
            old.x, old.y, old.z, 0.1, old.presence
        )
    return PoseObservation(
        context=ObservationContext(
            source_id="fixture",
            sequence_id=0,
            captured_at_ns=captured_at_ns,
            model_id="pose",
            inference_ms=1.0,
            dropped_before=dropped_before,
        ),
        state=ObservationState.VALID,
        landmarks=tuple(points),
    )


def test_pose_features_normalize_geometry_and_masks() -> None:
    extractor = PoseFeatureExtractor(PoseFeatureConfig(min_visibility=0.5))

    features = extractor.extract(_observation(low_visibility_index=13))

    assert features.state is ObservationState.VALID
    assert features.normalization_scale == pytest.approx(0.2)
    assert features.shoulder_tilt_deg == pytest.approx(0.0)
    assert features.torso_lean_from_vertical_deg == pytest.approx(0.0)
    assert features.nose_offset_from_shoulders == pytest.approx((0.0, -1.0, -0.5))
    assert features.missing_mask[13] is True
    assert features.normalized_landmarks[13] is None
    assert features.valid_landmark_fraction == pytest.approx(32 / 33)


def test_shoulder_tilt_is_orientation_invariant_to_image_left_right_order() -> None:
    observation = _observation()
    points = list(observation.landmarks)
    points[11], points[12] = points[12], points[11]
    reversed_shoulders = PoseObservation(
        context=observation.context,
        state=observation.state,
        landmarks=tuple(points),
    )

    features = PoseFeatureExtractor().extract(reversed_shoulders)

    assert features.shoulder_tilt_deg == pytest.approx(0.0)


def test_pose_motion_uses_true_time_and_common_valid_points() -> None:
    extractor = PoseFeatureExtractor()
    first = extractor.extract(_observation(captured_at_ns=100_000_000))
    second = extractor.extract(
        _observation(captured_at_ns=200_000_000, wrist_delta_x=0.02)
    )

    assert first.upper_body_motion_per_second is None
    assert second.upper_body_motion_per_second is not None
    assert second.upper_body_motion_per_second > 0


def test_pose_motion_rate_changes_with_elapsed_time() -> None:
    fast = PoseFeatureExtractor()
    fast.extract(_observation(captured_at_ns=100_000_000))
    fast_result = fast.extract(
        _observation(captured_at_ns=200_000_000, wrist_delta_x=0.02)
    )
    slow = PoseFeatureExtractor()
    slow.extract(_observation(captured_at_ns=100_000_000))
    slow_result = slow.extract(
        _observation(captured_at_ns=300_000_000, wrist_delta_x=0.02)
    )

    assert fast_result.upper_body_motion_per_second == pytest.approx(
        2 * slow_result.upper_body_motion_per_second
    )


def test_pose_normalization_is_translation_and_scale_invariant() -> None:
    original = _observation()
    transformed_points = tuple(
        Landmark3D(
            x=point.x * 1.5 + 0.1,
            y=point.y * 1.5 - 0.2,
            z=point.z * 1.5 + 0.05,
            visibility=point.visibility,
            presence=point.presence,
        )
        for point in original.landmarks
    )
    transformed = PoseObservation(
        context=original.context,
        state=original.state,
        landmarks=transformed_points,
    )

    first = PoseFeatureExtractor().extract(original)
    second = PoseFeatureExtractor().extract(transformed)

    for first_point, second_point in zip(
        first.normalized_landmarks,
        second.normalized_landmarks,
        strict=True,
    ):
        assert second_point == pytest.approx(first_point)


def test_pose_motion_does_not_bridge_long_or_missing_gaps() -> None:
    extractor = PoseFeatureExtractor(PoseFeatureConfig(max_motion_gap_ms=100))
    extractor.extract(_observation(captured_at_ns=100_000_000))
    late = extractor.extract(_observation(captured_at_ns=300_000_000))
    missing = extractor.extract(
        PoseObservation(
            context=ObservationContext(
                source_id="fixture",
                sequence_id=1,
                captured_at_ns=350_000_000,
                model_id="pose",
                inference_ms=1.0,
            ),
            state=ObservationState.MISSING,
            reason="pose_not_detected",
        )
    )
    after_missing = extractor.extract(_observation(captured_at_ns=400_000_000))

    assert late.upper_body_motion_per_second is None
    assert missing.state is ObservationState.MISSING
    assert all(missing.missing_mask)
    assert after_missing.upper_body_motion_per_second is None


def test_pose_features_expose_dropped_and_temporal_gap() -> None:
    extractor = PoseFeatureExtractor(PoseFeatureConfig(max_motion_gap_ms=100))
    first = extractor.extract(_observation(captured_at_ns=100_000_000))
    second = extractor.extract(
        _observation(
            captured_at_ns=250_000_000,
            dropped_before=3,
        )
    )

    assert first.dt_ns is None
    assert first.temporal_gap is False
    assert second.dt_ns == 150_000_000
    assert second.dropped_before == 3
    assert second.temporal_gap is True


def test_pose_feature_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        PoseFeatureConfig(min_visibility=math.inf)
