import math

import numpy as np
import pytest

from deskmate_advance.features.ergonomics.stereo import (
    CameraCalibration,
    StereoCalibration,
    StereoCorrespondences,
    StereoEstimateState,
    StereoQualityConfig,
    StereoScreenDistanceEstimator,
)


def _camera() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=(
            (800.0, 0.0, 640.0),
            (0.0, 800.0, 360.0),
            (0.0, 0.0, 1.0),
        ),
        distortion_coefficients=(),
        image_size=(1280, 720),
    )


def _calibration() -> StereoCalibration:
    return StereoCalibration(
        rig_id="deskmate-stereo-fixture",
        calibration_sha256="1" * 64,
        left=_camera(),
        right=_camera(),
        rotation_right_from_left=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_right_from_left_m=(-0.12, 0.0, 0.0),
    )


def _project(points: np.ndarray, calibration: StereoCalibration) -> tuple[tuple[float, float], ...]:
    camera = calibration.left.matrix_array()
    pixels = (camera @ points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    return tuple(tuple(float(value) for value in row) for row in pixels)


def _matches(
    calibration: StereoCalibration,
    *,
    rotation_left_from_world: np.ndarray | None = None,
    translation_left_from_world: np.ndarray | None = None,
    synchronization_skew_ms: float = 0.5,
) -> StereoCorrespondences:
    rotation = (
        np.eye(3)
        if rotation_left_from_world is None
        else np.asarray(rotation_left_from_world, dtype=np.float64)
    )
    translation = (
        np.zeros(3)
        if translation_left_from_world is None
        else np.asarray(translation_left_from_world, dtype=np.float64)
    )
    screen_world = np.asarray(
        [
            (-0.35, -0.22, 2.0),
            (0.35, -0.22, 2.0),
            (0.35, 0.22, 2.0),
            (-0.35, 0.22, 2.0),
        ],
        dtype=np.float64,
    )
    face_world = np.asarray(
        [
            (-0.03, 0.05, 1.4),
            (0.03, 0.05, 1.4),
        ],
        dtype=np.float64,
    )
    screen_left = (rotation @ screen_world.T).T + translation
    face_left = (rotation @ face_world.T).T + translation
    right_rotation = calibration.rotation_array()
    right_translation = calibration.translation_array()
    screen_right = (right_rotation @ screen_left.T).T + right_translation
    face_right = (right_rotation @ face_left.T).T + right_translation
    return StereoCorrespondences(
        face_left_xy=_project(face_left, calibration),
        face_right_xy=_project(face_right, calibration),
        screen_left_xy=_project(screen_left, calibration),
        screen_right_xy=_project(screen_right, calibration),
        synchronization_skew_ms=synchronization_skew_ms,
    )


def test_exact_face_to_screen_plane_distance_is_metric() -> None:
    calibration = _calibration()
    estimator = StereoScreenDistanceEstimator(calibration)

    result = estimator.estimate(_matches(calibration))

    assert result.state is StereoEstimateState.VALID
    assert result.absolute_distance_claimed is True
    assert result.distance_m == pytest.approx(0.6, abs=1e-9)
    assert result.face_points_used == 2
    assert result.screen_points_used == 4
    assert result.screen_plane is not None
    assert result.screen_plane.rmse_m == pytest.approx(0.0, abs=1e-10)
    assert result.max_reprojection_error_px == pytest.approx(0.0, abs=1e-8)


def test_distance_is_invariant_to_mobile_rig_pose() -> None:
    calibration = _calibration()
    estimator = StereoScreenDistanceEstimator(calibration)
    angle = math.radians(7.0)
    rotation = np.asarray(
        [
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        ],
        dtype=np.float64,
    )
    moved = _matches(
        calibration,
        rotation_left_from_world=rotation,
        translation_left_from_world=np.asarray((0.08, -0.03, 0.05)),
    )

    result = estimator.estimate(moved)

    assert result.state is StereoEstimateState.VALID
    assert result.distance_m == pytest.approx(0.6, abs=1e-8)


def test_unsynchronized_pair_fails_closed_without_distance() -> None:
    calibration = _calibration()
    estimator = StereoScreenDistanceEstimator(
        calibration,
        StereoQualityConfig(max_sync_skew_ms=3.0),
    )

    result = estimator.estimate(
        _matches(calibration, synchronization_skew_ms=3.1)
    )

    assert result.state is StereoEstimateState.MISSING
    assert result.distance_m is None
    assert result.absolute_distance_claimed is False
    assert result.reason == "stereo_pair_not_synchronized"


def test_insufficient_screen_evidence_fails_closed() -> None:
    calibration = _calibration()
    original = _matches(calibration)
    matches = StereoCorrespondences(
        face_left_xy=original.face_left_xy,
        face_right_xy=original.face_right_xy,
        screen_left_xy=original.screen_left_xy[:2],
        screen_right_xy=original.screen_right_xy[:2],
        synchronization_skew_ms=original.synchronization_skew_ms,
    )

    result = StereoScreenDistanceEstimator(calibration).estimate(matches)

    assert result.state is StereoEstimateState.MISSING
    assert result.reason == "insufficient_screen_points"
    assert result.absolute_distance_claimed is False


def test_out_of_image_correspondence_is_not_accepted_as_metric_evidence() -> None:
    calibration = _calibration()
    original = _matches(calibration)
    matches = StereoCorrespondences(
        face_left_xy=original.face_left_xy,
        face_right_xy=original.face_right_xy,
        screen_left_xy=(*original.screen_left_xy[:3], (1280.0, 100.0)),
        screen_right_xy=original.screen_right_xy,
        synchronization_skew_ms=original.synchronization_skew_ms,
    )

    result = StereoScreenDistanceEstimator(calibration).estimate(matches)

    assert result.state is StereoEstimateState.MISSING
    assert result.reason == "screen_triangulation_unavailable"
    assert result.absolute_distance_claimed is False


def test_bad_epipolar_correspondences_fail_reprojection_quality_gate() -> None:
    calibration = _calibration()
    original = _matches(calibration)
    shifted_right = tuple((x, y + 20.0) for x, y in original.screen_right_xy)
    matches = StereoCorrespondences(
        face_left_xy=original.face_left_xy,
        face_right_xy=original.face_right_xy,
        screen_left_xy=original.screen_left_xy,
        screen_right_xy=shifted_right,
        synchronization_skew_ms=original.synchronization_skew_ms,
    )

    result = StereoScreenDistanceEstimator(calibration).estimate(matches)

    assert result.state is StereoEstimateState.MISSING
    assert result.reason == "screen_triangulation_unavailable"
    assert result.absolute_distance_claimed is False


def test_plane_fit_rejects_one_depth_outlier_and_keeps_four_inliers() -> None:
    calibration = _calibration()
    original = _matches(calibration)
    center_left = (640.0, 360.0)
    center_right_wrong_depth = (608.0, 360.0)
    matches = StereoCorrespondences(
        face_left_xy=original.face_left_xy,
        face_right_xy=original.face_right_xy,
        screen_left_xy=(*original.screen_left_xy, center_left),
        screen_right_xy=(*original.screen_right_xy, center_right_wrong_depth),
        synchronization_skew_ms=original.synchronization_skew_ms,
    )
    config = StereoQualityConfig(min_screen_inliers=4)

    result = StereoScreenDistanceEstimator(calibration, config).estimate(matches)

    assert result.state is StereoEstimateState.VALID
    assert result.distance_m == pytest.approx(0.6, abs=1e-8)
    assert result.screen_points_used == 4
    assert result.screen_plane is not None
    assert result.screen_plane.inlier_source_indices == (0, 1, 2, 3)


def test_invalid_stereo_calibration_is_rejected_before_runtime() -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        StereoCalibration(
            rig_id="invalid-rig",
            calibration_sha256="2" * 64,
            left=_camera(),
            right=_camera(),
            rotation_right_from_left=(
                (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            translation_right_from_left_m=(-0.12, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="baseline"):
        StereoCalibration(
            rig_id="invalid-rig",
            calibration_sha256="2" * 64,
            left=_camera(),
            right=_camera(),
            rotation_right_from_left=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            translation_right_from_left_m=(0.0, 0.0, 0.0),
        )


def test_correspondence_contract_rejects_left_right_count_drift() -> None:
    with pytest.raises(ValueError, match="face correspondence counts"):
        StereoCorrespondences(
            face_left_xy=((1.0, 2.0),),
            face_right_xy=(),
            screen_left_xy=(),
            screen_right_xy=(),
            synchronization_skew_ms=0.0,
        )
