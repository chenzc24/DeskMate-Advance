from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.features.ergonomics.face import FaceFeatures
from deskmate_advance.features.ergonomics.live import LiveSnapshot
from deskmate_advance.features.ergonomics.pose import PoseFeatures
from deskmate_advance.perception.ergonomics import (
    LuminanceObservation,
    ObservationState,
)
from deskmate_advance.temporal.ergonomics.calibration import (
    CalibrationCollector,
    CalibrationConfig,
    CalibrationState,
)


def _frame(
    timestamp_ms: int,
    *,
    source_id: str = "camera",
    device_index: int = 0,
) -> FramePacket:
    return FramePacket(
        sequence_id=timestamp_ms,
        captured_at_ns=timestamp_ms * 1_000_000,
        source_id=source_id,
        device_index=device_index,
        width=2,
        height=2,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _pose(
    timestamp_ms: int,
    *,
    source_id: str = "camera",
    state: ObservationState = ObservationState.VALID,
    shoulder: float | None = 1.0,
    torso: float | None = 2.0,
) -> PoseFeatures:
    return PoseFeatures(
        source_id=source_id,
        sequence_id=timestamp_ms,
        captured_at_ns=timestamp_ms * 1_000_000,
        state=state,
        model_id="pose",
        model_version="test",
        asset_sha256=None,
        config_sha256=None,
        dropped_before=0,
        dt_ns=None,
        temporal_gap=False,
        missing_mask=(),
        normalized_landmarks=(),
        valid_landmark_fraction=1.0,
        normalization_scale=1.0,
        shoulder_tilt_deg=shoulder,
        torso_lean_from_vertical_deg=torso,
        nose_offset_from_shoulders=None,
        upper_body_motion_per_second=None,
    )


def _face(
    timestamp_ms: int,
    *,
    source_id: str = "camera",
    state: ObservationState = ObservationState.VALID,
    area: float | None = 0.1,
    rotation: tuple[float, float, float] | None = (3.0, 4.0, 0.0),
    blink: float | None = 0.1,
) -> FaceFeatures:
    component_state = (
        ObservationState.VALID if state is ObservationState.VALID else state
    )
    return FaceFeatures(
        source_id=source_id,
        sequence_id=timestamp_ms,
        captured_at_ns=timestamp_ms * 1_000_000,
        state=state,
        model_id="face",
        model_version="test",
        asset_sha256=None,
        config_sha256=None,
        dropped_before=0,
        dt_ns=None,
        geometry_state=component_state,
        rotation_state=component_state,
        blink_state=component_state,
        face_center_xy=(0.5, 0.5),
        face_bbox_width_ratio=None,
        face_bbox_height_ratio=None,
        face_bbox_area_ratio=area,
        raw_rotation_xyz_deg=rotation,
        raw_translation_xyz=None,
        eye_blink_left=blink,
        eye_blink_right=blink,
        eye_blink_mean=blink,
        valid_eye_dt_ns=None,
    )


def _luminance(
    timestamp_ms: int,
    *,
    source_id: str = "camera",
    state: ObservationState = ObservationState.VALID,
    mean: float | None = 80.0,
    p90: float | None = 120.0,
) -> LuminanceObservation:
    return LuminanceObservation(
        source_id=source_id,
        sequence_id=timestamp_ms,
        captured_at_ns=timestamp_ms * 1_000_000,
        state=state,
        mean=mean,
        median=mean,
        p10=mean,
        p90=p90,
    )


def _snapshot(
    timestamp_ms: int,
    *,
    source_id: str = "camera",
    device_index: int = 0,
    pose: PoseFeatures | None = None,
    face: FaceFeatures | None = None,
    luminance: LuminanceObservation | None = None,
    pose_ran: bool = True,
    face_ran: bool = True,
    luminance_ran: bool = True,
    pose_stale: bool = False,
    face_stale: bool = False,
) -> LiveSnapshot:
    frame = _frame(
        timestamp_ms,
        source_id=source_id,
        device_index=device_index,
    )
    return LiveSnapshot(
        frame=frame,
        luminance=luminance or _luminance(timestamp_ms, source_id=source_id),
        luminance_ran=luminance_ran,
        luminance_age_ms=0.0,
        pose_observation=None,
        pose_features=pose or _pose(timestamp_ms, source_id=source_id),
        pose_ran=pose_ran,
        pose_age_ms=0.0,
        pose_stale=pose_stale,
        face_observation=None,
        face_features=face or _face(timestamp_ms, source_id=source_id),
        face_ran=face_ran,
        face_age_ms=0.0,
        face_stale=face_stale,
    )


def test_calibration_builds_immutable_robust_scalar_profile() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=40,
            minimum_pose_samples=5,
            minimum_face_samples=5,
            minimum_luminance_samples=5,
        )
    )
    shoulders = (1.0, 2.0, 100.0, 3.0, 4.0)
    torsos = (5.0, None, 7.0, None, 9.0)
    areas = (0.08, 0.09, 0.9, 0.10, 0.11)
    rotations_x = (1.0, 2.0, 90.0, 3.0, 4.0)
    rotations_y = (-1.0, -2.0, -90.0, -3.0, -4.0)
    blinks = (0.10, 0.20, 0.95, 0.30, 0.40)
    means = (70.0, 75.0, 250.0, 80.0, 85.0)

    for index, timestamp_ms in enumerate((0, 10, 20, 30, 40)):
        progress = collector.update(
            _snapshot(
                timestamp_ms,
                pose=_pose(
                    timestamp_ms,
                    shoulder=shoulders[index],
                    torso=torsos[index],
                ),
                face=_face(
                    timestamp_ms,
                    area=areas[index],
                    rotation=(rotations_x[index], rotations_y[index], 0.0),
                    blink=blinks[index],
                ),
                luminance=_luminance(
                    timestamp_ms,
                    mean=means[index],
                    p90=means[index] + 30,
                ),
            )
        )

    profile = collector.profile
    assert progress.state is CalibrationState.READY
    assert progress.ready is True
    assert progress.not_ready_reasons == ()
    assert profile is not None
    assert profile.source_id == "camera"
    assert profile.device_index == 0
    assert profile.window_started_at_ns == 0
    assert profile.window_ended_at_ns == 40_000_000
    assert profile.pose_samples == 5
    assert profile.torso_lean_samples == 3
    assert profile.face_samples == 5
    assert profile.eye_open_samples == 5
    assert profile.luminance_samples == 5
    assert profile.shoulder_tilt_deg == pytest.approx(3.0)
    assert profile.torso_lean_from_vertical_deg == pytest.approx(7.0)
    assert profile.face_bbox_area_ratio == pytest.approx(0.10)
    assert profile.head_rotation_x_deg == pytest.approx(3.0)
    assert profile.head_rotation_y_deg == pytest.approx(-3.0)
    assert profile.eye_open_score == pytest.approx(0.70)
    assert profile.mean_luminance == pytest.approx(80.0)
    assert profile.p90_luminance == pytest.approx(110.0)
    with pytest.raises(FrozenInstanceError):
        profile.shoulder_tilt_deg = 0.0  # type: ignore[misc]


def test_cached_stale_and_invalid_evidence_is_not_counted_as_negative() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=20,
            minimum_pose_samples=2,
            minimum_face_samples=2,
            minimum_luminance_samples=2,
        )
    )
    collector.update(_snapshot(0))
    collector.update(
        _snapshot(
            5,
            pose=_pose(0),
            face=_face(0),
            luminance=_luminance(0),
            pose_ran=False,
            face_ran=False,
            luminance_ran=False,
        )
    )
    collector.update(
        _snapshot(
            10,
            pose=_pose(10, state=ObservationState.MISSING, shoulder=None),
            face=_face(10, state=ObservationState.MISSING, area=None, rotation=None),
            luminance=_luminance(
                10,
                state=ObservationState.MISSING,
                mean=None,
                p90=None,
            ),
            pose_stale=True,
            face_stale=True,
        )
    )
    progress = collector.update(_snapshot(20))

    assert progress.ready is True
    assert progress.counts.pose == 2
    assert progress.counts.face == 2
    assert progress.counts.luminance == 2
    assert collector.profile is not None


def test_closed_insufficient_window_reports_reasons_and_requires_reset() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=10,
            minimum_pose_samples=2,
            minimum_face_samples=2,
            minimum_luminance_samples=2,
        )
    )
    collector.update(_snapshot(0))
    progress = collector.update(
        _snapshot(
            10,
            pose=_pose(10, shoulder=None, torso=None),
            face=_face(10, area=None, rotation=None, blink=None),
            luminance=_luminance(10, mean=None, p90=None),
        )
    )

    assert progress.state is CalibrationState.NOT_READY
    assert progress.ready is False
    assert progress.not_ready_reasons == (
        "pose_samples:1/2",
        "face_samples:1/2",
        "luminance_samples:1/2",
    )
    collector.update(_snapshot(20))
    assert collector.progress.counts.pose == 1
    assert collector.profile is None

    collector.reset()
    assert collector.progress.state is CalibrationState.COLLECTING
    assert collector.progress.counts.pose == 0
    assert "calibration_duration_incomplete" in collector.progress.not_ready_reasons


def test_calibration_rejects_non_increasing_time_or_camera_change() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=10,
            minimum_pose_samples=1,
            minimum_face_samples=1,
            minimum_luminance_samples=1,
        )
    )
    collector.update(_snapshot(0))
    with pytest.raises(ValueError, match="increase strictly"):
        collector.update(_snapshot(0))
    with pytest.raises(ValueError, match="one source and camera"):
        collector.update(_snapshot(1, device_index=1))
    with pytest.raises(ValueError, match="one source and camera"):
        collector.update(_snapshot(1, source_id="another-camera"))


def test_calibration_rejects_mismatched_fresh_feature_timestamp() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=10,
            minimum_pose_samples=1,
            minimum_face_samples=1,
            minimum_luminance_samples=1,
        )
    )

    with pytest.raises(ValueError, match="fresh pose timestamp"):
        collector.update(_snapshot(1, pose=_pose(0)))
    assert collector.progress.counts.pose == 0

    with pytest.raises(ValueError, match="fresh face timestamp"):
        collector.update(_snapshot(1, face=_face(0)))
    assert collector.progress.counts.pose == 0


@pytest.mark.parametrize(
    "config",
    (
        CalibrationConfig(duration_ms=1),
        CalibrationConfig(minimum_pose_samples=1),
        CalibrationConfig(minimum_face_samples=1),
        CalibrationConfig(minimum_luminance_samples=1),
    ),
)
def test_valid_calibration_configs_construct(config: CalibrationConfig) -> None:
    assert config.duration_ms > 0


def test_calibration_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        CalibrationConfig(duration_ms=0)
    with pytest.raises(ValueError, match="minimum_pose_samples"):
        CalibrationConfig(minimum_pose_samples=0)
    with pytest.raises(ValueError, match="maximum_samples_per_metric"):
        CalibrationConfig(minimum_face_samples=5, maximum_samples_per_metric=4)
    with pytest.raises(TypeError, match="duration_ms"):
        CalibrationConfig(duration_ms=True)


def test_calibration_rejects_a_long_unobserved_interval() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=1_000,
            minimum_pose_samples=1,
            minimum_face_samples=1,
            minimum_luminance_samples=1,
            maximum_snapshot_gap_ms=50,
        )
    )
    collector.update(_snapshot(0))
    progress = collector.update(_snapshot(100))

    assert progress.state is CalibrationState.NOT_READY
    assert progress.elapsed_ms == pytest.approx(100)
    assert "calibration_duration_incomplete" in progress.not_ready_reasons
    assert any(
        reason.startswith("evidence_discontinuity:snapshot_gap_ms")
        for reason in progress.not_ready_reasons
    )
    assert collector.profile is None


def test_explicit_camera_gap_freezes_in_progress_calibration() -> None:
    collector = CalibrationCollector(
        CalibrationConfig(
            duration_ms=1_000,
            minimum_pose_samples=1,
            minimum_face_samples=1,
            minimum_luminance_samples=1,
        )
    )
    collector.update(_snapshot(0))
    progress = collector.mark_evidence_gap(
        10_000_000,
        reason="capture_read_failed",
    )

    assert progress.state is CalibrationState.NOT_READY
    assert "evidence_discontinuity:capture_read_failed" in progress.not_ready_reasons
