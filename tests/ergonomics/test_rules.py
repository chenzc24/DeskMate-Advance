from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.features.ergonomics import FaceFeatures, LiveSnapshot, PoseFeatures
from deskmate_advance.perception.ergonomics import (
    AudioLevelObservation,
    LuminanceObservation,
    ObservationState,
)
from deskmate_advance.temporal.ergonomics.calibration import CalibrationProfile
from deskmate_advance.temporal.ergonomics.core import SemanticState, TemporalPhase
from deskmate_advance.temporal.ergonomics.rules import (
    ErgonomicsEventConfig,
    ErgonomicsRuleEngine,
)


CONFIG_PATH = Path("configs/ergonomics/events.json")


def _config() -> ErgonomicsEventConfig:
    data = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    for name in ErgonomicsRuleEngine.EVENT_NAMES:
        section = data[name]
        section["enter_duration_ms"] = 0
        section["exit_duration_ms"] = 0
        section["cooldown_ms"] = 0
    data["low_blink_rate"]["window_ms"] = 1_000
    data["low_blink_rate"]["minimum_valid_ms"] = 100
    return ErgonomicsEventConfig.from_mapping(data)


def _profile() -> CalibrationProfile:
    return CalibrationProfile(
        source_id="camera",
        device_index=0,
        window_started_at_ns=0,
        window_ended_at_ns=500_000_000,
        pose_samples=20,
        torso_lean_samples=20,
        face_samples=20,
        eye_open_samples=20,
        luminance_samples=5,
        shoulder_tilt_deg=0.0,
        torso_lean_from_vertical_deg=0.0,
        face_bbox_area_ratio=0.1,
        head_rotation_x_deg=0.0,
        head_rotation_y_deg=0.0,
        eye_open_score=0.9,
        mean_luminance=100.0,
        p90_luminance=150.0,
    )


def _pose(
    timestamp_ns: int,
    *,
    motion: float | None = 0.01,
    shoulder: float | None = 20.0,
    torso: float | None = 20.0,
) -> PoseFeatures:
    return PoseFeatures(
        source_id="camera",
        sequence_id=0,
        captured_at_ns=timestamp_ns,
        state=ObservationState.VALID,
        model_id="pose",
        model_version="test",
        asset_sha256=None,
        config_sha256=None,
        dropped_before=0,
        dt_ns=100_000_000,
        temporal_gap=False,
        missing_mask=(False,) * 33,
        normalized_landmarks=(None,) * 33,
        valid_landmark_fraction=1.0,
        normalization_scale=1.0,
        shoulder_tilt_deg=shoulder,
        torso_lean_from_vertical_deg=torso,
        nose_offset_from_shoulders=None,
        upper_body_motion_per_second=motion,
    )


def _face(
    timestamp_ns: int,
    *,
    area: float = 0.2,
    rotation: tuple[float, float, float] = (30.0, 0.0, 0.0),
    blink: float = 0.1,
) -> FaceFeatures:
    return FaceFeatures(
        source_id="camera",
        sequence_id=0,
        captured_at_ns=timestamp_ns,
        state=ObservationState.VALID,
        model_id="face",
        model_version="test",
        asset_sha256=None,
        config_sha256=None,
        dropped_before=0,
        dt_ns=100_000_000,
        geometry_state=ObservationState.VALID,
        rotation_state=ObservationState.VALID,
        blink_state=ObservationState.VALID,
        face_center_xy=(0.5, 0.5),
        face_bbox_width_ratio=0.4,
        face_bbox_height_ratio=0.5,
        face_bbox_area_ratio=area,
        raw_rotation_xyz_deg=rotation,
        raw_translation_xyz=(0.0, 0.0, 0.0),
        eye_blink_left=blink,
        eye_blink_right=blink,
        eye_blink_mean=blink,
        valid_eye_dt_ns=100_000_000,
    )


def _snapshot(
    timestamp_ns: int,
    *,
    pose: PoseFeatures | None = None,
    face: FaceFeatures | None = None,
    pose_stale: bool = False,
    face_stale: bool = False,
    luminance_mean: float = 20.0,
    luminance_p90: float = 20.0,
    luminance_age_ms: float = 0.0,
) -> LiveSnapshot:
    pose = pose if pose is not None else _pose(timestamp_ns)
    face = face if face is not None else _face(timestamp_ns)
    frame = FramePacket(
        sequence_id=0,
        captured_at_ns=timestamp_ns,
        source_id="camera",
        device_index=0,
        width=2,
        height=2,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    luminance_timestamp_ns = timestamp_ns - round(luminance_age_ms * 1_000_000)
    luminance = LuminanceObservation(
        source_id="camera",
        sequence_id=0,
        captured_at_ns=luminance_timestamp_ns,
        state=ObservationState.VALID,
        mean=luminance_mean,
        median=luminance_mean,
        p10=luminance_mean,
        p90=luminance_p90,
    )
    return LiveSnapshot(
        frame=frame,
        luminance=luminance,
        luminance_ran=luminance_age_ms == 0,
        luminance_age_ms=luminance_age_ms,
        pose_observation=None,
        pose_features=pose,
        pose_ran=True,
        pose_age_ms=0.0,
        pose_stale=pose_stale,
        face_observation=None,
        face_features=face,
        face_ran=True,
        face_age_ms=0.0,
        face_stale=face_stale,
    )


def _audio(timestamp_ns: int, *, dbfs: float = -10.0) -> AudioLevelObservation:
    return AudioLevelObservation(
        source_id="microphone",
        window_started_at_ns=timestamp_ns - 100_000_000,
        window_ended_at_ns=timestamp_ns,
        sample_rate_hz=16_000,
        sample_count=1_600,
        state=ObservationState.VALID,
        rms=0.1,
        dbfs=dbfs,
    )


def test_repository_event_config_parses_as_development_defaults() -> None:
    config = ErgonomicsEventConfig.load(CONFIG_PATH)

    assert config.schema_version == "1.0"
    assert config.status == "development_defaults_not_acceptance_thresholds"
    assert config.distance.exit_ratio < config.distance.enter_ratio
    assert config.noise.exit_dbfs < config.noise.enter_dbfs


def test_all_rules_evaluate_independently_and_blink_waits_for_valid_time() -> None:
    engine = ErgonomicsRuleEngine(_config(), profile=_profile())
    first_ns = 1_000_000_000
    first = engine.update(_snapshot(first_ns), audio_level=_audio(first_ns))
    second_ns = first_ns + 100_000_000
    second = engine.update(_snapshot(second_ns), audio_level=_audio(second_ns))

    for name in (
        "static_too_long",
        "bad_posture",
        "screen_too_close",
        "head_off_center",
        "environment_too_dark",
        "noise_too_high",
    ):
        assert first.evaluation(name).semantic_state is SemanticState.WARNING
    assert first.evaluation("environment_too_bright").semantic_state is SemanticState.NORMAL
    assert first.evaluation("low_blink_rate").semantic_state is SemanticState.UNKNOWN
    assert second.evaluation("low_blink_rate").semantic_state is SemanticState.WARNING
    assert len(second.evaluations) == len(ErgonomicsRuleEngine.EVENT_NAMES)


def test_stale_or_missing_evidence_becomes_unknown_and_does_not_clear_warning() -> None:
    engine = ErgonomicsRuleEngine(_config(), profile=_profile())
    timestamp_ns = 3_000_000_000
    engine.update(_snapshot(timestamp_ns), audio_level=_audio(timestamp_ns))
    stale = engine.update(
        _snapshot(
            timestamp_ns + 100_000_000,
            pose_stale=True,
            face_stale=True,
            luminance_age_ms=2_000,
        ),
        audio_level=None,
    )

    for name in ErgonomicsRuleEngine.EVENT_NAMES:
        assert stale.evaluation(name).semantic_state is SemanticState.UNKNOWN
    assert stale.evaluation("screen_too_close").phase is TemporalPhase.ACTIVE


def test_distance_uses_numeric_hysteresis_after_warning() -> None:
    engine = ErgonomicsRuleEngine(_config(), profile=_profile())
    timestamp_ns = 1_000_000_000
    entered = engine.update(
        _snapshot(timestamp_ns, face=_face(timestamp_ns, area=0.15)),
        audio_level=_audio(timestamp_ns, dbfs=-40),
    )
    held_ns = timestamp_ns + 100_000_000
    held = engine.update(
        _snapshot(held_ns, face=_face(held_ns, area=0.135)),
        audio_level=_audio(held_ns, dbfs=-40),
    )
    cleared_ns = held_ns + 100_000_000
    cleared = engine.update(
        _snapshot(cleared_ns, face=_face(cleared_ns, area=0.12)),
        audio_level=_audio(cleared_ns, dbfs=-40),
    )

    assert entered.evaluation("screen_too_close").semantic_state is SemanticState.WARNING
    assert held.evaluation("screen_too_close").semantic_state is SemanticState.WARNING
    assert cleared.evaluation("screen_too_close").semantic_state is SemanticState.NORMAL


def test_partial_posture_can_warn_but_cannot_claim_normal() -> None:
    profile = _profile()
    profile = CalibrationProfile(
        **{
            field: getattr(profile, field)
            for field in profile.__dataclass_fields__
            if field != "torso_lean_from_vertical_deg"
        },
        torso_lean_from_vertical_deg=None,
    )
    engine = ErgonomicsRuleEngine(_config(), profile=profile)
    first_ns = 1_000_000_000
    warning = engine.update(
        _snapshot(first_ns, pose=_pose(first_ns, shoulder=20, torso=None)),
        audio_level=_audio(first_ns),
    )
    normal_ns = first_ns + 100_000_000
    incomplete = engine.update(
        _snapshot(normal_ns, pose=_pose(normal_ns, shoulder=0, torso=None)),
        audio_level=_audio(normal_ns),
    )

    assert warning.evaluation("bad_posture").semantic_state is SemanticState.WARNING
    assert incomplete.evaluation("bad_posture").semantic_state is SemanticState.UNKNOWN
    assert incomplete.evaluation("bad_posture").reason == "incomplete_posture_coverage"


def test_rule_engine_rejects_repeated_time_and_future_audio() -> None:
    engine = ErgonomicsRuleEngine(_config(), profile=_profile())
    timestamp_ns = 1_000_000_000
    future_audio = _audio(timestamp_ns + 100_000_000)
    result = engine.update(_snapshot(timestamp_ns), audio_level=future_audio)

    assert result.evaluation("noise_too_high").semantic_state is SemanticState.UNKNOWN
    with pytest.raises(ValueError, match="increase strictly"):
        engine.update(_snapshot(timestamp_ns), audio_level=None)


def test_config_rejects_inverted_hysteresis() -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["screen_too_close"]["exit_face_area_ratio_to_baseline"] = 2.0

    with pytest.raises(ValueError, match="distance exit"):
        ErgonomicsEventConfig.from_mapping(data)


def test_config_rejects_silent_schema_change() -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["schema_version"] = "2.0"

    with pytest.raises(ValueError, match="unsupported event config schema"):
        ErgonomicsEventConfig.from_mapping(data)


def test_long_camera_gap_cannot_complete_entry_confirmation() -> None:
    data = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    data["environment_too_dark"]["enter_duration_ms"] = 1_000
    data["environment_too_dark"]["exit_duration_ms"] = 0
    data["environment_too_dark"]["cooldown_ms"] = 0
    config = ErgonomicsEventConfig.from_mapping(data)
    engine = ErgonomicsRuleEngine(config, profile=_profile())
    first_ns = 1_000_000_000

    entering = engine.update(_snapshot(first_ns), audio_level=_audio(first_ns))
    gap_ns = first_ns + 10_000_000_000
    interrupted = engine.update(_snapshot(gap_ns), audio_level=_audio(gap_ns))
    resumed_ns = gap_ns + 100_000_000
    resumed = engine.update(_snapshot(resumed_ns), audio_level=_audio(resumed_ns))

    assert entering.evaluation("environment_too_dark").phase is TemporalPhase.ENTERING
    assert interrupted.evaluation("environment_too_dark").semantic_state is SemanticState.UNKNOWN
    assert interrupted.evaluation("environment_too_dark").phase is TemporalPhase.IDLE
    assert resumed.evaluation("environment_too_dark").phase is TemporalPhase.ENTERING
    assert resumed.evaluation("environment_too_dark").semantic_state is SemanticState.NORMAL


def test_explicit_gap_pauses_confirmed_active_duration() -> None:
    engine = ErgonomicsRuleEngine(_config(), profile=_profile())
    first_ns = 1_000_000_000
    first = engine.update(_snapshot(first_ns), audio_level=_audio(first_ns))
    gap = engine.mark_evidence_gap(
        first_ns + 10_000_000_000,
        reason="capture_read_failed",
    )
    resumed_ns = first_ns + 10_100_000_000
    resumed = engine.update(_snapshot(resumed_ns), audio_level=_audio(resumed_ns))
    continued_ns = resumed_ns + 100_000_000
    continued = engine.update(
        _snapshot(continued_ns),
        audio_level=_audio(continued_ns),
    )

    assert first.evaluation("environment_too_dark").active_duration_ms == 0
    assert gap.evaluation("environment_too_dark").semantic_state is SemanticState.UNKNOWN
    assert gap.evaluation("environment_too_dark").phase is TemporalPhase.ACTIVE
    assert gap.evaluation("environment_too_dark").active_duration_ms == 0
    assert resumed.evaluation("environment_too_dark").active_duration_ms == 0
    assert continued.evaluation("environment_too_dark").active_duration_ms == pytest.approx(100)


def test_rejected_first_snapshot_does_not_pollute_camera_identity() -> None:
    engine = ErgonomicsRuleEngine(_config())
    timestamp_ns = 1_000_000_000
    valid = _snapshot(timestamp_ns)
    mismatched = replace(
        valid,
        frame=replace(valid.frame, source_id="wrong-camera"),
    )

    with pytest.raises(ValueError, match="source does not match"):
        engine.update(mismatched)
    accepted = engine.update(valid)

    assert accepted.source_id == "camera"


def test_rule_engine_rejects_inconsistent_feature_age() -> None:
    engine = ErgonomicsRuleEngine(_config())
    snapshot = _snapshot(1_000_000_000)

    with pytest.raises(ValueError, match="pose age"):
        engine.update(replace(snapshot, pose_age_ms=1.0))


def test_profile_rejects_non_finite_optional_torso_baseline() -> None:
    profile = replace(_profile(), torso_lean_from_vertical_deg=float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        ErgonomicsRuleEngine(_config(), profile=profile)
