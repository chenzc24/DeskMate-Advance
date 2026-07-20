"""Calibrated, independent Part A ergonomic condition evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from deskmate_advance.features.ergonomics import LiveSnapshot
from deskmate_advance.perception.ergonomics import (
    AudioLevelObservation,
    ObservationState,
)

from .blink import BlinkRateSnapshot, BlinkRateTracker, BlinkTrackerConfig
from .calibration import CalibrationConfig, CalibrationProfile
from .core import (
    ConditionState,
    SemanticState,
    TemporalPhase,
    TemporalStateConfig,
    TemporalStateMachine,
)


EvidenceValue = str | float | int | bool | None


@dataclass(frozen=True, slots=True)
class RuleTiming:
    enter_duration_ms: int
    exit_duration_ms: int
    cooldown_ms: int

    def temporal_config(self) -> TemporalStateConfig:
        return TemporalStateConfig(
            enter_duration_ms=self.enter_duration_ms,
            exit_duration_ms=self.exit_duration_ms,
            cooldown_ms=self.cooldown_ms,
        )


@dataclass(frozen=True, slots=True)
class StaticRuleConfig:
    enter_motion: float
    exit_motion: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class PostureRuleConfig:
    enter_shoulder_deg: float
    enter_torso_deg: float
    exit_shoulder_deg: float
    exit_torso_deg: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class DistanceRuleConfig:
    enter_ratio: float
    exit_ratio: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class HeadRuleConfig:
    enter_x_deg: float
    enter_y_deg: float
    exit_x_deg: float
    exit_y_deg: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class LuminanceRuleConfig:
    enter_value: float
    exit_value: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class NoiseRuleConfig:
    enter_dbfs: float
    exit_dbfs: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class BlinkRuleConfig:
    tracker: BlinkTrackerConfig
    recovery_blinks_per_minute: float
    timing: RuleTiming


@dataclass(frozen=True, slots=True)
class ErgonomicsEventConfig:
    schema_version: str
    status: str
    maximum_evidence_gap_ms: int
    luminance_stale_after_ms: int
    audio_stale_after_ms: int
    calibration: CalibrationConfig
    static: StaticRuleConfig
    posture: PostureRuleConfig
    distance: DistanceRuleConfig
    head: HeadRuleConfig
    blink: BlinkRuleConfig
    dark: LuminanceRuleConfig
    bright: LuminanceRuleConfig
    noise: NoiseRuleConfig

    @classmethod
    def load(cls, path: Path) -> ErgonomicsEventConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("event config root must be an object")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ErgonomicsEventConfig:
        schema_version = _non_empty_text(data, "schema_version")
        status = _non_empty_text(data, "status")
        if schema_version != "1.0":
            raise ValueError(f"unsupported event config schema: {schema_version}")
        calibration = _section(data, "calibration")
        static = _section(data, "static_too_long")
        posture = _section(data, "bad_posture")
        distance = _section(data, "screen_too_close")
        head = _section(data, "head_off_center")
        blink = _section(data, "low_blink_rate")
        dark = _section(data, "environment_too_dark")
        bright = _section(data, "environment_too_bright")
        noise = _section(data, "noise_too_high")

        static_enter = _finite_number(static, "motion_threshold_body_per_second", 0)
        static_exit = _finite_number(
            static, "exit_motion_threshold_body_per_second", static_enter
        )
        shoulder_enter = _finite_number(posture, "shoulder_delta_deg", 0)
        torso_enter = _finite_number(posture, "torso_delta_deg", 0)
        shoulder_exit = _finite_number(posture, "exit_shoulder_delta_deg", 0)
        torso_exit = _finite_number(posture, "exit_torso_delta_deg", 0)
        distance_enter = _finite_number(distance, "face_area_ratio_to_baseline", 1)
        distance_exit = _finite_number(
            distance, "exit_face_area_ratio_to_baseline", 0
        )
        head_x_enter = _finite_number(head, "raw_x_delta_deg", 0)
        head_y_enter = _finite_number(head, "raw_y_delta_deg", 0)
        head_x_exit = _finite_number(head, "exit_raw_x_delta_deg", 0)
        head_y_exit = _finite_number(head, "exit_raw_y_delta_deg", 0)
        blink_minimum = _finite_number(blink, "minimum_blinks_per_minute", 0)
        blink_recovery = _finite_number(blink, "recovery_blinks_per_minute", 0)
        dark_enter = _finite_number(dark, "mean_luminance", 0, 255)
        dark_exit = _finite_number(dark, "exit_mean_luminance", 0, 255)
        bright_enter = _finite_number(bright, "p90_luminance", 0, 255)
        bright_exit = _finite_number(bright, "exit_p90_luminance", 0, 255)
        noise_enter = _finite_number(noise, "dbfs", -200, 0)
        noise_exit = _finite_number(noise, "exit_dbfs", -200, 0)

        if static_exit <= static_enter:
            raise ValueError("static exit motion must exceed its entry threshold")
        if shoulder_exit >= shoulder_enter or torso_exit >= torso_enter:
            raise ValueError("posture exit deltas must be lower than entry deltas")
        if not 0 < distance_exit < distance_enter:
            raise ValueError("distance exit ratio must be in (0, entry ratio)")
        if head_x_exit >= head_x_enter or head_y_exit >= head_y_enter:
            raise ValueError("head exit deltas must be lower than entry deltas")
        if blink_recovery < blink_minimum:
            raise ValueError("blink recovery rate must not be below entry rate")
        if dark_exit <= dark_enter:
            raise ValueError("dark exit luminance must exceed entry luminance")
        if bright_exit >= bright_enter:
            raise ValueError("bright exit luminance must be below entry luminance")
        if noise_exit >= noise_enter:
            raise ValueError("noise exit dBFS must be below entry dBFS")

        return cls(
            schema_version=schema_version,
            status=status,
            maximum_evidence_gap_ms=_positive_int(
                data, "maximum_evidence_gap_ms"
            ),
            luminance_stale_after_ms=_positive_int(
                data, "luminance_stale_after_ms"
            ),
            audio_stale_after_ms=_positive_int(data, "audio_stale_after_ms"),
            calibration=CalibrationConfig(
                duration_ms=_positive_int(calibration, "duration_ms"),
                maximum_snapshot_gap_ms=_positive_int(
                    calibration, "maximum_snapshot_gap_ms"
                ),
                minimum_pose_samples=_positive_int(
                    calibration, "minimum_pose_samples"
                ),
                minimum_face_samples=_positive_int(
                    calibration, "minimum_face_samples"
                ),
                minimum_luminance_samples=_positive_int(
                    calibration, "minimum_luminance_samples"
                ),
            ),
            static=StaticRuleConfig(static_enter, static_exit, _timing(static)),
            posture=PostureRuleConfig(
                shoulder_enter,
                torso_enter,
                shoulder_exit,
                torso_exit,
                _timing(posture),
            ),
            distance=DistanceRuleConfig(
                distance_enter, distance_exit, _timing(distance)
            ),
            head=HeadRuleConfig(
                head_x_enter,
                head_y_enter,
                head_x_exit,
                head_y_exit,
                _timing(head),
            ),
            blink=BlinkRuleConfig(
                tracker=BlinkTrackerConfig(
                    closed_score=_finite_number(blink, "closed_score", 0, 1),
                    open_score=_finite_number(blink, "open_score", 0, 1),
                    window_ms=_positive_int(blink, "window_ms"),
                    minimum_valid_ms=_positive_int(blink, "minimum_valid_ms"),
                    minimum_blinks_per_minute=blink_minimum,
                    minimum_closed_ms=_non_negative_int(
                        blink, "minimum_closed_ms"
                    ),
                    maximum_closed_ms=_positive_int(blink, "maximum_closed_ms"),
                    maximum_valid_gap_ms=_positive_int(
                        blink, "maximum_valid_gap_ms"
                    ),
                ),
                recovery_blinks_per_minute=blink_recovery,
                timing=_timing(blink),
            ),
            dark=LuminanceRuleConfig(dark_enter, dark_exit, _timing(dark)),
            bright=LuminanceRuleConfig(
                bright_enter, bright_exit, _timing(bright)
            ),
            noise=NoiseRuleConfig(noise_enter, noise_exit, _timing(noise)),
        )


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    event_name: str
    condition: ConditionState
    semantic_state: SemanticState
    phase: TemporalPhase
    observed_at_ns: int
    evidence_elapsed_ms: float
    active_duration_ms: float
    cooldown_remaining_ms: float
    reason: str | None
    evidence: tuple[tuple[str, EvidenceValue], ...]

    def evidence_dict(self) -> dict[str, EvidenceValue]:
        return dict(self.evidence)


@dataclass(frozen=True, slots=True)
class ErgonomicsRuleSnapshot:
    source_id: str
    captured_at_ns: int
    config_schema_version: str
    config_status: str
    evaluations: tuple[RuleEvaluation, ...]
    blink_rate: BlinkRateSnapshot | None

    def evaluation(self, event_name: str) -> RuleEvaluation:
        for item in self.evaluations:
            if item.event_name == event_name:
                return item
        raise KeyError(event_name)


class ErgonomicsRuleEngine:
    """Evaluate all Part A functions independently over one live snapshot."""

    EVENT_NAMES = (
        "static_too_long",
        "bad_posture",
        "screen_too_close",
        "head_off_center",
        "low_blink_rate",
        "environment_too_dark",
        "environment_too_bright",
        "noise_too_high",
    )

    def __init__(
        self,
        config: ErgonomicsEventConfig,
        *,
        profile: CalibrationProfile | None = None,
    ) -> None:
        self.config = config
        self._profile: CalibrationProfile | None = None
        self._machines = self._new_machines()
        self._blink_tracker = BlinkRateTracker(config.blink.tracker)
        self._last_timestamp_ns: int | None = None
        self._source_id: str | None = None
        self._device_index: int | None = None
        if profile is not None:
            self.set_profile(profile)

    @property
    def profile(self) -> CalibrationProfile | None:
        return self._profile

    def set_profile(self, profile: CalibrationProfile) -> None:
        if not isinstance(profile, CalibrationProfile):
            raise TypeError("profile must be a CalibrationProfile")
        if not profile.source_id.strip() or profile.device_index < 0:
            raise ValueError("calibration profile has invalid camera identity")
        if (
            profile.window_started_at_ns < 0
            or profile.window_ended_at_ns <= profile.window_started_at_ns
        ):
            raise ValueError("calibration profile has invalid window timestamps")
        required = (
            profile.shoulder_tilt_deg,
            profile.face_bbox_area_ratio,
            profile.head_rotation_x_deg,
            profile.head_rotation_y_deg,
            profile.mean_luminance,
            profile.p90_luminance,
        )
        if not all(math.isfinite(value) for value in required):
            raise ValueError("calibration profile contains non-finite baselines")
        for optional in (
            profile.torso_lean_from_vertical_deg,
            profile.eye_open_score,
        ):
            if optional is not None and not math.isfinite(optional):
                raise ValueError("calibration profile contains non-finite baselines")
        if profile.face_bbox_area_ratio <= 0:
            raise ValueError("calibration face-area baseline must be positive")
        if profile.eye_open_score is not None and not 0 <= profile.eye_open_score <= 1:
            raise ValueError("calibration eye-open score must be in [0, 1]")
        self._profile = profile
        self.reset(keep_profile=True)

    def reset(self, *, keep_profile: bool = True) -> None:
        for machine in self._machines.values():
            machine.reset()
        self._blink_tracker.reset()
        self._last_timestamp_ns = None
        self._source_id = None
        self._device_index = None
        if not keep_profile:
            self._profile = None

    def update(
        self,
        snapshot: LiveSnapshot,
        *,
        audio_level: AudioLevelObservation | None = None,
    ) -> ErgonomicsRuleSnapshot:
        timestamp_ns = snapshot.frame.captured_at_ns
        self._validate_snapshot(snapshot)
        discontinuity = self._discontinuity_reason(snapshot)
        self._last_timestamp_ns = timestamp_ns
        if discontinuity is not None:
            blink_rate = self._blink_tracker.update(
                timestamp_ns=timestamp_ns,
                left_score=None,
                right_score=None,
                valid=False,
            )
            return self._unknown_snapshot(
                timestamp_ns,
                source_id=snapshot.frame.source_id,
                reason=discontinuity,
                blink_rate=blink_rate,
            )

        blink_rate = self._update_blink(snapshot)
        inputs = (
            self._static_condition(snapshot),
            self._posture_condition(snapshot),
            self._distance_condition(snapshot),
            self._head_condition(snapshot),
            self._blink_condition(snapshot, blink_rate),
            self._dark_condition(snapshot),
            self._bright_condition(snapshot),
            self._noise_condition(snapshot, audio_level),
        )
        return self._snapshot_from_inputs(
            source_id=snapshot.frame.source_id,
            timestamp_ns=timestamp_ns,
            inputs=inputs,
            blink_rate=blink_rate,
        )

    def mark_evidence_gap(
        self,
        timestamp_ns: int,
        *,
        reason: str,
    ) -> ErgonomicsRuleSnapshot:
        """Advance every lane with UNKNOWN for an explicit camera miss."""

        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must not be empty")
        if self._last_timestamp_ns is None or self._source_id is None:
            raise RuntimeError("cannot mark a gap before the first live snapshot")
        if timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("rule-engine gap timestamps must increase strictly")
        self._last_timestamp_ns = timestamp_ns
        blink_rate = self._blink_tracker.update(
            timestamp_ns=timestamp_ns,
            left_score=None,
            right_score=None,
            valid=False,
        )
        return self._unknown_snapshot(
            timestamp_ns,
            source_id=self._source_id,
            reason=f"evidence_discontinuity:{reason}",
            blink_rate=blink_rate,
        )

    def _new_machines(self) -> dict[str, TemporalStateMachine]:
        timings = (
            self.config.static.timing,
            self.config.posture.timing,
            self.config.distance.timing,
            self.config.head.timing,
            self.config.blink.timing,
            self.config.dark.timing,
            self.config.bright.timing,
            self.config.noise.timing,
        )
        return {
            event_name: TemporalStateMachine(timing.temporal_config())
            for event_name, timing in zip(self.EVENT_NAMES, timings, strict=True)
        }

    def _validate_snapshot(self, snapshot: LiveSnapshot) -> None:
        frame = snapshot.frame
        timestamp_ns = frame.captured_at_ns
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("rule-engine camera timestamps must increase strictly")
        if self._source_id is not None and (
            frame.source_id != self._source_id
            or frame.device_index != self._device_index
        ):
            raise ValueError("rule-engine snapshots must come from one camera")
        if self._profile is not None and (
            frame.source_id != self._profile.source_id
            or frame.device_index != self._profile.device_index
        ):
            raise ValueError("calibration profile does not match the live camera")
        self._validate_record(
            "luminance",
            source_id=snapshot.luminance.source_id,
            captured_at_ns=snapshot.luminance.captured_at_ns,
            age_ms=snapshot.luminance_age_ms,
            ran=snapshot.luminance_ran,
            frame=snapshot.frame,
        )
        self._validate_optional_record(
            "pose",
            record=snapshot.pose_features,
            age_ms=snapshot.pose_age_ms,
            ran=snapshot.pose_ran,
            frame=snapshot.frame,
        )
        self._validate_optional_record(
            "face",
            record=snapshot.face_features,
            age_ms=snapshot.face_age_ms,
            ran=snapshot.face_ran,
            frame=snapshot.frame,
        )
        if self._source_id is None:
            self._source_id = frame.source_id
            self._device_index = frame.device_index

    def _discontinuity_reason(self, snapshot: LiveSnapshot) -> str | None:
        if snapshot.frame.dropped_before > 0:
            return (
                "evidence_discontinuity:camera_frames_dropped:"
                f"{snapshot.frame.dropped_before}"
            )
        if self._last_timestamp_ns is None:
            return None
        gap_ms = (
            snapshot.frame.captured_at_ns - self._last_timestamp_ns
        ) / 1_000_000
        if gap_ms > self.config.maximum_evidence_gap_ms:
            return f"evidence_discontinuity:snapshot_gap_ms:{gap_ms:.3f}"
        return None

    @staticmethod
    def _validate_optional_record(
        label: str,
        *,
        record: Any | None,
        age_ms: float | None,
        ran: bool,
        frame: Any,
    ) -> None:
        if record is None:
            if age_ms is not None:
                raise ValueError(f"{label} age exists without a feature record")
            return
        ErgonomicsRuleEngine._validate_record(
            label,
            source_id=record.source_id,
            captured_at_ns=record.captured_at_ns,
            age_ms=age_ms,
            ran=ran,
            frame=frame,
        )

    @staticmethod
    def _validate_record(
        label: str,
        *,
        source_id: str,
        captured_at_ns: int,
        age_ms: float | None,
        ran: bool,
        frame: Any,
    ) -> None:
        if source_id != frame.source_id:
            raise ValueError(f"{label} source does not match camera frame")
        if captured_at_ns > frame.captured_at_ns:
            raise ValueError(f"{label} timestamp is later than camera frame")
        expected_age_ms = (frame.captured_at_ns - captured_at_ns) / 1_000_000
        if (
            age_ms is None
            or not math.isfinite(age_ms)
            or age_ms < 0
            or not math.isclose(age_ms, expected_age_ms, abs_tol=1e-6)
        ):
            raise ValueError(f"{label} age does not match its timestamp")
        if ran and captured_at_ns != frame.captured_at_ns:
            raise ValueError(f"fresh {label} timestamp must match camera frame")

    def _unknown_snapshot(
        self,
        timestamp_ns: int,
        *,
        source_id: str,
        reason: str,
        blink_rate: BlinkRateSnapshot | None,
    ) -> ErgonomicsRuleSnapshot:
        inputs = tuple(
            (ConditionState.UNKNOWN, reason, {"evidence_continuous": False})
            for _ in self.EVENT_NAMES
        )
        return self._snapshot_from_inputs(
            source_id=source_id,
            timestamp_ns=timestamp_ns,
            inputs=inputs,
            blink_rate=blink_rate,
        )

    def _snapshot_from_inputs(
        self,
        *,
        source_id: str,
        timestamp_ns: int,
        inputs: tuple[
            tuple[ConditionState, str | None, dict[str, EvidenceValue]], ...
        ],
        blink_rate: BlinkRateSnapshot | None,
    ) -> ErgonomicsRuleSnapshot:
        evaluations = tuple(
            self._advance(event_name, condition, reason, evidence, timestamp_ns)
            for event_name, (condition, reason, evidence) in zip(
                self.EVENT_NAMES, inputs, strict=True
            )
        )
        return ErgonomicsRuleSnapshot(
            source_id=source_id,
            captured_at_ns=timestamp_ns,
            config_schema_version=self.config.schema_version,
            config_status=self.config.status,
            evaluations=evaluations,
            blink_rate=blink_rate,
        )

    def _update_blink(self, snapshot: LiveSnapshot) -> BlinkRateSnapshot | None:
        features = snapshot.face_features
        if snapshot.face_ran:
            valid = (
                features is not None
                and not snapshot.face_stale
                and features.blink_state is ObservationState.VALID
            )
            return self._blink_tracker.update(
                timestamp_ns=snapshot.frame.captured_at_ns,
                left_score=features.eye_blink_left if features is not None else None,
                right_score=features.eye_blink_right if features is not None else None,
                valid=valid,
            )
        try:
            return self._blink_tracker.snapshot(snapshot.frame.captured_at_ns)
        except RuntimeError:
            return None

    def _static_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        features = snapshot.pose_features
        motion = features.upper_body_motion_per_second if features is not None else None
        evidence = {
            "motion_body_per_second": motion,
            "enter_below": self.config.static.enter_motion,
            "exit_above": self.config.static.exit_motion,
        }
        if (
            features is None
            or snapshot.pose_stale
            or snapshot.pose_age_ms is None
            or snapshot.pose_age_ms > self.config.maximum_evidence_gap_ms
            or features.state is not ObservationState.VALID
            or features.temporal_gap
            or motion is None
            or not math.isfinite(motion)
        ):
            return ConditionState.UNKNOWN, "pose_motion_unavailable", evidence
        alarming = self._alarming("static_too_long")
        threshold = (
            self.config.static.exit_motion
            if alarming
            else self.config.static.enter_motion
        )
        return _truth(motion <= threshold), None, evidence

    def _posture_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        profile = self._profile
        features = snapshot.pose_features
        if profile is None:
            return ConditionState.UNKNOWN, "calibration_profile_unavailable", {}
        shoulder = features.shoulder_tilt_deg if features is not None else None
        torso = features.torso_lean_from_vertical_deg if features is not None else None
        shoulder_delta = (
            abs(shoulder - profile.shoulder_tilt_deg)
            if shoulder is not None and math.isfinite(shoulder)
            else None
        )
        torso_delta = (
            abs(torso - profile.torso_lean_from_vertical_deg)
            if torso is not None
            and profile.torso_lean_from_vertical_deg is not None
            and math.isfinite(torso)
            else None
        )
        evidence = {
            "shoulder_delta_deg": shoulder_delta,
            "torso_delta_deg": torso_delta,
            "torso_coverage": torso_delta is not None,
        }
        if (
            features is None
            or snapshot.pose_stale
            or snapshot.pose_age_ms is None
            or snapshot.pose_age_ms > self.config.maximum_evidence_gap_ms
            or features.state is not ObservationState.VALID
        ):
            return ConditionState.UNKNOWN, "pose_posture_unavailable", evidence
        alarming = self._alarming("bad_posture")
        shoulder_threshold = (
            self.config.posture.exit_shoulder_deg
            if alarming
            else self.config.posture.enter_shoulder_deg
        )
        torso_threshold = (
            self.config.posture.exit_torso_deg
            if alarming
            else self.config.posture.enter_torso_deg
        )
        if (shoulder_delta is not None and shoulder_delta >= shoulder_threshold) or (
            torso_delta is not None and torso_delta >= torso_threshold
        ):
            return ConditionState.TRUE, None, evidence
        if shoulder_delta is not None and torso_delta is not None:
            return ConditionState.FALSE, None, evidence
        return ConditionState.UNKNOWN, "incomplete_posture_coverage", evidence

    def _distance_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        profile = self._profile
        features = snapshot.face_features
        area = features.face_bbox_area_ratio if features is not None else None
        ratio = (
            area / profile.face_bbox_area_ratio
            if profile is not None
            and area is not None
            and math.isfinite(area)
            and area > 0
            else None
        )
        evidence = {
            "face_area_ratio_to_baseline": ratio,
            "absolute_distance_claimed": False,
        }
        if profile is None:
            return ConditionState.UNKNOWN, "calibration_profile_unavailable", evidence
        if (
            features is None
            or snapshot.face_stale
            or snapshot.face_age_ms is None
            or snapshot.face_age_ms > self.config.maximum_evidence_gap_ms
            or features.geometry_state is not ObservationState.VALID
            or ratio is None
        ):
            return ConditionState.UNKNOWN, "face_geometry_unavailable", evidence
        threshold = (
            self.config.distance.exit_ratio
            if self._alarming("screen_too_close")
            else self.config.distance.enter_ratio
        )
        return _truth(ratio >= threshold), None, evidence

    def _head_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        profile = self._profile
        features = snapshot.face_features
        rotation = features.raw_rotation_xyz_deg if features is not None else None
        delta_x = (
            abs(rotation[0] - profile.head_rotation_x_deg)
            if profile is not None and rotation is not None
            else None
        )
        delta_y = (
            abs(rotation[1] - profile.head_rotation_y_deg)
            if profile is not None and rotation is not None
            else None
        )
        evidence = {
            "raw_x_delta_deg": delta_x,
            "raw_y_delta_deg": delta_y,
            "direction_sign_calibrated": False,
        }
        if profile is None:
            return ConditionState.UNKNOWN, "calibration_profile_unavailable", evidence
        if (
            features is None
            or snapshot.face_stale
            or snapshot.face_age_ms is None
            or snapshot.face_age_ms > self.config.maximum_evidence_gap_ms
            or features.rotation_state is not ObservationState.VALID
            or delta_x is None
            or delta_y is None
            or not math.isfinite(delta_x)
            or not math.isfinite(delta_y)
        ):
            return ConditionState.UNKNOWN, "head_rotation_unavailable", evidence
        alarming = self._alarming("head_off_center")
        threshold_x = self.config.head.exit_x_deg if alarming else self.config.head.enter_x_deg
        threshold_y = self.config.head.exit_y_deg if alarming else self.config.head.enter_y_deg
        return _truth(delta_x >= threshold_x or delta_y >= threshold_y), None, evidence

    def _blink_condition(
        self,
        snapshot: LiveSnapshot,
        rate: BlinkRateSnapshot | None,
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        evidence = {
            "blinks_per_minute": rate.blinks_per_minute if rate else None,
            "blink_count": rate.blink_count if rate else 0,
            "valid_eye_ms": rate.valid_duration_ms if rate else 0.0,
        }
        features = snapshot.face_features
        if (
            rate is None
            or features is None
            or snapshot.face_stale
            or snapshot.face_age_ms is None
            or snapshot.face_age_ms > self.config.maximum_evidence_gap_ms
            or features.blink_state is not ObservationState.VALID
            or rate.low_rate is None
            or rate.blinks_per_minute is None
        ):
            return (
                ConditionState.UNKNOWN,
                rate.reason if rate is not None and rate.reason else "blink_evidence_unavailable",
                evidence,
            )
        threshold = (
            self.config.blink.recovery_blinks_per_minute
            if self._alarming("low_blink_rate")
            else self.config.blink.tracker.minimum_blinks_per_minute
        )
        return _truth(rate.blinks_per_minute < threshold), None, evidence

    def _dark_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        mean = snapshot.luminance.mean
        evidence = {"mean_luminance": mean}
        if (
            snapshot.luminance.state is not ObservationState.VALID
            or mean is None
            or not math.isfinite(mean)
            or snapshot.luminance_age_ms > self.config.luminance_stale_after_ms
        ):
            return ConditionState.UNKNOWN, "luminance_unavailable_or_stale", evidence
        threshold = (
            self.config.dark.exit_value
            if self._alarming("environment_too_dark")
            else self.config.dark.enter_value
        )
        return _truth(mean <= threshold), None, evidence

    def _bright_condition(
        self, snapshot: LiveSnapshot
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        p90 = snapshot.luminance.p90
        evidence = {"p90_luminance": p90}
        if (
            snapshot.luminance.state is not ObservationState.VALID
            or p90 is None
            or not math.isfinite(p90)
            or snapshot.luminance_age_ms > self.config.luminance_stale_after_ms
        ):
            return ConditionState.UNKNOWN, "luminance_unavailable_or_stale", evidence
        threshold = (
            self.config.bright.exit_value
            if self._alarming("environment_too_bright")
            else self.config.bright.enter_value
        )
        return _truth(p90 >= threshold), None, evidence

    def _noise_condition(
        self,
        snapshot: LiveSnapshot,
        audio: AudioLevelObservation | None,
    ) -> tuple[ConditionState, str | None, dict[str, EvidenceValue]]:
        timestamp_ns = snapshot.frame.captured_at_ns
        dbfs = audio.dbfs if audio is not None else None
        age_ms = (
            (timestamp_ns - audio.window_ended_at_ns) / 1_000_000
            if audio is not None and audio.window_ended_at_ns <= timestamp_ns
            else None
        )
        evidence = {
            "dbfs": dbfs,
            "age_ms": age_ms,
            "window_started_at_ns": (
                audio.window_started_at_ns if audio is not None else None
            ),
            "window_ended_at_ns": (
                audio.window_ended_at_ns if audio is not None else None
            ),
            "spl_calibrated": False,
        }
        if (
            audio is None
            or audio.state is not ObservationState.VALID
            or dbfs is None
            or not math.isfinite(dbfs)
            or age_ms is None
            or age_ms > self.config.audio_stale_after_ms
        ):
            return ConditionState.UNKNOWN, "audio_level_unavailable_or_stale", evidence
        threshold = (
            self.config.noise.exit_dbfs
            if self._alarming("noise_too_high")
            else self.config.noise.enter_dbfs
        )
        return _truth(dbfs >= threshold), None, evidence

    def _advance(
        self,
        event_name: str,
        condition: ConditionState,
        reason: str | None,
        evidence: Mapping[str, EvidenceValue],
        timestamp_ns: int,
    ) -> RuleEvaluation:
        state = self._machines[event_name].update(condition, timestamp_ns)
        return RuleEvaluation(
            event_name=event_name,
            condition=condition,
            semantic_state=state.semantic_state,
            phase=state.phase,
            observed_at_ns=timestamp_ns,
            evidence_elapsed_ms=state.evidence_elapsed_ms,
            active_duration_ms=state.active_duration_ms,
            cooldown_remaining_ms=state.cooldown_remaining_ms,
            reason=reason,
            evidence=tuple(evidence.items()),
        )

    def _alarming(self, event_name: str) -> bool:
        return self._machines[event_name].phase in {
            TemporalPhase.ACTIVE,
            TemporalPhase.EXITING,
        }


def _truth(value: bool) -> ConditionState:
    return ConditionState.TRUE if value else ConditionState.FALSE


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _non_empty_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


def _finite_number(
    data: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    output = float(value)
    if not math.isfinite(output) or output < minimum or (
        maximum is not None and output > maximum
    ):
        raise ValueError(f"{key} is outside its allowed range")
    return output


def _non_negative_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _positive_int(data: Mapping[str, Any], key: str) -> int:
    value = _non_negative_int(data, key)
    if value == 0:
        raise ValueError(f"{key} must be positive")
    return value


def _timing(data: Mapping[str, Any]) -> RuleTiming:
    return RuleTiming(
        enter_duration_ms=_non_negative_int(data, "enter_duration_ms"),
        exit_duration_ms=_non_negative_int(data, "exit_duration_ms"),
        cooldown_ms=_non_negative_int(data, "cooldown_ms"),
    )
