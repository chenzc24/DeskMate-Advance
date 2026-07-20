"""Timestamp-aware Pose features for Part A ergonomics functions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from deskmate_advance.perception.ergonomics.observations import (
    Landmark3D,
    ObservationState,
    PoseObservation,
)


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0

_MOTION_INDICES = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
)

NormalizedPoint = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PoseFeatureConfig:
    min_visibility: float = 0.5
    min_presence: float = 0.5
    max_motion_gap_ms: int = 500

    def __post_init__(self) -> None:
        if not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be in [0, 1]")
        if not 0 <= self.min_presence <= 1:
            raise ValueError("min_presence must be in [0, 1]")
        if self.max_motion_gap_ms <= 0:
            raise ValueError("max_motion_gap_ms must be positive")


@dataclass(frozen=True, slots=True)
class PoseFeatures:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    state: ObservationState
    model_id: str
    model_version: str
    asset_sha256: str | None
    config_sha256: str | None
    dropped_before: int
    dt_ns: int | None
    temporal_gap: bool
    missing_mask: tuple[bool, ...]
    normalized_landmarks: tuple[NormalizedPoint | None, ...]
    valid_landmark_fraction: float
    normalization_scale: float | None
    shoulder_tilt_deg: float | None
    torso_lean_from_vertical_deg: float | None
    nose_offset_from_shoulders: NormalizedPoint | None
    upper_body_motion_per_second: float | None
    reason: str | None = None


def _midpoint(first: Landmark3D, second: Landmark3D) -> np.ndarray:
    return np.array(
        [
            (first.x + second.x) / 2,
            (first.y + second.y) / 2,
            (first.z + second.z) / 2,
        ],
        dtype=np.float64,
    )


class PoseFeatureExtractor:
    """Build partial-safe normalized Pose features and true-time motion."""

    def __init__(self, config: PoseFeatureConfig | None = None) -> None:
        self.config = config or PoseFeatureConfig()
        self._last_observation_at_ns: int | None = None
        self._previous_at_ns: int | None = None
        self._previous_normalized: tuple[NormalizedPoint | None, ...] | None = None

    def extract(self, observation: PoseObservation) -> PoseFeatures:
        context = observation.context
        dt_ns = self._observation_delta(context.captured_at_ns)
        temporal_gap = context.dropped_before > 0 or (
            dt_ns is not None
            and dt_ns > self.config.max_motion_gap_ms * 1_000_000
        )
        if observation.state is not ObservationState.VALID:
            self._clear_motion_history()
            return PoseFeatures(
                source_id=context.source_id,
                sequence_id=context.sequence_id,
                captured_at_ns=context.captured_at_ns,
                state=observation.state,
                model_id=context.model_id,
                model_version=context.model_version,
                asset_sha256=context.asset_sha256,
                config_sha256=context.config_sha256,
                dropped_before=context.dropped_before,
                dt_ns=dt_ns,
                temporal_gap=temporal_gap,
                missing_mask=(True,) * 33,
                normalized_landmarks=(None,) * 33,
                valid_landmark_fraction=0.0,
                normalization_scale=None,
                shoulder_tilt_deg=None,
                torso_lean_from_vertical_deg=None,
                nose_offset_from_shoulders=None,
                upper_body_motion_per_second=None,
                reason=observation.reason,
            )
        if len(observation.landmarks) != 33:
            self._clear_motion_history()
            return PoseFeatures(
                source_id=context.source_id,
                sequence_id=context.sequence_id,
                captured_at_ns=context.captured_at_ns,
                state=ObservationState.ERROR,
                model_id=context.model_id,
                model_version=context.model_version,
                asset_sha256=context.asset_sha256,
                config_sha256=context.config_sha256,
                dropped_before=context.dropped_before,
                dt_ns=dt_ns,
                temporal_gap=temporal_gap,
                missing_mask=(True,) * 33,
                normalized_landmarks=(None,) * 33,
                valid_landmark_fraction=0.0,
                normalization_scale=None,
                shoulder_tilt_deg=None,
                torso_lean_from_vertical_deg=None,
                nose_offset_from_shoulders=None,
                upper_body_motion_per_second=None,
                reason=f"unexpected_pose_landmark_count:{len(observation.landmarks)}",
            )

        valid = tuple(self._is_valid(item) for item in observation.landmarks)
        missing_mask = tuple(not value for value in valid)
        center, scale = self._normalization(observation.landmarks, valid)
        normalized = self._normalize(observation.landmarks, valid, center, scale)
        shoulder_tilt = self._shoulder_tilt(observation.landmarks, valid)
        torso_lean = self._torso_lean(observation.landmarks, valid)
        nose_offset = self._nose_offset(observation.landmarks, valid, scale)
        motion = self._motion(context.captured_at_ns, normalized)
        return PoseFeatures(
            source_id=context.source_id,
            sequence_id=context.sequence_id,
            captured_at_ns=context.captured_at_ns,
            state=ObservationState.VALID,
            model_id=context.model_id,
            model_version=context.model_version,
            asset_sha256=context.asset_sha256,
            config_sha256=context.config_sha256,
            dropped_before=context.dropped_before,
            dt_ns=dt_ns,
            temporal_gap=temporal_gap,
            missing_mask=missing_mask,
            normalized_landmarks=normalized,
            valid_landmark_fraction=sum(valid) / 33,
            normalization_scale=scale,
            shoulder_tilt_deg=shoulder_tilt,
            torso_lean_from_vertical_deg=torso_lean,
            nose_offset_from_shoulders=nose_offset,
            upper_body_motion_per_second=motion,
        )

    def _is_valid(self, point: Landmark3D) -> bool:
        coordinates_finite = all(math.isfinite(value) for value in (point.x, point.y, point.z))
        visibility_valid = (
            point.visibility is None or point.visibility >= self.config.min_visibility
        )
        presence_valid = (
            point.presence is None or point.presence >= self.config.min_presence
        )
        return coordinates_finite and visibility_valid and presence_valid

    @staticmethod
    def _normalization(
        landmarks: tuple[Landmark3D, ...],
        valid: tuple[bool, ...],
    ) -> tuple[np.ndarray | None, float | None]:
        shoulders_valid = valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]
        hips_valid = valid[LEFT_HIP] and valid[RIGHT_HIP]
        if hips_valid:
            center = _midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
        elif shoulders_valid:
            center = _midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
        else:
            return None, None
        scale_pair = (
            (LEFT_SHOULDER, RIGHT_SHOULDER)
            if shoulders_valid
            else (LEFT_HIP, RIGHT_HIP)
        )
        first, second = (landmarks[index] for index in scale_pair)
        scale = math.hypot(second.x - first.x, second.y - first.y)
        if not math.isfinite(scale) or scale <= 1e-9:
            return center, None
        return center, scale

    @staticmethod
    def _normalize(
        landmarks: tuple[Landmark3D, ...],
        valid: tuple[bool, ...],
        center: np.ndarray | None,
        scale: float | None,
    ) -> tuple[NormalizedPoint | None, ...]:
        if center is None or scale is None:
            return (None,) * 33
        output: list[NormalizedPoint | None] = []
        for is_valid, point in zip(valid, landmarks, strict=True):
            if not is_valid:
                output.append(None)
                continue
            output.append(
                (
                    (point.x - float(center[0])) / scale,
                    (point.y - float(center[1])) / scale,
                    (point.z - float(center[2])) / scale,
                )
            )
        return tuple(output)

    @staticmethod
    def _shoulder_tilt(
        landmarks: tuple[Landmark3D, ...],
        valid: tuple[bool, ...],
    ) -> float | None:
        if not (valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]):
            return None
        left, right = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        angle = math.degrees(math.atan2(right.y - left.y, right.x - left.x))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        return angle

    @staticmethod
    def _torso_lean(
        landmarks: tuple[Landmark3D, ...],
        valid: tuple[bool, ...],
    ) -> float | None:
        if not all(
            valid[index]
            for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        ):
            return None
        shoulder_center = _midpoint(
            landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        )
        hip_center = _midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
        vector = shoulder_center - hip_center
        if math.hypot(float(vector[0]), float(vector[1])) <= 1e-9:
            return None
        return math.degrees(math.atan2(float(vector[0]), -float(vector[1])))

    @staticmethod
    def _nose_offset(
        landmarks: tuple[Landmark3D, ...],
        valid: tuple[bool, ...],
        scale: float | None,
    ) -> NormalizedPoint | None:
        if scale is None or not all(
            valid[index] for index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
        ):
            return None
        center = _midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
        nose = landmarks[NOSE]
        return (
            (nose.x - float(center[0])) / scale,
            (nose.y - float(center[1])) / scale,
            (nose.z - float(center[2])) / scale,
        )

    def _motion(
        self,
        captured_at_ns: int,
        normalized: tuple[NormalizedPoint | None, ...],
    ) -> float | None:
        previous_at = self._previous_at_ns
        previous = self._previous_normalized
        self._previous_at_ns = captured_at_ns
        self._previous_normalized = normalized
        if previous_at is None or previous is None:
            return None
        delta_ns = captured_at_ns - previous_at
        if delta_ns <= 0 or delta_ns > self.config.max_motion_gap_ms * 1_000_000:
            return None
        distances = []
        for index in _MOTION_INDICES:
            current_point, previous_point = normalized[index], previous[index]
            if current_point is None or previous_point is None:
                continue
            distances.append(math.dist(current_point, previous_point))
        if not distances:
            return None
        return float(np.mean(distances)) / (delta_ns / 1_000_000_000)

    def _clear_motion_history(self) -> None:
        self._previous_at_ns = None
        self._previous_normalized = None

    def _observation_delta(self, captured_at_ns: int) -> int | None:
        previous = self._last_observation_at_ns
        self._last_observation_at_ns = captured_at_ns
        if previous is None or captured_at_ns <= previous:
            return None
        return captured_at_ns - previous
