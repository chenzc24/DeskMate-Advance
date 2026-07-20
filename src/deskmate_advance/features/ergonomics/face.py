"""Face geometry, raw rotation and blink evidence for Part A."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from deskmate_advance.perception.ergonomics.observations import (
    FaceObservation,
    ObservationState,
)


@dataclass(frozen=True, slots=True)
class FaceFeatureConfig:
    expected_landmark_count: int = 478

    def __post_init__(self) -> None:
        if self.expected_landmark_count <= 0:
            raise ValueError("expected_landmark_count must be positive")


@dataclass(frozen=True, slots=True)
class FaceFeatures:
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
    geometry_state: ObservationState
    rotation_state: ObservationState
    blink_state: ObservationState
    face_center_xy: tuple[float, float] | None
    face_bbox_width_ratio: float | None
    face_bbox_height_ratio: float | None
    face_bbox_area_ratio: float | None
    raw_rotation_xyz_deg: tuple[float, float, float] | None
    raw_translation_xyz: tuple[float, float, float] | None
    eye_blink_left: float | None
    eye_blink_right: float | None
    eye_blink_mean: float | None
    valid_eye_dt_ns: int | None
    rotation_reason: str | None = None
    blink_reason: str | None = None
    reason: str | None = None


def _blendshape_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


class FaceFeatureExtractor:
    """Extract raw, calibratable face features without product thresholds."""

    def __init__(self, config: FaceFeatureConfig | None = None) -> None:
        self.config = config or FaceFeatureConfig()
        self._last_observation_at_ns: int | None = None
        self._last_valid_eye_at_ns: int | None = None

    def extract(self, observation: FaceObservation) -> FaceFeatures:
        context = observation.context
        dt_ns = self._observation_delta(context.captured_at_ns)
        if observation.state is not ObservationState.VALID:
            self._last_valid_eye_at_ns = None
            return self._empty(
                observation.state, observation, observation.reason, dt_ns
            )
        if len(observation.landmarks) != self.config.expected_landmark_count:
            return self._empty(
                ObservationState.ERROR,
                observation,
                f"unexpected_face_landmark_count:{len(observation.landmarks)}",
                dt_ns,
            )
        coordinates = np.array(
            [(item.x, item.y, item.z) for item in observation.landmarks],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(coordinates)):
            return self._empty(
                ObservationState.ERROR,
                observation,
                "non_finite_face_landmarks",
                dt_ns,
            )
        minimum = np.min(coordinates[:, :2], axis=0)
        maximum = np.max(coordinates[:, :2], axis=0)
        width, height = maximum - minimum
        center = (minimum + maximum) / 2
        rotation, translation, rotation_state, rotation_reason = self._matrix_features(
            observation.transformation_matrix
        )
        scores = {
            _blendshape_key(item.name): item.score
            for item in observation.blendshapes
            if math.isfinite(item.score) and 0 <= item.score <= 1
        }
        left = scores.get("eyeblinkleft")
        right = scores.get("eyeblinkright")
        blink_complete = left is not None and right is not None
        if blink_complete:
            blink_state = ObservationState.VALID
            blink_reason = None
            valid_eye_dt_ns = (
                context.captured_at_ns - self._last_valid_eye_at_ns
                if self._last_valid_eye_at_ns is not None
                and context.captured_at_ns > self._last_valid_eye_at_ns
                else None
            )
            self._last_valid_eye_at_ns = context.captured_at_ns
        else:
            blink_state = ObservationState.MISSING
            blink_reason = "incomplete_blink_scores"
            valid_eye_dt_ns = None
            self._last_valid_eye_at_ns = None
        return FaceFeatures(
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
            geometry_state=ObservationState.VALID,
            rotation_state=rotation_state,
            blink_state=blink_state,
            face_center_xy=(float(center[0]), float(center[1])),
            face_bbox_width_ratio=float(width),
            face_bbox_height_ratio=float(height),
            face_bbox_area_ratio=float(width * height),
            raw_rotation_xyz_deg=rotation,
            raw_translation_xyz=translation,
            eye_blink_left=left,
            eye_blink_right=right,
            eye_blink_mean=((left + right) / 2 if blink_complete else None),
            valid_eye_dt_ns=valid_eye_dt_ns,
            rotation_reason=rotation_reason,
            blink_reason=blink_reason,
        )

    @staticmethod
    def _matrix_features(
        matrix_rows: tuple[tuple[float, ...], ...],
    ) -> tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
        ObservationState,
        str | None,
    ]:
        if not matrix_rows:
            return None, None, ObservationState.MISSING, "matrix_not_available"
        matrix = np.asarray(matrix_rows, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None, None, ObservationState.ERROR, "invalid_transformation_matrix"
        rotation = matrix[:3, :3]
        axis_y_scale = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        singular = axis_y_scale < 1e-6
        if not singular:
            x_angle = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            y_angle = math.atan2(-float(rotation[2, 0]), axis_y_scale)
            z_angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            x_angle = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
            y_angle = math.atan2(-float(rotation[2, 0]), axis_y_scale)
            z_angle = 0.0
        angles = tuple(math.degrees(value) for value in (x_angle, y_angle, z_angle))
        translation = tuple(float(value) for value in matrix[:3, 3])
        return angles, translation, ObservationState.VALID, None

    @staticmethod
    def _empty(
        state: ObservationState,
        observation: FaceObservation,
        reason: str | None,
        dt_ns: int | None,
    ) -> FaceFeatures:
        context = observation.context
        return FaceFeatures(
            source_id=context.source_id,
            sequence_id=context.sequence_id,
            captured_at_ns=context.captured_at_ns,
            state=state,
            model_id=context.model_id,
            model_version=context.model_version,
            asset_sha256=context.asset_sha256,
            config_sha256=context.config_sha256,
            dropped_before=context.dropped_before,
            dt_ns=dt_ns,
            geometry_state=state,
            rotation_state=state,
            blink_state=state,
            face_center_xy=None,
            face_bbox_width_ratio=None,
            face_bbox_height_ratio=None,
            face_bbox_area_ratio=None,
            raw_rotation_xyz_deg=None,
            raw_translation_xyz=None,
            eye_blink_left=None,
            eye_blink_right=None,
            eye_blink_mean=None,
            valid_eye_dt_ns=None,
            reason=reason,
        )

    def _observation_delta(self, captured_at_ns: int) -> int | None:
        previous = self._last_observation_at_ns
        self._last_observation_at_ns = captured_at_ns
        if previous is None or captured_at_ns <= previous:
            return None
        return captured_at_ns - previous
