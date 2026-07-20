"""Bounded, non-recording Part A live-camera inference engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Protocol

from deskmate_advance.domain.frame import FramePacket
from deskmate_advance.perception.ergonomics import (
    FaceObservation,
    LuminanceCalculator,
    LuminanceObservation,
    ObservationState,
    PoseObservation,
)

from .benchmark import Reservoir
from .face import FaceFeatureExtractor, FaceFeatures
from .pose import PoseFeatureExtractor, PoseFeatures


class PoseObserver(Protocol):
    def observe(self, frame: FramePacket) -> PoseObservation: ...


class FaceObserver(Protocol):
    def observe(self, frame: FramePacket) -> FaceObservation: ...


@dataclass(frozen=True, slots=True)
class LiveScheduleConfig:
    pose_hz: float = 10.0
    face_hz: float = 10.0
    luminance_hz: float = 2.0
    stale_after_ms: int = 500
    metric_reservoir_size: int = 2000

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0 < value <= 1000
            for value in (self.pose_hz, self.face_hz, self.luminance_hz)
        ):
            raise ValueError("model and luminance rates must be in (0, 1000] Hz")
        if self.stale_after_ms <= 0:
            raise ValueError("stale_after_ms must be positive")
        if self.metric_reservoir_size <= 0:
            raise ValueError("metric_reservoir_size must be positive")


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    frame: FramePacket
    luminance: LuminanceObservation
    luminance_ran: bool
    luminance_age_ms: float
    pose_observation: PoseObservation | None
    pose_features: PoseFeatures | None
    pose_ran: bool
    pose_age_ms: float | None
    pose_stale: bool
    face_observation: FaceObservation | None
    face_features: FaceFeatures | None
    face_ran: bool
    face_age_ms: float | None
    face_stale: bool


class _ModelStats:
    def __init__(self, reservoir_size: int) -> None:
        self.runs = 0
        self.states: Counter[str] = Counter()
        self.latency = Reservoir(reservoir_size)

    def add(self, state: ObservationState, latency_ms: float) -> None:
        self.runs += 1
        self.states[state.value] += 1
        self.latency.add(latency_ms)

    def summary(self) -> dict[str, Any]:
        counts = {state.value: self.states[state.value] for state in ObservationState}
        return {
            "runs": self.runs,
            "state_counts": counts,
            "state_rates": {
                name: count / self.runs if self.runs else None
                for name, count in counts.items()
            },
            "latency_ms": self.latency.summary(),
        }


class PartALiveEngine:
    """Schedule Pose/Face at independent rates over the latest camera frame."""

    def __init__(
        self,
        *,
        pose: PoseObserver,
        face: FaceObserver,
        schedule: LiveScheduleConfig | None = None,
        pose_features: PoseFeatureExtractor | None = None,
        face_features: FaceFeatureExtractor | None = None,
        luminance: LuminanceCalculator | None = None,
    ) -> None:
        self.pose = pose
        self.face = face
        self.schedule = schedule or LiveScheduleConfig()
        self.pose_feature_extractor = pose_features or PoseFeatureExtractor()
        self.face_feature_extractor = face_features or FaceFeatureExtractor()
        self.luminance_calculator = luminance or LuminanceCalculator()
        self._pose_period_ns = round(1_000_000_000 / self.schedule.pose_hz)
        self._face_period_ns = round(1_000_000_000 / self.schedule.face_hz)
        self._luminance_period_ns = round(
            1_000_000_000 / self.schedule.luminance_hz
        )
        self._last_pose_run_ns: int | None = None
        self._last_face_run_ns: int | None = None
        self._last_luminance_run_ns: int | None = None
        self._latest_pose: PoseObservation | None = None
        self._latest_pose_features: PoseFeatures | None = None
        self._latest_face: FaceObservation | None = None
        self._latest_face_features: FaceFeatures | None = None
        self._latest_luminance: LuminanceObservation | None = None
        self._first_frame_ns: int | None = None
        self._last_frame_ns: int | None = None
        self._frames = 0
        self._dropped_before = 0
        self._pose_stats = _ModelStats(self.schedule.metric_reservoir_size)
        self._face_stats = _ModelStats(self.schedule.metric_reservoir_size)
        self._luminance_runs = 0

    def process(self, frame: FramePacket) -> LiveSnapshot:
        if self._last_frame_ns is not None and frame.captured_at_ns <= self._last_frame_ns:
            raise ValueError("camera frame timestamps must increase monotonically")
        self._frames += 1
        self._dropped_before += frame.dropped_before
        if self._first_frame_ns is None:
            self._first_frame_ns = frame.captured_at_ns
        self._last_frame_ns = frame.captured_at_ns
        luminance_ran = self._due(
            frame.captured_at_ns,
            self._last_luminance_run_ns,
            self._luminance_period_ns,
        )
        if luminance_ran:
            self._last_luminance_run_ns = frame.captured_at_ns
            self._latest_luminance = self.luminance_calculator.observe(frame)
            self._luminance_runs += 1
        if self._latest_luminance is None:  # First frame is always due.
            raise RuntimeError("luminance scheduler did not initialize")

        pose_ran = self._due(
            frame.captured_at_ns,
            self._last_pose_run_ns,
            self._pose_period_ns,
        )
        if pose_ran:
            self._last_pose_run_ns = frame.captured_at_ns
            self._latest_pose = self.pose.observe(frame)
            self._latest_pose_features = self.pose_feature_extractor.extract(
                self._latest_pose
            )
            self._pose_stats.add(
                self._latest_pose.state,
                self._latest_pose.context.inference_ms,
            )

        face_ran = self._due(
            frame.captured_at_ns,
            self._last_face_run_ns,
            self._face_period_ns,
        )
        if face_ran:
            self._last_face_run_ns = frame.captured_at_ns
            self._latest_face = self.face.observe(frame)
            self._latest_face_features = self.face_feature_extractor.extract(
                self._latest_face
            )
            self._face_stats.add(
                self._latest_face.state,
                self._latest_face.context.inference_ms,
            )

        pose_age = self._age_ms(frame.captured_at_ns, self._latest_pose)
        face_age = self._age_ms(frame.captured_at_ns, self._latest_face)
        return LiveSnapshot(
            frame=frame,
            luminance=self._latest_luminance,
            luminance_ran=luminance_ran,
            luminance_age_ms=max(
                0.0,
                (
                    frame.captured_at_ns
                    - self._latest_luminance.captured_at_ns
                )
                / 1_000_000,
            ),
            pose_observation=self._latest_pose,
            pose_features=self._latest_pose_features,
            pose_ran=pose_ran,
            pose_age_ms=pose_age,
            pose_stale=self._stale(pose_age),
            face_observation=self._latest_face,
            face_features=self._latest_face_features,
            face_ran=face_ran,
            face_age_ms=face_age,
            face_stale=self._stale(face_age),
        )

    def summary(self) -> dict[str, Any]:
        duration_seconds = (
            (self._last_frame_ns - self._first_frame_ns) / 1_000_000_000
            if self._first_frame_ns is not None
            and self._last_frame_ns is not None
            and self._last_frame_ns > self._first_frame_ns
            else 0.0
        )
        return {
            "records_media": False,
            "frames": self._frames,
            "capture_duration_seconds": duration_seconds,
            "effective_capture_fps": (
                (self._frames - 1) / duration_seconds
                if self._frames > 1 and duration_seconds > 0
                else None
            ),
            "dropped_before_total": self._dropped_before,
            "pose": self._pose_stats.summary(),
            "face": self._face_stats.summary(),
            "luminance": {
                "runs": self._luminance_runs,
                "configured_hz": self.schedule.luminance_hz,
            },
            "latest_evidence": self._latest_evidence(),
        }

    def _latest_evidence(self) -> dict[str, Any]:
        pose = self._latest_pose
        pose_features = self._latest_pose_features
        face = self._latest_face
        face_features = self._latest_face_features
        luminance = self._latest_luminance
        return {
            "pose": {
                "state": pose.state.value if pose is not None else "not_run",
                "reason": pose.reason if pose is not None else None,
                "valid_landmark_fraction": (
                    pose_features.valid_landmark_fraction
                    if pose_features is not None
                    else None
                ),
                "shoulder_tilt_deg": (
                    pose_features.shoulder_tilt_deg
                    if pose_features is not None
                    else None
                ),
                "torso_lean_from_vertical_deg": (
                    pose_features.torso_lean_from_vertical_deg
                    if pose_features is not None
                    else None
                ),
                "upper_body_motion_per_second": (
                    pose_features.upper_body_motion_per_second
                    if pose_features is not None
                    else None
                ),
            },
            "face": {
                "state": face.state.value if face is not None else "not_run",
                "reason": face.reason if face is not None else None,
                "geometry_state": (
                    face_features.geometry_state.value
                    if face_features is not None
                    else "not_run"
                ),
                "rotation_state": (
                    face_features.rotation_state.value
                    if face_features is not None
                    else "not_run"
                ),
                "blink_state": (
                    face_features.blink_state.value
                    if face_features is not None
                    else "not_run"
                ),
                "face_bbox_area_ratio": (
                    face_features.face_bbox_area_ratio
                    if face_features is not None
                    else None
                ),
                "raw_rotation_xyz_deg": (
                    face_features.raw_rotation_xyz_deg
                    if face_features is not None
                    else None
                ),
                "eye_blink_left": (
                    face_features.eye_blink_left
                    if face_features is not None
                    else None
                ),
                "eye_blink_right": (
                    face_features.eye_blink_right
                    if face_features is not None
                    else None
                ),
            },
            "luminance": {
                "state": (
                    luminance.state.value if luminance is not None else "not_run"
                ),
                "mean": luminance.mean if luminance is not None else None,
                "p10": luminance.p10 if luminance is not None else None,
                "p90": luminance.p90 if luminance is not None else None,
            },
        }

    @staticmethod
    def _due(current_ns: int, previous_ns: int | None, period_ns: int) -> bool:
        return previous_ns is None or current_ns - previous_ns >= period_ns

    @staticmethod
    def _age_ms(
        current_ns: int,
        observation: PoseObservation | FaceObservation | None,
    ) -> float | None:
        if observation is None:
            return None
        return max(0.0, (current_ns - observation.context.captured_at_ns) / 1_000_000)

    def _stale(self, age_ms: float | None) -> bool:
        return age_ms is None or age_ms > self.schedule.stale_after_ms
