"""MediaPipe pose boundary and a camera-centric target person tracker."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket

from .config import ActorAttributionConfig

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - clear runtime diagnostic
    mp = None  # type: ignore[assignment]


class PoseModelError(RuntimeError):
    """Raised when the offline pose model cannot be loaded or executed."""


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    x: float
    y: float
    visibility: float
    presence: float

    @property
    def confidence(self) -> float:
        return min(self.visibility, self.presence)


@dataclass(frozen=True, slots=True)
class PersonPoseEvidence:
    detector_index: int
    nose: LandmarkPoint
    left_shoulder: LandmarkPoint
    right_shoulder: LandmarkPoint
    left_wrist: LandmarkPoint
    right_wrist: LandmarkPoint
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def body_anchor(self) -> tuple[float, float]:
        return (
            (self.left_shoulder.x + self.right_shoulder.x) / 2.0,
            (self.left_shoulder.y + self.right_shoulder.y) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class PoseFrameEvidence:
    observed_at_ns: int
    poses: tuple[PersonPoseEvidence, ...]
    inference_latency_ms: float


class MediaPipePoseAdapter:
    def __init__(self, config: ActorAttributionConfig) -> None:
        if mp is None:
            raise PoseModelError("MediaPipe is unavailable")
        config.verify_pose_asset()
        self.config = config
        self._last_video_timestamp_ms = -1
        try:
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(config.pose_asset_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=config.num_poses,
                    min_pose_detection_confidence=config.minimum_pose_confidence,
                    min_pose_presence_confidence=config.minimum_pose_confidence,
                    min_tracking_confidence=config.minimum_pose_confidence,
                )
            )
        except (RuntimeError, ValueError) as exc:
            raise PoseModelError(f"failed to load pose model: {exc}") from exc

    def recognize(self, frame: FramePacket) -> PoseFrameEvidence:
        if frame.color_space is not ColorSpace.BGR:
            raise PoseModelError("pose adapter requires a BGR FramePacket")
        image = np.asarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise PoseModelError("pose adapter requires an HxWx3 uint8 frame")
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        timestamp_ms = max(
            self._last_video_timestamp_ms + 1,
            frame.captured_at_ns // 1_000_000,
        )
        self._last_video_timestamp_ms = timestamp_ms
        started_ns = time.perf_counter_ns()
        try:
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )
        except (RuntimeError, ValueError) as exc:
            raise PoseModelError(f"pose inference failed: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        return PoseFrameEvidence(
            observed_at_ns=frame.captured_at_ns,
            poses=self._to_owned_poses(result),
            inference_latency_ms=latency_ms,
        )

    @staticmethod
    def _point(landmark: Any) -> LandmarkPoint:
        return LandmarkPoint(
            float(landmark.x),
            float(landmark.y),
            float(landmark.visibility or 0.0),
            float(landmark.presence or 0.0),
        )

    def _to_owned_poses(self, result: Any) -> tuple[PersonPoseEvidence, ...]:
        owned: list[PersonPoseEvidence] = []
        for index, landmarks in enumerate(result.pose_landmarks):
            visible = [
                point
                for point in landmarks
                if min(float(point.visibility or 0.0), float(point.presence or 0.0))
                >= self.config.minimum_pose_confidence
            ]
            values = visible or list(landmarks)
            xs = [float(point.x) for point in values]
            ys = [float(point.y) for point in values]
            owned.append(
                PersonPoseEvidence(
                    detector_index=index,
                    nose=self._point(landmarks[0]),
                    left_shoulder=self._point(landmarks[11]),
                    right_shoulder=self._point(landmarks[12]),
                    left_wrist=self._point(landmarks[15]),
                    right_wrist=self._point(landmarks[16]),
                    bbox_xyxy=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return tuple(owned)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseAdapter":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class TargetPersonTrack:
    track_id: str
    pose: PersonPoseEvidence
    observed_at_ns: int


class TargetPersonTracker:
    """Track the face-selected person without trusting detector result order."""

    def __init__(self, config: ActorAttributionConfig) -> None:
        self.config = config
        self._track: TargetPersonTrack | None = None
        self._sequence = 0

    @property
    def track(self) -> TargetPersonTrack | None:
        return self._track

    def acquire(
        self,
        poses: tuple[PersonPoseEvidence, ...],
        *,
        face_bbox_xywh: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        observed_at_ns: int,
    ) -> TargetPersonTrack | None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        x, y, width, height = face_bbox_xywh
        face_center = (
            (x + width / 2.0) / frame_width,
            (y + height / 2.0) / frame_height,
        )
        ranked = sorted(
            (
                (math.dist(face_center, (pose.nose.x, pose.nose.y)), pose)
                for pose in poses
            ),
            key=lambda item: item[0],
        )
        if not ranked or ranked[0][0] > self.config.maximum_face_pose_distance:
            return None
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < self.config.minimum_assignment_margin
        ):
            return None
        self._sequence += 1
        self._track = TargetPersonTrack(
            f"person-track:{self._sequence}", ranked[0][1], observed_at_ns
        )
        return self._track

    def update(
        self,
        poses: tuple[PersonPoseEvidence, ...],
        *,
        observed_at_ns: int,
    ) -> TargetPersonTrack | None:
        current = self._track
        if current is None:
            return None
        ranked = sorted(
            (
                (math.dist(current.pose.body_anchor, pose.body_anchor), pose)
                for pose in poses
            ),
            key=lambda item: item[0],
        )
        if not ranked or ranked[0][0] > self.config.maximum_track_jump:
            return None
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < self.config.minimum_assignment_margin
        ):
            return None
        self._track = TargetPersonTrack(
            current.track_id, ranked[0][1], observed_at_ns
        )
        return self._track

    def clear(self) -> None:
        self._track = None
