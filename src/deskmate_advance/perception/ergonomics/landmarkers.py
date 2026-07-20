"""MediaPipe Pose and Face adapters for Part A.

MediaPipe objects are converted immediately into immutable project records.
The adapters use timestamped VIDEO mode so replay and live scheduling can share
the same observation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import mediapipe as mp
import numpy as np

from deskmate_advance.domain.frame import ColorSpace, FramePacket

from .observations import (
    BlendshapeScore,
    FaceObservation,
    Landmark3D,
    ObservationContext,
    ObservationState,
    PoseObservation,
)


@dataclass(frozen=True, slots=True)
class PoseLandmarkerConfig:
    asset_path: Path
    model_id: str = "mediapipe_pose_landmarker_full"
    model_version: str = "unversioned"
    asset_sha256: str | None = None
    config_sha256: str | None = None
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class FaceLandmarkerConfig:
    asset_path: Path
    model_id: str = "mediapipe_face_landmarker"
    model_version: str = "unversioned"
    asset_sha256: str | None = None
    config_sha256: str | None = None
    num_faces: int = 1
    output_face_blendshapes: bool = True
    output_facial_transformation_matrixes: bool = True
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _image_from_frame(frame: FramePacket) -> mp.Image:
    if frame.color_space is ColorSpace.BGR:
        rgb = frame.image[..., ::-1]
    elif frame.color_space is ColorSpace.RGB:
        rgb = frame.image
    else:  # Defensive if the shared enum grows.
        raise ValueError(f"unsupported color space: {frame.color_space}")
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )


def _landmark(record: Any) -> Landmark3D:
    visibility = getattr(record, "visibility", None)
    presence = getattr(record, "presence", None)
    return Landmark3D(
        x=float(record.x),
        y=float(record.y),
        z=float(record.z),
        visibility=float(visibility) if visibility is not None else None,
        presence=float(presence) if presence is not None else None,
    )


class _TimestampedAdapter:
    def __init__(self) -> None:
        self._last_timestamp_ms: int | None = None

    def _timestamp_ms(self, frame: FramePacket) -> int | None:
        timestamp_ms = frame.captured_at_ns // 1_000_000
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            return None
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms


class PoseLandmarkerAdapter(_TimestampedAdapter):
    """Convert Pose Landmarker results to a Part A PoseObservation."""

    def __init__(self, config: PoseLandmarkerConfig, *, task: Any | None = None) -> None:
        super().__init__()
        self.config = config
        self._task = task or self._create_task(config)

    @staticmethod
    def _create_task(config: PoseLandmarkerConfig) -> Any:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(config.asset_path.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=config.num_poses,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def observe(self, frame: FramePacket) -> PoseObservation:
        started_ns = time.perf_counter_ns()
        timestamp_ms = self._timestamp_ms(frame)
        if timestamp_ms is None:
            return PoseObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason="non_monotonic_millisecond_timestamp",
            )
        try:
            result = self._task.detect_for_video(_image_from_frame(frame), timestamp_ms)
        except Exception as exc:  # Runtime failures become explicit evidence gaps.
            return PoseObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason=f"inference_failed:{type(exc).__name__}",
            )
        if not result.pose_landmarks:
            return PoseObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.MISSING,
                reason="pose_not_detected",
            )
        landmarks = tuple(_landmark(item) for item in result.pose_landmarks[0])
        if len(landmarks) != 33:
            return PoseObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason=f"unexpected_pose_landmark_count:{len(landmarks)}",
            )
        world_groups = getattr(result, "pose_world_landmarks", ())
        world = tuple(_landmark(item) for item in world_groups[0]) if world_groups else ()
        return PoseObservation(
            context=self._context(frame, started_ns),
            state=ObservationState.VALID,
            landmarks=landmarks,
            world_landmarks=world,
        )

    def _context(self, frame: FramePacket, started_ns: int) -> ObservationContext:
        return ObservationContext(
            source_id=frame.source_id,
            sequence_id=frame.sequence_id,
            captured_at_ns=frame.captured_at_ns,
            model_id=self.config.model_id,
            inference_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            dropped_before=frame.dropped_before,
            model_version=self.config.model_version,
            asset_sha256=self.config.asset_sha256,
            config_sha256=self.config.config_sha256,
        )

    def close(self) -> None:
        close = getattr(self._task, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> PoseLandmarkerAdapter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class FaceLandmarkerAdapter(_TimestampedAdapter):
    """Convert Face Landmarker results to a Part A FaceObservation."""

    def __init__(self, config: FaceLandmarkerConfig, *, task: Any | None = None) -> None:
        super().__init__()
        self.config = config
        self._task = task or self._create_task(config)

    @staticmethod
    def _create_task(config: FaceLandmarkerConfig) -> Any:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(config.asset_path.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=config.num_faces,
            output_face_blendshapes=config.output_face_blendshapes,
            output_facial_transformation_matrixes=(
                config.output_facial_transformation_matrixes
            ),
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def observe(self, frame: FramePacket) -> FaceObservation:
        started_ns = time.perf_counter_ns()
        timestamp_ms = self._timestamp_ms(frame)
        if timestamp_ms is None:
            return FaceObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason="non_monotonic_millisecond_timestamp",
            )
        try:
            result = self._task.detect_for_video(_image_from_frame(frame), timestamp_ms)
        except Exception as exc:
            return FaceObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason=f"inference_failed:{type(exc).__name__}",
            )
        if not result.face_landmarks:
            return FaceObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.MISSING,
                reason="face_not_detected",
            )
        landmarks = tuple(_landmark(item) for item in result.face_landmarks[0])
        if len(landmarks) != 478:
            return FaceObservation(
                context=self._context(frame, started_ns),
                state=ObservationState.ERROR,
                reason=f"unexpected_face_landmark_count:{len(landmarks)}",
            )
        blendshape_groups = getattr(result, "face_blendshapes", ())
        blendshapes = (
            tuple(
                BlendshapeScore(
                    name=str(item.category_name),
                    score=float(item.score),
                )
                for item in blendshape_groups[0]
            )
            if blendshape_groups
            else ()
        )
        matrix_groups = getattr(result, "facial_transformation_matrixes", ())
        matrix = (
            tuple(tuple(float(value) for value in row) for row in matrix_groups[0])
            if matrix_groups
            else ()
        )
        return FaceObservation(
            context=self._context(frame, started_ns),
            state=ObservationState.VALID,
            landmarks=landmarks,
            blendshapes=blendshapes,
            transformation_matrix=matrix,
        )

    def _context(self, frame: FramePacket, started_ns: int) -> ObservationContext:
        return ObservationContext(
            source_id=frame.source_id,
            sequence_id=frame.sequence_id,
            captured_at_ns=frame.captured_at_ns,
            model_id=self.config.model_id,
            inference_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            dropped_before=frame.dropped_before,
            model_version=self.config.model_version,
            asset_sha256=self.config.asset_sha256,
            config_sha256=self.config.config_sha256,
        )

    def close(self) -> None:
        close = getattr(self._task, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> FaceLandmarkerAdapter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
