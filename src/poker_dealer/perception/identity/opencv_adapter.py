"""OpenCV YuNet/SFace boundary returning in-memory face features."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket

from .config import FaceIdentityConfig


@dataclass(frozen=True, slots=True)
class DetectedFaceFeature:
    observed_at_ns: int
    bbox_xywh: tuple[int, int, int, int]
    detection_score: float
    embedding: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.observed_at_ns < 0:
            raise ValueError("face timestamp must be non-negative")
        if not 0.0 <= self.detection_score <= 1.0:
            raise ValueError("face detection score must be in [0, 1]")
        if any(value < 0 for value in self.bbox_xywh):
            raise ValueError("face bounding box values must be non-negative")
        if self.embedding.ndim != 1 or self.embedding.size == 0:
            raise ValueError("face embedding must be a non-empty vector")
        if not np.isfinite(self.embedding).all():
            raise ValueError("face embedding must be finite")
        if not np.isclose(float(np.linalg.norm(self.embedding)), 1.0, atol=1e-4):
            raise ValueError("face embedding must be L2-normalized")


@dataclass(frozen=True, slots=True)
class FaceFrameEvidence:
    observed_at_ns: int
    detected_face_count: int
    low_quality_face_count: int
    features: tuple[DetectedFaceFeature, ...]
    inference_latency_ms: float


class FaceIdentityModelError(RuntimeError):
    """Raised when local face detection/embedding cannot run."""


class OpenCvFaceIdentityAdapter:
    """Detect, align and embed faces without persisting frames or features."""

    def __init__(self, config: FaceIdentityConfig) -> None:
        config.verify_assets()
        self.config = config
        options = config.detector_options
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(config.detector.asset_path),
                "",
                tuple(options["input_size"]),  # type: ignore[arg-type]
                float(options["score_threshold"]),
                float(options["nms_threshold"]),
                int(options["top_k"]),
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(config.embedder.asset_path), ""
            )
        except cv2.error as exc:
            raise FaceIdentityModelError(f"failed to load face models: {exc}") from exc

    def analyze(self, frame: FramePacket) -> FaceFrameEvidence:
        if frame.color_space is not ColorSpace.BGR:
            raise FaceIdentityModelError("face adapter requires a BGR FramePacket")
        image = np.asarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise FaceIdentityModelError("face adapter requires HxWx3 uint8 input")
        height, width = image.shape[:2]
        started_ns = time.perf_counter_ns()
        try:
            self._detector.setInputSize((width, height))
            _retval, faces = self._detector.detect(image)
        except cv2.error as exc:
            raise FaceIdentityModelError(f"face detection failed: {exc}") from exc
        if faces is None:
            return FaceFrameEvidence(
                frame.captured_at_ns,
                0,
                0,
                (),
                (time.perf_counter_ns() - started_ns) / 1_000_000,
            )

        features: list[DetectedFaceFeature] = []
        low_quality = 0
        minimum_size = int(self.config.detector_options["minimum_face_size_px"])
        for face in faces:
            x, y, box_width, box_height = (int(max(0, value)) for value in face[:4])
            if min(box_width, box_height) < minimum_size:
                low_quality += 1
                continue
            try:
                aligned = self._recognizer.alignCrop(image, face)
                raw_feature = self._recognizer.feature(aligned)
            except cv2.error as exc:
                raise FaceIdentityModelError(f"face embedding failed: {exc}") from exc
            embedding = np.asarray(raw_feature, dtype=np.float32).reshape(-1).copy()
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(norm) or norm <= 0.0:
                low_quality += 1
                continue
            embedding /= norm
            embedding.setflags(write=False)
            features.append(
                DetectedFaceFeature(
                    observed_at_ns=frame.captured_at_ns,
                    bbox_xywh=(x, y, box_width, box_height),
                    detection_score=float(face[14]),
                    embedding=embedding,
                )
            )
        return FaceFrameEvidence(
            observed_at_ns=frame.captured_at_ns,
            detected_face_count=int(len(faces)),
            low_quality_face_count=low_quality,
            features=tuple(features),
            inference_latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
        )
