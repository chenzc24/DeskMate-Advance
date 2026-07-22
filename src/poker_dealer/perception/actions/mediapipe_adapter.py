"""MediaPipe boundary adapter returning only owned scalar gesture evidence."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket

from .config import GesturePilotConfig
from .temporal import GestureFrameEvidence

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised through clear runtime error
    mp = None  # type: ignore[assignment]


class GestureModelError(RuntimeError):
    """Raised when the offline gesture model cannot be loaded or executed."""


class MediaPipeGestureAdapter:
    """Synchronous bounded inference for one or more hands in VIDEO mode."""

    def __init__(self, config: GesturePilotConfig) -> None:
        if mp is None:
            raise GestureModelError(
                "MediaPipe is unavailable; install the project's Stage 2A "
                "environment before running the gesture pilot"
            )
        config.verify_model_asset()
        self.config = config
        self._last_video_timestamp_ms = -1
        self.last_inference_latency_ms: float | None = None
        try:
            self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
                mp.tasks.vision.GestureRecognizerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(config.model.asset_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=config.model.num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        except (RuntimeError, ValueError) as exc:
            raise GestureModelError(f"failed to load gesture model: {exc}") from exc

    def recognize(self, frame: FramePacket) -> GestureFrameEvidence:
        evidence, latency_ms = self._recognize_all_with_latency(frame)
        if evidence:
            return evidence[0]
        return GestureFrameEvidence(
            observed_at_ns=frame.captured_at_ns,
            hand_present=False,
            hand_in_focus_roi=False,
            gesture_label=None,
            gesture_score=None,
            inference_latency_ms=latency_ms,
        )

    def recognize_all(self, frame: FramePacket) -> tuple[GestureFrameEvidence, ...]:
        """Return owned evidence for every detected hand, up to config limit."""

        evidence, _latency_ms = self._recognize_all_with_latency(frame)
        return evidence

    def _recognize_all_with_latency(
        self, frame: FramePacket
    ) -> tuple[tuple[GestureFrameEvidence, ...], float]:
        if frame.color_space is not ColorSpace.BGR:
            raise GestureModelError("gesture adapter requires a BGR FramePacket")
        image = np.asarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise GestureModelError("gesture adapter requires an HxWx3 uint8 frame")
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        timestamp_ms = max(
            self._last_video_timestamp_ms + 1,
            frame.captured_at_ns // 1_000_000,
        )
        self._last_video_timestamp_ms = timestamp_ms
        started = time.perf_counter_ns()
        try:
            result = self._recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                timestamp_ms,
            )
        except (RuntimeError, ValueError) as exc:
            raise GestureModelError(f"gesture inference failed: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.last_inference_latency_ms = latency_ms
        return (
            self._to_owned_evidences(result, frame.captured_at_ns, latency_ms),
            latency_ms,
        )

    def _to_owned_evidences(
        self, result: Any, observed_at_ns: int, latency_ms: float
    ) -> tuple[GestureFrameEvidence, ...]:
        if not result.hand_landmarks:
            return ()

        owned: list[GestureFrameEvidence] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            centroid_x = sum(float(point.x) for point in landmarks) / len(landmarks)
            centroid_y = sum(float(point.y) for point in landmarks) / len(landmarks)
            wrist_x = float(landmarks[0].x)
            wrist_y = float(landmarks[0].y)
            in_roi = self.config.focus_roi.contains(centroid_x, centroid_y)

            label: str | None = None
            score: float | None = None
            if len(result.gestures) > index and result.gestures[index]:
                category = result.gestures[index][0]
                label = str(category.category_name)
                score = float(category.score)
            handedness: str | None = None
            if len(result.handedness) > index and result.handedness[index]:
                handedness = str(result.handedness[index][0].category_name)

            owned.append(
                GestureFrameEvidence(
                    observed_at_ns=observed_at_ns,
                    hand_present=True,
                    hand_in_focus_roi=in_roi,
                    gesture_label=label,
                    gesture_score=score,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    wrist_x=wrist_x,
                    wrist_y=wrist_y,
                    handedness=handedness,
                    detector_index=index,
                    inference_latency_ms=latency_ms,
                )
            )
        return tuple(owned)

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self) -> MediaPipeGestureAdapter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
