"""OpenCV DNN adapter for the pinned LGD 52-class card detector."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from poker_dealer.domain import CardIdentity, ColorSpace, FramePacket

from .config import CardPilotConfig, card_identity_from_code


@dataclass(frozen=True, slots=True)
class CardDetection:
    card: CardIdentity
    confidence: float
    bbox_xywh: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("card detection confidence must be in [0, 1]")
        if any(value < 0 for value in self.bbox_xywh):
            raise ValueError("card detection box values must be non-negative")


@dataclass(frozen=True, slots=True)
class CardFrameEvidence:
    source_id: str
    sequence_id: int
    observed_at_ns: int
    card: CardIdentity | None
    confidence: float | None
    detections: tuple[CardDetection, ...]
    inference_latency_ms: float
    quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or self.sequence_id < 0 or self.observed_at_ns < 0:
            raise ValueError("card frame identity and timestamp must be valid")
        if (self.card is None) != (self.confidence is None):
            raise ValueError("card and confidence must either both exist or both be absent")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("card frame confidence must be in [0, 1]")
        if self.inference_latency_ms < 0.0:
            raise ValueError("card inference latency must be non-negative")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("card quality flags cannot be empty")


class CardModelError(RuntimeError):
    """Raised when the local card model cannot load or execute safely."""


@dataclass(frozen=True, slots=True)
class _Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def _class_aware_nms(
    detections: list[tuple[int, float, tuple[int, int, int, int]]],
    score_threshold: float,
    iou_threshold: float,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    kept: list[tuple[int, float, tuple[int, int, int, int]]] = []
    class_ids = sorted({item[0] for item in detections})
    for class_id in class_ids:
        group = [item for item in detections if item[0] == class_id]
        indices = cv2.dnn.NMSBoxes(
            [item[2] for item in group],
            [item[1] for item in group],
            score_threshold,
            iou_threshold,
        )
        for index in np.asarray(indices).reshape(-1):
            kept.append(group[int(index)])
    kept.sort(key=lambda item: item[1], reverse=True)
    return kept


def decode_card_detections(
    output: NDArray[np.float32],
    config: CardPilotConfig,
    letterbox: _Letterbox,
) -> tuple[CardDetection, ...]:
    """Decode one bounded YOLO11 output without leaking framework tensors."""

    predictions = np.asarray(output)
    if predictions.ndim != 3 or predictions.shape[0] != 1:
        raise CardModelError(f"unexpected card model output shape: {predictions.shape}")
    expected_features = 4 + len(config.model.class_codes)
    if predictions.shape[1] == expected_features:
        rows = predictions[0].T
    elif predictions.shape[2] == expected_features:
        rows = predictions[0]
    else:
        raise CardModelError(f"unexpected card model output shape: {predictions.shape}")
    if not np.isfinite(rows).all():
        raise CardModelError("card model produced non-finite output")

    scores = rows[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    candidate_indices = np.flatnonzero(
        confidences >= config.inference.ambiguity_confidence
    )
    if candidate_indices.size == 0:
        return ()
    ordered = candidate_indices[
        np.argsort(confidences[candidate_indices])[::-1]
    ][: config.inference.maximum_candidate_detections]

    decoded: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for index in ordered:
        center_x, center_y, box_width, box_height = (
            float(value) for value in rows[int(index), :4]
        )
        x1 = (center_x - box_width / 2.0 - letterbox.pad_x) / letterbox.scale
        y1 = (center_y - box_height / 2.0 - letterbox.pad_y) / letterbox.scale
        x2 = (center_x + box_width / 2.0 - letterbox.pad_x) / letterbox.scale
        y2 = (center_y + box_height / 2.0 - letterbox.pad_y) / letterbox.scale
        x1 = min(max(x1, 0.0), float(letterbox.original_width))
        y1 = min(max(y1, 0.0), float(letterbox.original_height))
        x2 = min(max(x2, 0.0), float(letterbox.original_width))
        y2 = min(max(y2, 0.0), float(letterbox.original_height))
        width = max(0, int(round(x2 - x1)))
        height = max(0, int(round(y2 - y1)))
        if width == 0 or height == 0:
            continue
        decoded.append(
            (
                int(class_ids[int(index)]),
                float(confidences[int(index)]),
                (int(round(x1)), int(round(y1)), width, height),
            )
        )
    kept = _class_aware_nms(
        decoded,
        config.inference.ambiguity_confidence,
        config.inference.nms_iou_threshold,
    )
    return tuple(
        CardDetection(
            card_identity_from_code(config.model.class_codes[class_id]),
            confidence,
            bbox,
        )
        for class_id, confidence, bbox in kept
    )


class OpenCvCardRecognitionAdapter:
    """Run bounded fixed-ROI inference and return project-owned frame evidence."""

    def __init__(self, config: CardPilotConfig) -> None:
        config.verify_assets()
        self.config = config
        try:
            self._network = cv2.dnn.readNetFromONNX(str(config.model.asset_path))
        except cv2.error as exc:
            raise CardModelError(f"failed to load card model: {exc}") from exc

    def analyze(self, frame: FramePacket) -> CardFrameEvidence:
        if frame.color_space is not ColorSpace.BGR:
            raise CardModelError("card adapter requires a BGR FramePacket")
        image = np.asarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise CardModelError("card adapter requires HxWx3 uint8 input")
        blob, letterbox = self._preprocess(image)
        started_ns = time.perf_counter_ns()
        try:
            self._network.setInput(blob)
            output = self._network.forward()
        except cv2.error as exc:
            raise CardModelError(f"card inference failed: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        detections = decode_card_detections(output, self.config, letterbox)
        card, confidence, flags = self._select_identity(detections)
        return CardFrameEvidence(
            source_id=frame.source_id,
            sequence_id=frame.sequence_id,
            observed_at_ns=frame.captured_at_ns,
            card=card,
            confidence=confidence,
            detections=detections,
            inference_latency_ms=latency_ms,
            quality_flags=flags,
        )

    def _preprocess(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], _Letterbox]:
        input_width, input_height = self.config.model.input_size
        height, width = image.shape[:2]
        scale = min(input_width / width, input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        value = self.config.inference.letterbox_value
        canvas = np.full((input_height, input_width, 3), value, dtype=np.uint8)
        canvas[
            pad_y : pad_y + resized_height, pad_x : pad_x + resized_width
        ] = resized
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
        )
        return blob, _Letterbox(scale, pad_x, pad_y, width, height)

    def _select_identity(
        self, detections: tuple[CardDetection, ...]
    ) -> tuple[CardIdentity | None, float | None, tuple[str, ...]]:
        if not detections:
            return None, None, ("no_detection",)
        winner = detections[0]
        if winner.confidence < self.config.inference.minimum_confidence:
            return None, None, ("low_confidence",)
        distinct_runner = next(
            (item for item in detections[1:] if item.card != winner.card), None
        )
        if distinct_runner is not None and (
            distinct_runner.confidence >= self.config.inference.minimum_confidence
            or winner.confidence - distinct_runner.confidence
            < self.config.inference.minimum_confidence_margin
        ):
            return None, None, ("ambiguous_card_identity",)
        flags = ()
        if sum(item.card == winner.card for item in detections) > 1:
            flags = ("multiple_same_identity_detections",)
        return winner.card, winner.confidence, flags
