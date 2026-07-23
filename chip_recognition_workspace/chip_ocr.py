"""OCR denomination reading for YOLO-detected poker-chip crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from rapidocr import RapidOCR


ALLOWED_DENOMINATIONS = (1, 5, 10, 20)
_ALLOWED_TEXT = {str(value): value for value in ALLOWED_DENOMINATIONS}
_SHORT_CONFUSIONS = str.maketrans({"O": "0", "I": "1", "L": "1", "|": "1"})


@dataclass(frozen=True, slots=True)
class ChipOcrResult:
    denomination: int | None
    confidence: float | None
    raw_text: str | None
    bbox_xyxy: tuple[int, int, int, int]


def normalize_denomination(text: str) -> int | None:
    """Accept only an exact supported denomination after narrow OCR fixes."""

    compact = "".join(character for character in text.upper() if not character.isspace())
    if len(compact) > 2:
        return None
    normalized = compact.translate(_SHORT_CONFUSIONS)
    return _ALLOWED_TEXT.get(normalized)


def _center_crop(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    margin_y = max(0, int(round(height * 0.08)))
    margin_x = max(0, int(round(width * 0.08)))
    cropped = image[margin_y : height - margin_y, margin_x : width - margin_x]
    if cropped.size == 0:
        return image
    longest = max(cropped.shape[:2])
    scale = max(1.0, 480.0 / max(1, longest))
    return cv2.resize(
        cropped,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _threshold_variant(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class ChipOcrRecognizer:
    """Read fixed chip denominations without assigning unknown text a value."""

    def __init__(self, minimum_confidence: float = 0.45) -> None:
        if not 0.0 < minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in (0, 1]")
        self.minimum_confidence = minimum_confidence
        self._engine = RapidOCR()

        warmup = np.full((120, 240, 3), 255, dtype=np.uint8)
        cv2.putText(
            warmup,
            "20",
            (35, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.4,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        self._engine(warmup)

    def _read_variant(
        self, image: NDArray[np.uint8]
    ) -> tuple[int | None, float | None, str | None]:
        output = self._engine(image)
        texts: Sequence[str] = output.txts or ()
        scores: Sequence[float] = output.scores or ()
        best_raw: tuple[float, str] | None = None
        best_value: tuple[float, int, str] | None = None
        for text, score_value in zip(texts, scores):
            score = float(score_value)
            if best_raw is None or score > best_raw[0]:
                best_raw = (score, str(text))
            denomination = normalize_denomination(str(text))
            if denomination is None or score < self.minimum_confidence:
                continue
            if best_value is None or score > best_value[0]:
                best_value = (score, denomination, str(text))
        if best_value is not None:
            score, denomination, text = best_value
            return denomination, score, text
        if best_raw is not None:
            return None, best_raw[0], best_raw[1]
        return None, None, None

    def recognize(
        self,
        frame: NDArray[np.uint8],
        bbox_xyxy: Sequence[int],
    ) -> ChipOcrResult:
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(x1 + 1, min(frame_width, x2))
        y2 = max(y1 + 1, min(frame_height, y2))
        crop = _center_crop(frame[y1:y2, x1:x2])

        first = self._read_variant(crop)
        if first[0] is not None:
            return ChipOcrResult(*first, (x1, y1, x2, y2))
        second = self._read_variant(_threshold_variant(crop))
        if second[0] is not None:
            return ChipOcrResult(*second, (x1, y1, x2, y2))
        raw = second if (second[1] or 0.0) > (first[1] or 0.0) else first
        return ChipOcrResult(None, raw[1], raw[2], (x1, y1, x2, y2))

    def recognize_many(
        self,
        frame: NDArray[np.uint8],
        boxes: Sequence[Sequence[int]],
    ) -> list[ChipOcrResult]:
        return [self.recognize(frame, box) for box in boxes]
