"""Fixed-slot ROI cropping for the Stage 2B Laptop camera fixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker_dealer.domain import FramePacket, VisionSlot

from .config import NormalizedCardRoi


@dataclass(frozen=True, slots=True)
class PixelCardRoi:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("pixel card ROI must have non-negative origin and size")


def crop_fixed_card_roi(
    frame: FramePacket,
    roi: NormalizedCardRoi,
    slot: VisionSlot,
) -> tuple[FramePacket, PixelCardRoi]:
    """Copy one configured ROI into an immutable project-owned FramePacket."""

    x1 = min(frame.width - 1, max(0, int(round(roi.x_min * frame.width))))
    y1 = min(frame.height - 1, max(0, int(round(roi.y_min * frame.height))))
    x2 = min(frame.width, max(x1 + 1, int(round(roi.x_max * frame.width))))
    y2 = min(frame.height, max(y1 + 1, int(round(roi.y_max * frame.height))))
    image = np.ascontiguousarray(frame.image[y1:y2, x1:x2]).copy()
    image.setflags(write=False)
    pixel_roi = PixelCardRoi(x1, y1, x2 - x1, y2 - y1)
    return (
        FramePacket(
            sequence_id=frame.sequence_id,
            captured_at_ns=frame.captured_at_ns,
            source_id=f"{frame.source_id}:{slot.value}:fixed_roi",
            device_index=frame.device_index,
            width=pixel_roi.width,
            height=pixel_roi.height,
            color_space=frame.color_space,
            nominal_fps=frame.nominal_fps,
            dropped_before=frame.dropped_before,
            image=image,
        ),
        pixel_roi,
    )
