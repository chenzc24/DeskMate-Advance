"""Debug overlays for line-perception development."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from .observations import LineDetectionResult
except ImportError:
    from observations import LineDetectionResult


Image = NDArray[np.uint8]


def render_debug(frame: Image, result: LineDetectionResult) -> Image:
    output = frame.copy()
    height, width = output.shape[:2]
    observation = result.observation

    cv2.rectangle(
        output,
        (0, result.roi_top),
        (width - 1, height - 1),
        (255, 180, 0),
        2,
    )
    for boundary in result.band_boundaries_px[1:-1]:
        cv2.line(output, (0, boundary), (width - 1, boundary), (100, 100, 100), 1)

    center_x = width // 2
    cv2.line(
        output,
        (center_x, result.roi_top),
        (center_x, height - 1),
        (255, 255, 0),
        1,
    )
    if result.points_px:
        points = np.asarray(result.points_px, dtype=np.int32).reshape((-1, 1, 2))
        if len(points) >= 2:
            cv2.polylines(output, [points], False, (0, 255, 255), 3)
        for point in result.points_px:
            cv2.circle(output, point, 8, (0, 255, 0), -1)

    state_color = (0, 0, 255) if observation.line_lost else (0, 220, 0)
    state = "LINE LOST" if observation.line_lost else "TRACKED"
    cv2.putText(
        output,
        f"{state} confidence={observation.confidence:.2f}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        state_color,
        2,
        cv2.LINE_AA,
    )
    if not observation.line_lost:
        cv2.putText(
            output,
            (
                f"offset={observation.offset:+.3f} "
                f"heading={observation.heading:+.3f} "
                f"curve={observation.curvature:+.3f}"
            ),
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif observation.rejection_reason:
        cv2.putText(
            output,
            observation.rejection_reason,
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            state_color,
            2,
            cv2.LINE_AA,
        )
    return output


def colorize_mask(result: LineDetectionResult, frame_width: int) -> Image:
    mask = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    if mask.shape[1] != frame_width:
        mask = cv2.resize(mask, (frame_width, mask.shape[0]))
    return mask
