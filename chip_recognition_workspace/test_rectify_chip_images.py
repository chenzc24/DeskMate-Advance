"""Focused geometry tests for the offline chip rectification utility."""

from __future__ import annotations

import cv2
import numpy as np

from rectify_chip_images import EllipseEvidence, _ellipse_to_circle, _expand_bbox


def test_expand_bbox_adds_bounded_padding() -> None:
    assert _expand_bbox((10.0, 20.0, 110.0, 70.0), (100, 150, 3), 0.2) == (
        0,
        10,
        130,
        80,
    )


def test_ellipse_normalization_produces_a_circle() -> None:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    evidence = EllipseEvidence(
        center_xy=(400.0, 300.0),
        axes_wh=(500.0, 260.0),
        angle_degrees=23.0,
        aspect_ratio=0.52,
        contour_fill_ratio=1.0,
        center_offset_ratio=0.0,
        quality=1.0,
    )
    cv2.ellipse(
        image,
        (evidence.center_xy, evidence.axes_wh, evidence.angle_degrees),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    _, normalized = _ellipse_to_circle(image, evidence, 384)
    foreground = cv2.inRange(normalized, (200, 200, 200), (255, 255, 255))
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    assert len(contours) == 1
    _, _, width, height = cv2.boundingRect(contours[0])
    assert abs(width - height) <= 2
    assert width >= 325
