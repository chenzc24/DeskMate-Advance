from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from chip_template_matcher import (
    CENTER_FRACTION,
    DENOMINATIONS,
    ChipTemplateMatcher,
    center_number_view,
    color_signature,
    digit_mask,
)


_CHIP_COLORS = {
    "1": (160, 160, 120),
    "5": (80, 80, 210),
    "10": (210, 100, 50),
    "20": (80, 130, 80),
}


def _chip_with_text(text: str) -> np.ndarray:
    chip = np.full((384, 384, 3), _CHIP_COLORS[text], dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.5 if len(text) == 1 else 2.0
    thickness = 8
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((384 - width) // 2, (384 + height) // 2)
    cv2.putText(chip, text, origin, font, scale, (10, 10, 10), thickness, cv2.LINE_AA)
    return chip


def _library(tmp_path):
    templates = []
    for denomination in DENOMINATIONS:
        template_id = f"synthetic_{denomination}"
        mask = digit_mask(center_number_view(_chip_with_text(str(denomination))))
        mask_path = tmp_path / f"{template_id}.png"
        assert cv2.imwrite(str(mask_path), mask)
        templates.append(
            {
                "template_id": template_id,
                "denomination": denomination,
                "mask_file": mask_path.name,
                "color_signature": color_signature(
                    _chip_with_text(str(denomination))
                ).tolist(),
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "denominations": list(DENOMINATIONS),
                "templates": templates,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_center_number_view_uses_fixed_central_fraction():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    view = center_number_view(image)
    expected = round(300 * CENTER_FRACTION)
    assert view.shape == (expected, expected, 3)


def test_center_number_view_rejects_empty_image():
    with pytest.raises(ValueError):
        center_number_view(np.empty((0, 0), dtype=np.uint8))


def test_digit_mask_ignores_dark_print_outside_central_support():
    view = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(view, (0, 0), (20, 20), (0, 0, 0), -1)
    cv2.rectangle(view, (70, 45), (90, 115), (0, 0, 0), -1)
    mask = digit_mask(view)
    assert np.count_nonzero(mask[:20, :20]) == 0
    assert np.count_nonzero(mask[35:95, 55:75]) > 0


@pytest.mark.parametrize("denomination", DENOMINATIONS)
def test_matcher_recognizes_rotated_synthetic_templates(tmp_path, denomination):
    matcher = ChipTemplateMatcher(
        _library(tmp_path), minimum_score=0.50, minimum_margin=0.01
    )
    chip = _chip_with_text(str(denomination))
    transform = cv2.getRotationMatrix2D((192, 192), 70, 1.0)
    rotated = cv2.warpAffine(chip, transform, (384, 384), borderValue=(235, 235, 235))
    result = matcher.match_normalized_chip(rotated)
    assert result.accepted
    assert result.denomination == denomination


def test_matcher_rejects_blank_chip(tmp_path):
    matcher = ChipTemplateMatcher(_library(tmp_path))
    result = matcher.match_normalized_chip(
        np.full((384, 384, 3), 255, dtype=np.uint8)
    )
    assert not result.accepted
    assert result.denomination is None
