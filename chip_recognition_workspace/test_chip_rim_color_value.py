from __future__ import annotations

import numpy as np

import chip_rim_color_value as rim
from chip_template_matcher import ColorMatch


class _Matcher:
    def __init__(self, match: ColorMatch) -> None:
        self.match = match

    def match_color_signature(self, signature, **_kwargs):
        assert signature.shape == (12,)
        return self.match


def _evidence() -> rim.RimColourEvidence:
    return rim.RimColourEvidence(
        bbox_xyxy=(10, 20, 80, 90),
        signature=np.zeros(12, dtype=np.float32),
        pattern_feature=np.zeros(40, dtype=np.float32),
        ellipse_quality=0.9,
        ellipse_aspect_ratio=0.7,
        ellipse_minor_axis_px=70.0,
    )


def test_binary_rim_colour_accepts_ten(monkeypatch):
    monkeypatch.setattr(
        rim,
        "extract_rim_colour_evidence",
        lambda *_args, **_kwargs: (_evidence(), None),
    )
    match = ColorMatch(10, True, 0.97, 0.94, {10: 0.97, 20: 0.03}, {})
    result = rim.recognize_chip_rim_colour(
        _Matcher(match), np.zeros((100, 100, 3), dtype=np.uint8), (10, 20, 80, 90)
    )
    assert result.accepted
    assert result.denomination == 10
    assert result.decision_reason == "rim_colour_binary"
    assert result.digit_denomination is None


def test_binary_rim_colour_rejects_out_of_scope_value(monkeypatch):
    monkeypatch.setattr(
        rim,
        "extract_rim_colour_evidence",
        lambda *_args, **_kwargs: (_evidence(), None),
    )
    match = ColorMatch(5, True, 0.99, 0.98, {5: 0.99, 10: 0.01}, {})
    result = rim.recognize_chip_rim_colour(
        _Matcher(match), np.zeros((100, 100, 3), dtype=np.uint8), (10, 20, 80, 90)
    )
    assert not result.accepted
    assert result.denomination is None
    assert result.rejection_reason == "rim_colour_ambiguous"


def test_binary_rim_colour_preserves_geometry_rejection(monkeypatch):
    monkeypatch.setattr(
        rim,
        "extract_rim_colour_evidence",
        lambda *_args, **_kwargs: (None, "too_flat"),
    )
    match = ColorMatch(10, True, 0.99, 0.98, {10: 0.99, 20: 0.01}, {})
    result = rim.recognize_chip_rim_colour(
        _Matcher(match), np.zeros((100, 100, 3), dtype=np.uint8), (10, 20, 80, 90)
    )
    assert not result.accepted
    assert result.rejection_reason == "too_flat"
