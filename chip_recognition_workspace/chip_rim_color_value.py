"""Binary 10/20 chip denomination evidence from the detected outer rim colour."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from chip_live_value import (
    ChipValueObservation,
    _raw_outer_ring_signature,
    _unknown,
)
from chip_template_matcher import ChipTemplateMatcher, ColorMatch
from rectify_chip_images import (
    _expand_bbox,
)


Image = NDArray[np.uint8]
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RIM_COLOUR_MODEL = (
    ROOT
    / "models"
    / "assets"
    / "chip_recognition"
    / "rim-colour-binary-10-20-v1"
    / "model.json"
)


@dataclass(frozen=True, slots=True)
class RimColourEvidence:
    bbox_xyxy: tuple[int, int, int, int]
    signature: NDArray[np.float32]
    pattern_feature: NDArray[np.float32]
    ellipse_quality: float
    ellipse_aspect_ratio: float
    ellipse_minor_axis_px: float


@dataclass(frozen=True, slots=True)
class _BoxEllipse:
    center_xy: tuple[float, float]
    axes_wh: tuple[float, float]
    angle_degrees: float
    quality: float
    aspect_ratio: float


class RimColourBinaryClassifier:
    """Runtime-compatible logistic classifier for fixed-design values 10/20."""

    def __init__(
        self,
        _template_library: Path,
        *,
        minimum_score: float = 0.58,
        minimum_margin: float = 0.12,
        allowed_denominations: tuple[int, ...] = (10, 20),
        model_path: Path = DEFAULT_RIM_COLOUR_MODEL,
    ) -> None:
        if tuple(allowed_denominations) != (10, 20):
            raise ValueError("rim-colour binary classifier supports only (10, 20)")
        if not model_path.is_file():
            raise FileNotFoundError(f"rim-colour model is missing: {model_path}")
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if payload.get("active_denominations") != [10, 20]:
            raise ValueError("unexpected rim-colour denomination scope")
        self.mean = np.asarray(payload["feature_standardization"]["mean"], dtype=np.float32)
        self.scale = np.asarray(payload["feature_standardization"]["scale"], dtype=np.float32)
        self.coefficients = np.asarray(
            payload["logistic_regression"]["coefficients"], dtype=np.float32
        )
        self.intercept = float(payload["logistic_regression"]["intercept"])
        feature_count = int(payload["feature_count"])
        if self.mean.shape != (feature_count,) or self.scale.shape != (feature_count,):
            raise ValueError("rim-colour feature standardization shape is invalid")
        if self.coefficients.shape != (feature_count,):
            raise ValueError("rim-colour coefficient shape is invalid")
        self.minimum_score = max(float(minimum_score), 0.56)
        self.minimum_margin = max(float(minimum_margin), 0.12)

    def match_rim_feature(
        self,
        signature: NDArray[np.float32],
        *,
        minimum_score: float | None = None,
        minimum_margin: float | None = None,
    ) -> ColorMatch:
        feature = np.asarray(signature, dtype=np.float32)
        if feature.shape != self.mean.shape:
            raise ValueError(
                f"rim-colour feature must have shape {self.mean.shape}"
            )
        standardized = (feature - self.mean) / np.maximum(self.scale, 1e-6)
        logit = float(np.dot(self.coefficients, standardized) + self.intercept)
        probability_20 = float(
            1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        )
        scores = {10: 1.0 - probability_20, 20: probability_20}
        best = max(scores, key=scores.get)
        best_score = scores[best]
        margin = abs(scores[20] - scores[10])
        score_threshold = (
            self.minimum_score if minimum_score is None else float(minimum_score)
        )
        margin_threshold = (
            self.minimum_margin if minimum_margin is None else float(minimum_margin)
        )
        accepted = bool(
            best_score >= score_threshold and margin >= margin_threshold
        )
        return ColorMatch(
            denomination=best if accepted else None,
            accepted=accepted,
            best_score=round(best_score, 6),
            margin=round(margin, 6),
            scores={key: round(value, 6) for key, value in scores.items()},
            distances={10: round(probability_20, 6), 20: round(1.0 - probability_20, 6)},
        )

    # Kept for interface compatibility with the legacy colour matcher.
    match_color_signature = match_rim_feature


def _rim_pattern_feature(crop: Image, ellipse) -> NDArray[np.float32]:
    """Describe alternating rim colours without averaging them together."""

    height, width = crop.shape[:2]
    outer = np.zeros((height, width), dtype=np.uint8)
    inner = np.zeros((height, width), dtype=np.uint8)
    center = tuple(int(round(value)) for value in ellipse.center_xy)
    half_axes = tuple(max(2, value / 2.0) for value in ellipse.axes_wh)
    cv2.ellipse(
        outer,
        center,
        tuple(max(2, int(round(value * 0.96))) for value in half_axes),
        ellipse.angle_degrees,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        inner,
        center,
        tuple(max(1, int(round(value * 0.68))) for value in half_axes),
        ellipse.angle_degrees,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    annulus = (outer >= 128) & (inner < 128)
    count = int(np.count_nonzero(annulus))
    if count < 48:
        raise ValueError("rim pattern contains too few pixels")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue = hsv[:, :, 0][annulus]
    saturation = hsv[:, :, 1][annulus]
    value = hsv[:, :, 2][annulus]
    lab_a = lab[:, :, 1][annulus]
    lab_b = lab[:, :, 2][annulus]

    hue_hist = np.histogram(
        hue,
        bins=18,
        range=(0, 180),
        weights=saturation.astype(np.float32) / 255.0,
    )[0]
    histograms = [hue_hist]
    for channel in (saturation, value, lab_a, lab_b):
        histograms.append(np.histogram(channel, bins=4, range=(0, 256))[0])
    normalized_histograms = [
        histogram.astype(np.float32) / max(float(np.sum(histogram)), 1.0)
        for histogram in histograms
    ]

    blue = (hue >= 85) & (hue <= 135) & (saturation >= 45)
    green = (hue >= 30) & (hue < 85) & (saturation >= 35)
    flesh = (
        ((hue <= 30) | (hue >= 165))
        & (saturation >= 15)
        & (saturation <= 175)
        & (value >= 80)
    )
    dark_green = (
        (hue >= 30) & (hue <= 95) & (saturation >= 50) & (value <= 135)
    )
    ratios = np.asarray(
        [
            np.mean(blue),
            np.mean(green),
            np.mean(flesh),
            np.mean(dark_green),
            np.mean(value >= 165),
            np.mean(value <= 105),
        ],
        dtype=np.float32,
    )
    return np.concatenate((*normalized_histograms, ratios)).astype(np.float32)


def extract_rim_colour_evidence(
    image: Image,
    bbox_xyxy: Sequence[int],
    *,
    padding: float = 0.04,
    minimum_minor_axis_px: float = 36.0,
    minimum_aspect_ratio: float = 0.30,
    maximum_analysis_size: int = 224,
) -> tuple[RimColourEvidence | None, str | None]:
    """Sample an elliptical outer annulus directly inside the YOLO box."""

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR frame")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    if not 0.0 <= padding <= 0.5:
        raise ValueError("padding must be in [0, 0.5]")
    if minimum_minor_axis_px <= 0.0:
        raise ValueError("minimum_minor_axis_px must be positive")
    if not 0.0 < minimum_aspect_ratio <= 1.0:
        raise ValueError("minimum_aspect_ratio must be in (0, 1]")
    if maximum_analysis_size < 128:
        raise ValueError("maximum_analysis_size must be at least 128")

    bbox = tuple(int(value) for value in bbox_xyxy)
    x1, y1, x2, y2 = _expand_bbox(bbox, image.shape, padding)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None, "crop_too_small"
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    scale = min(1.0, maximum_analysis_size / max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (
                max(1, round(crop.shape[1] * scale)),
                max(1, round(crop.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    bbox_width = max(1.0, float(bbox[2] - bbox[0]))
    bbox_height = max(1.0, float(bbox[3] - bbox[1]))
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
    ellipse = _BoxEllipse(
        center_xy=(
            ((bbox[0] + bbox[2]) / 2.0 - x1) * scale,
            ((bbox[1] + bbox[3]) / 2.0 - y1) * scale,
        ),
        axes_wh=(bbox_width * scale, bbox_height * scale),
        angle_degrees=0.0,
        quality=1.0,
        aspect_ratio=aspect_ratio,
    )
    minor_axis_px = min(bbox_width, bbox_height)
    if minor_axis_px < minimum_minor_axis_px:
        return None, "too_far"
    if aspect_ratio < minimum_aspect_ratio:
        return None, "too_flat"
    try:
        signature = _raw_outer_ring_signature(crop, ellipse)
        pattern_feature = _rim_pattern_feature(crop, ellipse)
    except (cv2.error, ValueError) as exc:
        return None, f"rim_colour_error:{type(exc).__name__}"

    return (
        RimColourEvidence(
            bbox_xyxy=bbox,
            signature=signature,
            pattern_feature=pattern_feature,
            ellipse_quality=float(ellipse.quality),
            ellipse_aspect_ratio=float(ellipse.aspect_ratio),
            ellipse_minor_axis_px=float(minor_axis_px),
        ),
        None,
    )


def recognize_chip_rim_colour(
    matcher: ChipTemplateMatcher,
    image: Image,
    bbox_xyxy: Sequence[int],
    *,
    minimum_minor_axis_px: float = 36.0,
    minimum_aspect_ratio: float = 0.30,
) -> ChipValueObservation:
    """Classify a detected chip as 10/20 using outer-ring colour only."""

    started_ns = time.perf_counter_ns()
    bbox = tuple(int(value) for value in bbox_xyxy)
    evidence, rejection = extract_rim_colour_evidence(
        image,
        bbox,
        minimum_minor_axis_px=minimum_minor_axis_px,
        minimum_aspect_ratio=minimum_aspect_ratio,
    )
    if evidence is None:
        return _unknown(bbox, str(rejection), started_ns)

    match_function = getattr(matcher, "match_rim_feature", matcher.match_color_signature)
    match_input = (
        evidence.pattern_feature
        if hasattr(matcher, "match_rim_feature")
        else evidence.signature
    )
    colour = match_function(
        match_input,
        minimum_score=0.56,
        minimum_margin=0.12,
    )
    accepted = (
        colour.accepted
        and colour.denomination in (10, 20)
        and set(colour.scores) == {10, 20}
    )
    return ChipValueObservation(
        bbox_xyxy=evidence.bbox_xyxy,
        denomination=int(colour.denomination) if accepted else None,
        accepted=accepted,
        score=colour.best_score,
        margin=colour.margin,
        ellipse_quality=evidence.ellipse_quality,
        ellipse_aspect_ratio=evidence.ellipse_aspect_ratio,
        ellipse_minor_axis_px=evidence.ellipse_minor_axis_px,
        latency_ms=round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        decision_reason="rim_colour_binary" if accepted else None,
        rejection_reason=None if accepted else "rim_colour_ambiguous",
        raw_color_denomination=(
            int(colour.denomination) if colour.denomination in (10, 20) else None
        ),
        raw_color_score=colour.best_score,
        raw_color_margin=colour.margin,
    )


# The existing live pipeline expects this function name and keyword contract.
recognize_chip_value = recognize_chip_rim_colour
