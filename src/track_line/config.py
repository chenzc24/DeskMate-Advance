"""Configuration loading and validation for guide-line detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LineDetectorConfig:
    """Tunable parameters for a high-contrast floor guide line."""

    roi_top_ratio: float = 0.42
    segmentation_mode: str = "gray_otsu"
    line_polarity: str = "dark"
    white_saturation_max: int = 82
    white_value_min: int = 168
    green_hue_min: int = 32
    green_hue_max: int = 96
    green_saturation_min: int = 45
    green_value_min: int = 28
    minimum_green_roi_ratio: float = 0.18
    minimum_green_component_ratio: float = 0.01
    blur_kernel: int = 5
    morphology_kernel: int = 5
    minimum_component_area_ratio: float = 0.006
    maximum_component_area_ratio: float = 0.50
    minimum_component_width_ratio: float = 0.015
    maximum_component_width_ratio: float = 0.60
    target_component_area_ratio: float = 0.10
    minimum_valid_bands: int = 2
    minimum_confidence: float = 0.38
    continuity_scale: float = 0.28
    band_weights: tuple[float, float, float] = (0.15, 0.30, 0.55)

    def __post_init__(self) -> None:
        if not 0.05 <= self.roi_top_ratio < 0.95:
            raise ValueError("roi_top_ratio must be in [0.05, 0.95)")
        if self.segmentation_mode not in {"gray_otsu", "hsv_white_on_green"}:
            raise ValueError(
                "segmentation_mode must be 'gray_otsu' or 'hsv_white_on_green'"
            )
        if self.line_polarity not in {"dark", "light"}:
            raise ValueError("line_polarity must be 'dark' or 'light'")
        for name in (
            "white_saturation_max",
            "white_value_min",
            "green_saturation_min",
            "green_value_min",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be in [0, 255]")
        if not 0 <= self.green_hue_min < self.green_hue_max <= 179:
            raise ValueError("green hue range must satisfy 0 <= min < max <= 179")
        if not 0 <= self.minimum_green_roi_ratio <= 1:
            raise ValueError("minimum_green_roi_ratio must be in [0, 1]")
        if not 0 <= self.minimum_green_component_ratio <= 1:
            raise ValueError("minimum_green_component_ratio must be in [0, 1]")
        for name in ("blur_kernel", "morphology_kernel"):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if not 0 < self.minimum_component_area_ratio < self.maximum_component_area_ratio <= 1:
            raise ValueError("component area ratios are invalid")
        if not 0 < self.minimum_component_width_ratio < self.maximum_component_width_ratio <= 1:
            raise ValueError("component width ratios are invalid")
        if not 0 < self.target_component_area_ratio <= 1:
            raise ValueError("target_component_area_ratio must be in (0, 1]")
        if self.minimum_valid_bands not in {1, 2, 3}:
            raise ValueError("minimum_valid_bands must be 1, 2 or 3")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.continuity_scale <= 0:
            raise ValueError("continuity_scale must be positive")
        if len(self.band_weights) != 3 or any(weight < 0 for weight in self.band_weights):
            raise ValueError("band_weights must contain three non-negative values")
        if sum(self.band_weights) <= 0:
            raise ValueError("band_weights must have a positive sum")

    @classmethod
    def from_json(cls, path: str | Path) -> "LineDetectorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "band_weights" in payload:
            payload["band_weights"] = tuple(float(value) for value in payload["band_weights"])
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
