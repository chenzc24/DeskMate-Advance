"""Configuration and provenance checks for the Laptop gesture pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from poker_dealer.domain import PlayerActionType


@dataclass(frozen=True, slots=True)
class NormalizedRoi:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("ROI coordinates must be in [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI minimums must be below maximums")

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True, slots=True)
class GestureConfirmationConfig:
    minimum_score: float
    minimum_stable_frames: int
    minimum_stable_duration_ms: int
    release_frames: int
    cooldown_ms: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0, 1]")
        if self.minimum_stable_frames <= 0 or self.release_frames <= 0:
            raise ValueError("frame thresholds must be positive")
        if self.minimum_stable_duration_ms < 0 or self.cooldown_ms < 0:
            raise ValueError("duration thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class GestureModelConfig:
    model_id: str
    version: str
    asset_path: Path
    sha256: str
    framework: str
    framework_version: str
    num_hands: int

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.version.strip():
            raise ValueError("model ID and version are required")
        if len(self.sha256) != 64:
            raise ValueError("model SHA-256 must contain 64 hexadecimal digits")
        int(self.sha256, 16)
        if not 1 <= self.num_hands <= 4:
            raise ValueError("the Stage 2A pilot supports one to four hands")


@dataclass(frozen=True, slots=True)
class GesturePilotConfig:
    schema_version: str
    pilot_status: str
    model: GestureModelConfig
    focus_roi: NormalizedRoi
    gesture_to_action: Mapping[str, PlayerActionType]
    ignored_gestures: frozenset[str]
    confirmation: GestureConfirmationConfig
    calibration_version: str
    save_frames: bool
    max_seconds_default: int
    camera: Mapping[str, int | float | str]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported action pilot schema version")
        if self.pilot_status != "development_feasibility_only":
            raise ValueError("pilot must remain development feasibility only")
        if set(self.gesture_to_action.values()) != set(PlayerActionType):
            raise ValueError("gesture mapping must cover each poker action exactly")
        if len(self.gesture_to_action) != len(PlayerActionType):
            raise ValueError("each poker action requires one unique pilot gesture")
        if set(self.gesture_to_action) & self.ignored_gestures:
            raise ValueError("mapped and ignored gestures must not overlap")
        if self.save_frames:
            raise ValueError("the live feasibility pilot must not save frames")
        if self.max_seconds_default <= 0:
            raise ValueError("max_seconds_default must be positive")
        if not self.calibration_version.strip():
            raise ValueError("calibration_version is required")

    @classmethod
    def from_json(cls, path: str | Path) -> GesturePilotConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parents[2]
        model = value["model"]
        runtime = value["runtime"]
        return cls(
            schema_version=value["schema_version"],
            pilot_status=value["pilot_status"],
            model=GestureModelConfig(
                model_id=model["model_id"],
                version=model["version"],
                asset_path=project_root / model["asset_path"],
                sha256=model["sha256"].lower(),
                framework=model["framework"],
                framework_version=model["framework_version"],
                num_hands=model["num_hands"],
            ),
            focus_roi=NormalizedRoi(**value["focus_roi_normalized"]),
            gesture_to_action={
                label: PlayerActionType(action)
                for label, action in value["gesture_to_action"].items()
            },
            ignored_gestures=frozenset(value["ignored_canned_gestures"]),
            confirmation=GestureConfirmationConfig(**value["confirmation"]),
            calibration_version=runtime["calibration_version"],
            save_frames=runtime["save_frames"],
            max_seconds_default=runtime["max_seconds_default"],
            camera=dict(value["camera"]),
        )

    def verify_model_asset(self) -> str:
        if not self.model.asset_path.is_file():
            raise FileNotFoundError(
                f"gesture model asset is missing: {self.model.asset_path}"
            )
        digest = hashlib.sha256()
        with self.model.asset_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != self.model.sha256:
            raise ValueError(
                "gesture model SHA-256 mismatch: "
                f"expected {self.model.sha256}, got {actual}"
            )
        return actual
