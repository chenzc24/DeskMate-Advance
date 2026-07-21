"""Configuration and immutable asset checks for the Stage 2B card pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from poker_dealer.domain import CardIdentity, Rank, Suit


_SUIT_BY_CODE = {
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "H": Suit.HEARTS,
    "S": Suit.SPADES,
}
_RANK_BY_CODE = {rank.value: rank for rank in Rank}
_RANK_BY_CODE["10"] = Rank.TEN


def card_identity_from_code(code: str) -> CardIdentity:
    """Translate the pinned model's compact class code into project enums."""

    if len(code) < 2:
        raise ValueError(f"invalid card class code: {code!r}")
    try:
        rank = _RANK_BY_CODE[code[:-1]]
        suit = _SUIT_BY_CODE[code[-1]]
    except KeyError as exc:
        raise ValueError(f"invalid card class code: {code!r}") from exc
    return CardIdentity(rank, suit)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CardModelAsset:
    model_id: str
    version: str
    revision: str
    asset_path: Path
    asset_bytes: int
    sha256: str
    classes_path: Path
    classes_bytes: int
    classes_sha256: str
    framework: str
    framework_version: str
    license_identifier: str
    input_size: tuple[int, int]
    class_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all((self.model_id.strip(), self.version.strip(), self.revision.strip())):
            raise ValueError("card model ID, version and revision are required")
        for label, digest in (
            ("card model", self.sha256),
            ("card class mapping", self.classes_sha256),
        ):
            if len(digest) != 64:
                raise ValueError(f"{label} SHA-256 must contain 64 digits")
            int(digest, 16)
        if self.asset_bytes <= 0 or self.classes_bytes <= 0:
            raise ValueError("card model asset sizes must be positive")
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("card model input size must contain two positive values")
        identities = tuple(card_identity_from_code(code) for code in self.class_codes)
        if len(identities) != 52 or len(set(identities)) != 52:
            raise ValueError("card model class mapping must contain all 52 unique cards")
        if self.license_identifier != "AGPL-3.0":
            raise ValueError("the pinned card weights must retain their AGPL-3.0 license")

    def verify(self) -> tuple[str, str]:
        for label, path in (
            ("card model", self.asset_path),
            ("card class mapping", self.classes_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} asset is missing: {path}")
        if self.asset_path.stat().st_size != self.asset_bytes:
            raise ValueError("card model byte size does not match pinned config")
        if self.classes_path.stat().st_size != self.classes_bytes:
            raise ValueError("card class mapping byte size does not match pinned config")
        model_digest = _sha256_file(self.asset_path)
        if model_digest != self.sha256:
            raise ValueError(
                f"card model SHA-256 mismatch: expected {self.sha256}, got {model_digest}"
            )
        classes_digest = _sha256_file(self.classes_path)
        if classes_digest != self.classes_sha256:
            raise ValueError(
                "card class mapping SHA-256 mismatch: "
                f"expected {self.classes_sha256}, got {classes_digest}"
            )
        sidecar = json.loads(self.classes_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, list) or tuple(sidecar) != self.class_codes:
            raise ValueError("card class mapping sidecar does not match pinned config")
        return model_digest, classes_digest


@dataclass(frozen=True, slots=True)
class CardInferenceConfig:
    minimum_confidence: float
    ambiguity_confidence: float
    minimum_confidence_margin: float
    nms_iou_threshold: float
    maximum_candidate_detections: int
    letterbox_value: int

    def __post_init__(self) -> None:
        probabilities = (
            self.minimum_confidence,
            self.ambiguity_confidence,
            self.minimum_confidence_margin,
            self.nms_iou_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("card inference thresholds must be in [0, 1]")
        if self.ambiguity_confidence > self.minimum_confidence:
            raise ValueError("ambiguity confidence cannot exceed confirmation confidence")
        if not 1 <= self.maximum_candidate_detections <= 8400:
            raise ValueError("maximum candidate detections is out of bounds")
        if not 0 <= self.letterbox_value <= 255:
            raise ValueError("letterbox value must be in [0, 255]")


@dataclass(frozen=True, slots=True)
class CardConfirmationConfig:
    minimum_stable_frames: int
    minimum_stable_duration_ms: int
    stale_after_ms: int

    def __post_init__(self) -> None:
        if self.minimum_stable_frames <= 0:
            raise ValueError("minimum stable frames must be positive")
        if self.minimum_stable_duration_ms < 0 or self.stale_after_ms <= 0:
            raise ValueError("card confirmation durations are invalid")
        if self.minimum_stable_duration_ms > self.stale_after_ms:
            raise ValueError("confirmation duration cannot exceed stale timeout")


@dataclass(frozen=True, slots=True)
class NormalizedCardRoi:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("card ROI coordinates must be in [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("card ROI minimums must be below maximums")


@dataclass(frozen=True, slots=True)
class CardPilotConfig:
    schema_version: str
    pilot_status: str
    model: CardModelAsset
    inference: CardInferenceConfig
    confirmation: CardConfirmationConfig
    fixed_roi: NormalizedCardRoi
    calibration_version: str
    runtime_downloads: bool
    save_frames: bool
    max_seconds_default: int
    camera: Mapping[str, int | float | str]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported card pilot schema version")
        if self.pilot_status != "development_feasibility_only":
            raise ValueError("card pilot must remain development feasibility only")
        if not self.calibration_version.strip():
            raise ValueError("card calibration version is required")
        if self.runtime_downloads:
            raise ValueError("runtime model downloads are prohibited")
        if self.save_frames:
            raise ValueError("the card feasibility pilot must not save frames")
        if self.max_seconds_default <= 0:
            raise ValueError("card pilot max seconds must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> CardPilotConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parents[2]
        model = value["model"]
        runtime = value["runtime"]
        return cls(
            schema_version=str(value["schema_version"]),
            pilot_status=str(value["pilot_status"]),
            model=CardModelAsset(
                model_id=str(model["model_id"]),
                version=str(model["version"]),
                revision=str(model["revision"]),
                asset_path=project_root / str(model["asset_path"]),
                asset_bytes=int(model["asset_bytes"]),
                sha256=str(model["sha256"]).lower(),
                classes_path=project_root / str(model["classes_path"]),
                classes_bytes=int(model["classes_bytes"]),
                classes_sha256=str(model["classes_sha256"]).lower(),
                framework=str(model["framework"]),
                framework_version=str(model["framework_version"]),
                license_identifier=str(model["license_identifier"]),
                input_size=tuple(int(item) for item in model["input_size"]),
                class_codes=tuple(str(item) for item in model["class_codes"]),
            ),
            inference=CardInferenceConfig(**value["inference"]),
            confirmation=CardConfirmationConfig(**value["confirmation"]),
            fixed_roi=NormalizedCardRoi(**value["fixed_roi_normalized"]),
            calibration_version=str(runtime["calibration_version"]),
            runtime_downloads=bool(runtime["runtime_downloads"]),
            save_frames=bool(runtime["save_frames"]),
            max_seconds_default=int(runtime["max_seconds_default"]),
            camera=dict(value["camera"]),
        )

    def verify_assets(self) -> tuple[str, str]:
        return self.model.verify()

    @property
    def model_version(self) -> str:
        return f"{self.model.model_id}@{self.model.version}"
