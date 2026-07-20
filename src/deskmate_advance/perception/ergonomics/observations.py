"""Framework-independent Part A observation records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ObservationState(StrEnum):
    """Validity at the owned perception boundary."""

    VALID = "valid"
    MISSING = "missing"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ObservationContext:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    model_id: str
    inference_ms: float
    dropped_before: int = 0
    model_version: str = "unversioned"
    asset_sha256: str | None = None
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.sequence_id < 0 or self.captured_at_ns < 0:
            raise ValueError("sequence and capture timestamp must be non-negative")
        if not self.model_id.strip() or not self.model_version.strip():
            raise ValueError("model ID and version must not be empty")
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0:
            raise ValueError("inference_ms must be finite and non-negative")
        if self.dropped_before < 0:
            raise ValueError("dropped_before must be non-negative")
        for label, value in (
            ("asset_sha256", self.asset_sha256),
            ("config_sha256", self.config_sha256),
        ):
            if value is not None and not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class Landmark3D:
    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True, slots=True)
class BlendshapeScore:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class PoseObservation:
    context: ObservationContext
    state: ObservationState
    landmarks: tuple[Landmark3D, ...] = ()
    world_landmarks: tuple[Landmark3D, ...] = ()
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is ObservationState.VALID


@dataclass(frozen=True, slots=True)
class FaceObservation:
    context: ObservationContext
    state: ObservationState
    landmarks: tuple[Landmark3D, ...] = ()
    blendshapes: tuple[BlendshapeScore, ...] = ()
    transformation_matrix: tuple[tuple[float, ...], ...] = ()
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is ObservationState.VALID


@dataclass(frozen=True, slots=True)
class LuminanceObservation:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    state: ObservationState
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is ObservationState.VALID


@dataclass(frozen=True, slots=True)
class AudioLevelObservation:
    source_id: str
    window_started_at_ns: int
    window_ended_at_ns: int
    sample_rate_hz: int
    sample_count: int
    state: ObservationState
    rms: float | None
    dbfs: float | None
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is ObservationState.VALID
