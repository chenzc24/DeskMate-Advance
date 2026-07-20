"""Framework-independent Part A observation records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
