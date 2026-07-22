"""Explicit 13-slot geometry and deterministic multi-card spatial binding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from poker_dealer.domain import VisionSlot

from .config import NormalizedCardRoi


@dataclass(frozen=True, slots=True)
class CardSlotGeometryConfig:
    schema_version: str
    calibration_id: str
    target_geometry_validated: bool
    slots: Mapping[VisionSlot, NormalizedCardRoi]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported card-slot geometry schema version")
        if not self.calibration_id.strip():
            raise ValueError("card-slot calibration ID is required")
        missing = set(VisionSlot) - set(self.slots)
        extra = set(self.slots) - set(VisionSlot)
        if missing or extra:
            raise ValueError(
                "card-slot geometry must define exactly all 13 logical slots: "
                f"missing={sorted(item.value for item in missing)}, "
                f"extra={sorted(str(item) for item in extra)}"
            )
        items = tuple(self.slots.items())
        for index, (first_slot, first) in enumerate(items):
            for second_slot, second in items[index + 1 :]:
                overlaps = (
                    max(first.x_min, second.x_min)
                    < min(first.x_max, second.x_max)
                    and max(first.y_min, second.y_min)
                    < min(first.y_max, second.y_max)
                )
                if overlaps:
                    raise ValueError(
                        "card-slot ROIs must not overlap: "
                        f"{first_slot.value}, {second_slot.value}"
                    )

    @classmethod
    def from_json(cls, path: str | Path) -> CardSlotGeometryConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_slots = value.get("slots")
        if not isinstance(raw_slots, dict):
            raise ValueError("card-slot geometry slots must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            calibration_id=str(value.get("calibration_id", "")),
            target_geometry_validated=bool(
                value.get("target_geometry_validated", False)
            ),
            slots={
                VisionSlot(name): NormalizedCardRoi(**coordinates)
                for name, coordinates in raw_slots.items()
            },
        )

    def roi_for(self, slot: VisionSlot) -> NormalizedCardRoi:
        return self.slots[slot]


@dataclass(frozen=True, slots=True)
class DetectedCardBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    def __post_init__(self) -> None:
        NormalizedCardRoi(self.x_min, self.y_min, self.x_max, self.y_max)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("card detection confidence must be in [0, 1]")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass(frozen=True, slots=True)
class SlotBindingResult:
    bindings: Mapping[VisionSlot, int]
    unbound_detection_indices: tuple[int, ...]
    ambiguous_slots: tuple[VisionSlot, ...]

    @property
    def accepted(self) -> bool:
        return not self.unbound_detection_indices and not self.ambiguous_slots


def bind_detections_to_slots(
    detections: Sequence[DetectedCardBox],
    geometry: CardSlotGeometryConfig,
) -> SlotBindingResult:
    """Bind detection centres to one logical slot; ambiguity stays rejected."""

    candidates: dict[VisionSlot, list[int]] = {}
    unbound: list[int] = []
    for index, detection in enumerate(detections):
        x, y = detection.center
        containing = [
            slot
            for slot, roi in geometry.slots.items()
            if roi.x_min <= x <= roi.x_max and roi.y_min <= y <= roi.y_max
        ]
        if len(containing) != 1:
            unbound.append(index)
            continue
        candidates.setdefault(containing[0], []).append(index)
    ambiguous = tuple(
        sorted(
            (slot for slot, indices in candidates.items() if len(indices) != 1),
            key=lambda item: item.value,
        )
    )
    bindings = {
        slot: indices[0]
        for slot, indices in candidates.items()
        if len(indices) == 1
    }
    return SlotBindingResult(bindings, tuple(unbound), ambiguous)


__all__ = [
    "CardSlotGeometryConfig",
    "DetectedCardBox",
    "SlotBindingResult",
    "bind_detections_to_slots",
]
