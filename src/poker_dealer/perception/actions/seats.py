"""Fixed-ROI seat attribution for the four-seat Laptop gesture pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from poker_dealer.domain import SEAT_ORDER, Seat

from .config import GesturePilotConfig, NormalizedRoi
from .temporal import GestureFrameEvidence


def _overlap(left: NormalizedRoi, right: NormalizedRoi) -> bool:
    return (
        max(left.x_min, right.x_min) < min(left.x_max, right.x_max)
        and max(left.y_min, right.y_min) < min(left.y_max, right.y_max)
    )


@dataclass(frozen=True, slots=True)
class MultiSeatGesturePilotConfig:
    schema_version: str
    pilot_status: str
    layout_status: str
    gesture: GesturePilotConfig
    max_hands: int
    initial_focus_seat: Seat
    seat_rois: Mapping[Seat, NormalizedRoi]
    ui_controls: Mapping[str, str | list[str]]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported multiseat pilot schema version")
        if self.pilot_status != "development_feasibility_only":
            raise ValueError("multiseat pilot must remain development-only")
        if self.layout_status != "laptop_quadrant_pilot_not_target_geometry":
            raise ValueError("Laptop seat geometry must not claim target freeze")
        if set(self.seat_rois) != set(SEAT_ORDER):
            raise ValueError("multiseat pilot requires exactly four seat ROIs")
        if self.max_hands != 4 or self.gesture.model.num_hands != self.max_hands:
            raise ValueError("multiseat pilot must request four hand detections")
        for index, seat in enumerate(SEAT_ORDER):
            for other in SEAT_ORDER[index + 1 :]:
                if _overlap(self.seat_rois[seat], self.seat_rois[other]):
                    raise ValueError(f"seat ROIs overlap: {seat.value}/{other.value}")

    @classmethod
    def from_json(cls, path: str | Path) -> MultiSeatGesturePilotConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parents[2]
        gesture_path = project_root / value["gesture_config_path"]
        gesture = GesturePilotConfig.from_json(gesture_path)
        max_hands = int(value["max_hands"])
        gesture = replace(
            gesture,
            model=replace(gesture.model, num_hands=max_hands),
            calibration_version=(
                f"{gesture.calibration_version}-multiseat-quadrant-v1"
            ),
        )
        return cls(
            schema_version=value["schema_version"],
            pilot_status=value["pilot_status"],
            layout_status=value["layout_status"],
            gesture=gesture,
            max_hands=max_hands,
            initial_focus_seat=Seat(value["initial_focus_seat"]),
            seat_rois={
                Seat(seat): NormalizedRoi(**roi)
                for seat, roi in value["seat_rois_normalized"].items()
            },
            ui_controls=dict(value["ui_controls"]),
        )


@dataclass(frozen=True, slots=True)
class SeatRoutingResult:
    assignments: Mapping[Seat, tuple[GestureFrameEvidence, ...]]
    unassigned: tuple[GestureFrameEvidence, ...]
    ambiguous: tuple[GestureFrameEvidence, ...]


class SeatRoiRouter:
    """Assign full-frame hand centroids to non-overlapping fixed seat ROIs."""

    def __init__(self, seat_rois: Mapping[Seat, NormalizedRoi]) -> None:
        if set(seat_rois) != set(SEAT_ORDER):
            raise ValueError("seat router requires all four seat ROIs")
        self.seat_rois = dict(seat_rois)

    def route(
        self, evidence: tuple[GestureFrameEvidence, ...]
    ) -> SeatRoutingResult:
        assigned: dict[Seat, list[GestureFrameEvidence]] = {
            seat: [] for seat in SEAT_ORDER
        }
        unassigned: list[GestureFrameEvidence] = []
        ambiguous: list[GestureFrameEvidence] = []
        for hand in evidence:
            if not hand.hand_present:
                continue
            if hand.centroid_x is None or hand.centroid_y is None:
                raise ValueError("seat routing requires hand centroids")
            matches = [
                seat
                for seat, roi in self.seat_rois.items()
                if roi.contains(hand.centroid_x, hand.centroid_y)
            ]
            if len(matches) == 1:
                assigned[matches[0]].append(hand)
            elif matches:
                ambiguous.append(hand)
            else:
                unassigned.append(hand)
        return SeatRoutingResult(
            assignments={seat: tuple(items) for seat, items in assigned.items()},
            unassigned=tuple(unassigned),
            ambiguous=tuple(ambiguous),
        )

    @staticmethod
    def focus_evidence(
        routed: SeatRoutingResult,
        focus_seat: Seat,
        *,
        observed_at_ns: int,
        inference_latency_ms: float | None,
    ) -> GestureFrameEvidence:
        hands = routed.assignments[focus_seat]
        if not hands:
            return GestureFrameEvidence(
                observed_at_ns=observed_at_ns,
                hand_present=False,
                hand_in_focus_roi=False,
                gesture_label=None,
                gesture_score=None,
                inference_latency_ms=inference_latency_ms,
                quality_flags=("focus_seat_no_hand",),
            )
        if len(hands) > 1:
            return GestureFrameEvidence(
                observed_at_ns=observed_at_ns,
                hand_present=True,
                hand_in_focus_roi=True,
                gesture_label=None,
                gesture_score=None,
                centroid_x=sum(item.centroid_x or 0.0 for item in hands) / len(hands),
                centroid_y=sum(item.centroid_y or 0.0 for item in hands) / len(hands),
                inference_latency_ms=inference_latency_ms,
                quality_flags=("multiple_hands_in_focus_seat",),
            )
        return replace(
            hands[0],
            hand_in_focus_roi=True,
            quality_flags=tuple(
                dict.fromkeys(hands[0].quality_flags + (f"routed_to:{focus_seat.value}",))
            ),
        )
