from pathlib import Path

from poker_dealer.domain import VisionSlot
from poker_dealer.perception.cards import (
    CardSlotGeometryConfig,
    DetectedCardBox,
    bind_detections_to_slots,
)


CONFIG = Path("configs/perception/card_slots_development_v1.json")


def test_geometry_defines_all_thirteen_slots_and_is_not_target_validated() -> None:
    geometry = CardSlotGeometryConfig.from_json(CONFIG)
    assert set(geometry.slots) == set(VisionSlot)
    assert len(geometry.slots) == 13
    assert geometry.target_geometry_validated is False


def test_multi_card_binding_is_one_to_one_and_rejects_same_slot_collision() -> None:
    geometry = CardSlotGeometryConfig.from_json(CONFIG)
    first = geometry.roi_for(VisionSlot.BOARD_FLOP_1)
    second = geometry.roi_for(VisionSlot.BOARD_FLOP_2)
    detections = [
        DetectedCardBox(first.x_min, first.y_min, first.x_max, first.y_max, 0.9),
        DetectedCardBox(second.x_min, second.y_min, second.x_max, second.y_max, 0.8),
    ]
    result = bind_detections_to_slots(detections, geometry)
    assert result.accepted
    assert result.bindings == {
        VisionSlot.BOARD_FLOP_1: 0,
        VisionSlot.BOARD_FLOP_2: 1,
    }

    collision = bind_detections_to_slots([detections[0], detections[0]], geometry)
    assert not collision.accepted
    assert collision.ambiguous_slots == (VisionSlot.BOARD_FLOP_1,)
