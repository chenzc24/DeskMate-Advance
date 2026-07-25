from __future__ import annotations

from poker_dealer.domain import HandPhase, Seat
from poker_dealer.runtime.live_perception import LivePerceptionSession
from poker_dealer.runtime.ports import RuntimeObservationContext


def _context(hand_id: str) -> RuntimeObservationContext:
    return RuntimeObservationContext(
        session_id="four-player-session",
        hand_id=hand_id,
        state_version=9,
        hand_phase=HandPhase.AWAITING_ACTION,
        focus_seat=Seat.D,
        legal_actions=(),
        required_card_slots=(),
    )


class _ResettableCardTemporal:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class _FrameSource:
    def __init__(self) -> None:
        self.card_boxes = object()

    def set_card_detections(self, boxes: tuple[object, ...]) -> None:
        self.card_boxes = boxes


def test_action_context_includes_hand_id_across_consecutive_hands() -> None:
    session = object.__new__(LivePerceptionSession)
    session._identity_context = None
    resets: list[str] = []
    session._reset_action_context = lambda: resets.append("reset")  # type: ignore[method-assign]

    session._ensure_action_context(_context("hand-001"))
    session._ensure_action_context(_context("hand-001"))
    session._ensure_action_context(_context("hand-002"))

    assert resets == ["reset", "reset"]
    assert session._identity_context == ("hand-002", 9, Seat.D)


def test_card_temporal_and_batch_cache_reset_at_new_hand_boundary() -> None:
    session = object.__new__(LivePerceptionSession)
    temporal = _ResettableCardTemporal()
    frame_source = _FrameSource()
    session.card_temporal = temporal  # type: ignore[assignment]
    session.frame_source = frame_source  # type: ignore[assignment]
    session._card_hand_id = None
    session._card_batch_cache_key = ("camera", 4, ())
    session._card_batch_cache = {object(): object()}  # type: ignore[dict-item]

    session._ensure_card_hand_context("hand-001")
    session._card_batch_cache[object()] = object()  # type: ignore[index]
    session._ensure_card_hand_context("hand-001")
    session._ensure_card_hand_context("hand-002")

    assert temporal.resets == 2
    assert session._card_hand_id == "hand-002"
    assert session._card_batch_cache_key is None
    assert session._card_batch_cache == {}
    assert frame_source.card_boxes == ()
