"""Neutral card identities and perception observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rank(StrEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Suit(StrEnum):
    CLUBS = "clubs"
    DIAMONDS = "diamonds"
    HEARTS = "hearts"
    SPADES = "spades"


@dataclass(frozen=True, slots=True, order=True)
class CardIdentity:
    rank: Rank
    suit: Suit


class ObservationStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    EMPTY = "empty"
    FACE_DOWN = "face_down"
    FACE_UP_UNCONFIRMED = "face_up_unconfirmed"
    OCCLUDED = "occluded"


class VisionSlot(StrEnum):
    BOARD_FLOP_1 = "board_flop_1"
    BOARD_FLOP_2 = "board_flop_2"
    BOARD_FLOP_3 = "board_flop_3"
    BOARD_TURN = "board_turn"
    BOARD_RIVER = "board_river"
    SEAT_A_HOLE_1 = "seat_a_hole_1"
    SEAT_A_HOLE_2 = "seat_a_hole_2"
    SEAT_B_HOLE_1 = "seat_b_hole_1"
    SEAT_B_HOLE_2 = "seat_b_hole_2"
    SEAT_C_HOLE_1 = "seat_c_hole_1"
    SEAT_C_HOLE_2 = "seat_c_hole_2"
    SEAT_D_HOLE_1 = "seat_d_hole_1"
    SEAT_D_HOLE_2 = "seat_d_hole_2"


@dataclass(frozen=True, slots=True)
class CardObservation:
    """One slot observation; unknown is never converted to card absence."""

    observation_id: str
    slot_id: VisionSlot
    observed_at_ns: int
    status: ObservationStatus
    card: CardIdentity | None
    confidence: float | None
    model_version: str
    calibration_version: str
    stable_frames: int
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty")
        if self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be non-negative")
        if self.status is ObservationStatus.CONFIRMED and self.card is None:
            raise ValueError("confirmed observations require a card")
        if self.status is ObservationStatus.CONFIRMED and self.confidence is None:
            raise ValueError("confirmed observations require confidence")
        if self.status is not ObservationStatus.CONFIRMED and self.card is not None:
            raise ValueError("only confirmed observations may carry a card")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not self.calibration_version.strip():
            raise ValueError("calibration_version must not be empty")
        if self.stable_frames <= 0:
            raise ValueError("stable_frames must be positive")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("quality_flags cannot contain empty values")
