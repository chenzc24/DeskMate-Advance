"""Temporal confirmation boundary for fixed-slot card evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_dealer.domain import (
    CardIdentity,
    CardObservation,
    ObservationStatus,
    VisionSlot,
)

from .config import CardPilotConfig
from .opencv_adapter import CardFrameEvidence


@dataclass(slots=True)
class _SlotCandidate:
    card: CardIdentity
    first_observed_ns: int
    last_observed_ns: int
    stable_frames: int


class CardObservationPromoter:
    """Promote repeated frame evidence without making game-state decisions."""

    def __init__(self, config: CardPilotConfig) -> None:
        self.config = config
        self._candidates: dict[VisionSlot, _SlotCandidate] = {}
        self._confirmed: dict[VisionSlot, CardIdentity] = {}

    def process(
        self, slot: VisionSlot, evidence: CardFrameEvidence
    ) -> CardObservation:
        observation_id = (
            f"card:{evidence.source_id}:{evidence.sequence_id}:{slot.value}"
        )
        if evidence.card is None:
            self._candidates.pop(slot, None)
            return CardObservation(
                observation_id,
                slot,
                evidence.observed_at_ns,
                ObservationStatus.UNKNOWN,
                None,
                None,
                self.config.model_version,
                self.config.calibration_version,
                1,
                evidence.quality_flags,
            )

        candidate = self._candidates.get(slot)
        stale_ns = self.config.confirmation.stale_after_ms * 1_000_000
        stale_reset = (
            candidate is not None
            and evidence.observed_at_ns - candidate.last_observed_ns > stale_ns
        )
        if candidate is None or stale_reset or candidate.card != evidence.card:
            candidate = _SlotCandidate(
                evidence.card,
                evidence.observed_at_ns,
                evidence.observed_at_ns,
                1,
            )
            self._candidates[slot] = candidate
        else:
            if evidence.observed_at_ns < candidate.last_observed_ns:
                self._candidates.pop(slot, None)
                return CardObservation(
                    observation_id,
                    slot,
                    evidence.observed_at_ns,
                    ObservationStatus.UNKNOWN,
                    None,
                    None,
                    self.config.model_version,
                    self.config.calibration_version,
                    1,
                    ("non_monotonic_timestamp",),
                )
            candidate.last_observed_ns = evidence.observed_at_ns
            candidate.stable_frames += 1

        stable_duration_ns = (
            evidence.observed_at_ns - candidate.first_observed_ns
        )
        confirmed = (
            candidate.stable_frames
            >= self.config.confirmation.minimum_stable_frames
            and stable_duration_ns
            >= self.config.confirmation.minimum_stable_duration_ms * 1_000_000
        )
        flags = evidence.quality_flags
        if stale_reset:
            flags = (*flags, "stale_candidate_reset")
        if not confirmed:
            return CardObservation(
                observation_id,
                slot,
                evidence.observed_at_ns,
                ObservationStatus.FACE_UP_UNCONFIRMED,
                None,
                None,
                self.config.model_version,
                self.config.calibration_version,
                candidate.stable_frames,
                flags,
            )

        duplicate_slot = next(
            (
                other_slot
                for other_slot, card in self._confirmed.items()
                if other_slot != slot and card == evidence.card
            ),
            None,
        )
        if duplicate_slot is not None:
            self._candidates.pop(slot, None)
            return CardObservation(
                observation_id,
                slot,
                evidence.observed_at_ns,
                ObservationStatus.UNKNOWN,
                None,
                None,
                self.config.model_version,
                self.config.calibration_version,
                candidate.stable_frames,
                (*flags, f"duplicate_card_identity:{duplicate_slot.value}"),
            )
        self._confirmed[slot] = evidence.card
        return CardObservation(
            observation_id,
            slot,
            evidence.observed_at_ns,
            ObservationStatus.CONFIRMED,
            evidence.card,
            evidence.confidence,
            self.config.model_version,
            self.config.calibration_version,
            candidate.stable_frames,
            flags,
        )

    def reset(self) -> None:
        self._candidates.clear()
        self._confirmed.clear()


def card_observation_to_dict(observation: CardObservation) -> dict[str, Any]:
    """Serialize exactly the frozen CardObservation 1.0 contract."""

    card = None
    if observation.card is not None:
        card = {
            "rank": observation.card.rank.value,
            "suit": observation.card.suit.value,
        }
    return {
        "schema_version": "1.0",
        "observation_id": observation.observation_id,
        "slot_id": observation.slot_id.value,
        "observed_at_ns": observation.observed_at_ns,
        "status": observation.status.value,
        "card": card,
        "confidence": observation.confidence,
        "model_version": observation.model_version,
        "calibration_version": observation.calibration_version,
        "stable_frames": observation.stable_frames,
        "quality_flags": list(observation.quality_flags),
    }
