"""Context-safe scripted and exact recorded evidence sources."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    CardObservation,
    ObservationStatus,
    PlayerActionObservation,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    TableRole,
    VisionSlot,
)
from poker_dealer.perception.attribution import ActorBinding
from poker_dealer.perception.identity import (
    FaceIdentityObservation,
    FaceIdentityState,
)

from .event_log import RuntimeEventLog, RuntimeLogRecord
from .ports import ActionEvidence, RuntimeObservationContext
from .registration import FrozenSessionRoster, RegisteredParticipant


DEFAULT_SHOWDOWN_CARDS = {
    VisionSlot.BOARD_FLOP_1: CardIdentity(Rank.ACE, Suit.SPADES),
    VisionSlot.BOARD_FLOP_2: CardIdentity(Rank.KING, Suit.HEARTS),
    VisionSlot.BOARD_FLOP_3: CardIdentity(Rank.QUEEN, Suit.DIAMONDS),
    VisionSlot.BOARD_TURN: CardIdentity(Rank.JACK, Suit.CLUBS),
    VisionSlot.BOARD_RIVER: CardIdentity(Rank.TWO, Suit.SPADES),
    VisionSlot.SEAT_A_HOLE_1: CardIdentity(Rank.THREE, Suit.CLUBS),
    VisionSlot.SEAT_A_HOLE_2: CardIdentity(Rank.FOUR, Suit.CLUBS),
    VisionSlot.SEAT_B_HOLE_1: CardIdentity(Rank.FIVE, Suit.CLUBS),
    VisionSlot.SEAT_B_HOLE_2: CardIdentity(Rank.SIX, Suit.CLUBS),
    VisionSlot.SEAT_C_HOLE_1: CardIdentity(Rank.SEVEN, Suit.CLUBS),
    VisionSlot.SEAT_C_HOLE_2: CardIdentity(Rank.EIGHT, Suit.CLUBS),
    VisionSlot.SEAT_D_HOLE_1: CardIdentity(Rank.NINE, Suit.CLUBS),
    VisionSlot.SEAT_D_HOLE_2: CardIdentity(Rank.TEN, Suit.CLUBS),
}


class StepClock:
    def __init__(self, *, start_ns: int = 0, step_ns: int = 1_000_000) -> None:
        if start_ns < 0 or step_ns <= 0:
            raise ValueError("step clock values must be valid")
        self.value = start_ns
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


def default_replay_roster(
    session_id: str = "replay-session", button: Seat = Seat.A
) -> FrozenSessionRoster:
    roles = {
        Seat.A: TableRole.BUTTON,
        Seat.B: TableRole.SMALL_BLIND,
        Seat.C: TableRole.BIG_BLIND,
        Seat.D: TableRole.UNDER_THE_GUN,
    }
    # Role assignment above is correct only for Button A; use the domain helper
    # indirectly through the registration order for alternate buttons.
    if button is not Seat.A:
        from poker_dealer.domain import role_for_seat

        roles = {seat: role_for_seat(button, seat) for seat in Seat}
    return FrozenSessionRoster(
        session_id,
        button,
        1,
        tuple(
            RegisteredParticipant(
                participant_id=f"player-{seat.value}",
                seat=seat,
                initial_role=roles[seat],
                face_sample_count=5,
                voice_enrolled=True,
            )
            for seat in Seat
        ),
    )


class ScriptedReplaySources:
    """Deterministic fixture that emits valid observations for every context."""

    def __init__(
        self,
        cards: Mapping[VisionSlot, CardIdentity] = DEFAULT_SHOWDOWN_CARDS,
        *,
        actor_binding: bool = True,
    ) -> None:
        self.cards = dict(cards)
        self.actor_binding = actor_binding
        self.sequence = 0

    def reset_visual_settle(self, context: RuntimeObservationContext) -> None:
        del context

    def visual_is_settled(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> bool:
        del frame, context, observed_at_ns
        return True

    def observe_identity(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> FaceIdentityObservation | None:
        del frame
        if context.focus_seat is None:
            return None
        self.sequence += 1
        seat = context.focus_seat
        return FaceIdentityObservation(
            observation_id=f"replay-identity:{self.sequence}",
            session_id=context.session_id,
            expected_state_version=context.state_version,
            observed_at_ns=observed_at_ns,
            focus_seat=seat,
            identity_state=FaceIdentityState.MATCHED,
            player_id=f"player-{seat.value}",
            registered_seat=seat,
            similarity=0.99,
            second_best_similarity=0.1,
            stable_frames=5,
            stable_duration_ms=300,
            model_version="recorded-fixture-face@1",
            policy_version="recorded-fixture@1",
        )

    def observe_action(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> ActionEvidence | None:
        del frame
        if context.focus_seat is None:
            return None
        self.sequence += 1
        action = (
            PlayerActionType.CHECK
            if PlayerActionType.CHECK in context.legal_actions
            else PlayerActionType.CALL
        )
        observation = PlayerActionObservation(
            observation_id=f"replay-action:{self.sequence}",
            hand_id=context.hand_id,
            expected_state_version=context.state_version,
            window_started_at_ns=max(0, observed_at_ns - 300_000_000),
            observed_at_ns=observed_at_ns,
            focus_seat=context.focus_seat,
            evidence_state=ActionEvidenceState.CANDIDATE,
            candidate_action=action,
            confidence=0.99,
            stable_duration_ms=300,
            stable_frames=5,
            model_version="recorded-fixture-multimodal@1",
            calibration_version="recorded-fixture@1",
            quality_flags=("fusion_sources:gesture,speech",),
        )
        if not self.actor_binding:
            return ActionEvidence(observation)
        binding = ActorBinding(
            binding_id=f"replay-binding:{self.sequence}",
            session_id=context.session_id,
            hand_id=context.hand_id,
            expected_state_version=context.state_version,
            focus_seat=context.focus_seat,
            player_id=f"player-{context.focus_seat.value}",
            person_track_id=f"replay-track:{context.focus_seat.value}",
            verified_at_ns=max(0, observed_at_ns - 1),
            valid_until_ns=observed_at_ns + 1_000_000_000,
            identity_confidence=0.99,
            camera_epoch=context.camera_epoch,
        )
        return ActionEvidence(
            observation,
            binding,
            "recorded_face_pose_wrist",
            0.99,
        )

    def observe_card(
        self,
        frame,
        context: RuntimeObservationContext,
        slot: VisionSlot,
        observed_at_ns: int,
    ) -> CardObservation:
        del frame
        self.sequence += 1
        is_hole_deal = context.hand_phase.value == "dealing_hole"
        return CardObservation(
            observation_id=f"replay-card:{self.sequence}:{slot.value}",
            slot_id=slot,
            observed_at_ns=observed_at_ns,
            status=(
                ObservationStatus.FACE_DOWN
                if is_hole_deal
                else ObservationStatus.CONFIRMED
            ),
            card=None if is_hole_deal else self.cards[slot],
            confidence=None if is_hole_deal else 0.999,
            model_version="recorded-fixture-card@1",
            calibration_version="recorded-fixture@1",
            stable_frames=5,
        )


class RecordedReplaySources:
    """Replay exact serialized observations from a prior runtime log."""

    def __init__(self, log: RuntimeEventLog) -> None:
        self._identity = deque(log.evidence("face_identity_observation"))
        self._actions = deque(log.evidence("player_action_observation"))
        self._cards = deque(log.evidence("card_observation"))

    def reset_visual_settle(self, context: RuntimeObservationContext) -> None:
        del context

    def visual_is_settled(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> bool:
        del frame, context, observed_at_ns
        return True

    def observe_identity(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> FaceIdentityObservation | None:
        del frame, observed_at_ns
        record = self._pop_for_context(self._identity, context)
        return None if record is None else _identity_from_payload(record.payload)

    def observe_action(
        self, frame, context: RuntimeObservationContext, observed_at_ns: int
    ) -> ActionEvidence | None:
        del frame, observed_at_ns
        record = self._pop_for_context(self._actions, context)
        return None if record is None else _action_from_payload(record.payload)

    def observe_card(
        self,
        frame,
        context: RuntimeObservationContext,
        slot: VisionSlot,
        observed_at_ns: int,
    ) -> CardObservation | None:
        del frame, observed_at_ns
        if not self._cards:
            return None
        candidate = _card_from_payload(self._cards[0].payload)
        if candidate.slot_id is not slot:
            return None
        self._cards.popleft()
        return candidate

    @staticmethod
    def _pop_for_context(
        queue: deque[RuntimeLogRecord], context: RuntimeObservationContext
    ) -> RuntimeLogRecord | None:
        if not queue:
            return None
        payload = queue[0].payload
        if int(payload.get("expected_state_version", -1)) != context.state_version:
            return None
        if payload.get("focus_seat") != (
            context.focus_seat.value if context.focus_seat else None
        ):
            return None
        return queue.popleft()


def _identity_from_payload(value: Mapping[str, Any]) -> FaceIdentityObservation:
    registered = value.get("registered_seat")
    return FaceIdentityObservation(
        observation_id=str(value["observation_id"]),
        session_id=str(value["session_id"]),
        expected_state_version=int(value["expected_state_version"]),
        observed_at_ns=int(value["observed_at_ns"]),
        focus_seat=Seat(str(value["focus_seat"])),
        identity_state=FaceIdentityState(str(value["identity_state"])),
        player_id=str(value["player_id"]) if value.get("player_id") else None,
        registered_seat=Seat(str(registered)) if registered else None,
        similarity=float(value["similarity"]) if value.get("similarity") is not None else None,
        second_best_similarity=(
            float(value["second_best_similarity"])
            if value.get("second_best_similarity") is not None
            else None
        ),
        stable_frames=int(value["stable_frames"]),
        stable_duration_ms=int(value["stable_duration_ms"]),
        model_version=str(value["model_version"]),
        policy_version=str(value["policy_version"]),
        quality_flags=tuple(str(item) for item in value.get("quality_flags", ())),
    )


def _action_from_payload(value: Mapping[str, Any]) -> ActionEvidence:
    candidate = value.get("candidate_action")
    observation = PlayerActionObservation(
        observation_id=str(value["observation_id"]),
        hand_id=str(value["hand_id"]),
        expected_state_version=int(value["expected_state_version"]),
        window_started_at_ns=int(value["window_started_at_ns"]),
        observed_at_ns=int(value["observed_at_ns"]),
        focus_seat=Seat(str(value["focus_seat"])),
        evidence_state=ActionEvidenceState(str(value["evidence_state"])),
        candidate_action=PlayerActionType(str(candidate)) if candidate else None,
        confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
        stable_duration_ms=int(value["stable_duration_ms"]),
        stable_frames=int(value["stable_frames"]),
        model_version=str(value["model_version"]),
        calibration_version=str(value["calibration_version"]),
        quality_flags=tuple(str(item) for item in value.get("quality_flags", ())),
    )
    raw_binding = value.get("actor_binding")
    if not isinstance(raw_binding, Mapping):
        return ActionEvidence(observation)
    binding = ActorBinding(
        binding_id=str(raw_binding["binding_id"]),
        session_id=str(raw_binding["session_id"]),
        hand_id=str(raw_binding["hand_id"]),
        expected_state_version=int(raw_binding["expected_state_version"]),
        focus_seat=Seat(str(raw_binding["focus_seat"])),
        player_id=str(raw_binding["player_id"]),
        person_track_id=str(raw_binding["person_track_id"]),
        verified_at_ns=int(raw_binding["verified_at_ns"]),
        valid_until_ns=int(raw_binding["valid_until_ns"]),
        identity_confidence=float(raw_binding["identity_confidence"]),
        camera_epoch=int(raw_binding.get("camera_epoch", 0)),
    )
    confidence = value.get("attribution_confidence")
    return ActionEvidence(
        observation,
        binding,
        str(value.get("attribution_source") or "recorded"),
        float(confidence) if confidence is not None else 0.0,
        tuple(str(item) for item in value.get("attribution_quality_flags", ())),
    )


def _card_from_payload(value: Mapping[str, Any]) -> CardObservation:
    raw_card = value.get("card")
    card = None
    if isinstance(raw_card, Mapping):
        card = CardIdentity(Rank(str(raw_card["rank"])), Suit(str(raw_card["suit"])))
    confidence = value.get("confidence")
    return CardObservation(
        observation_id=str(value["observation_id"]),
        slot_id=VisionSlot(str(value["slot_id"])),
        observed_at_ns=int(value["observed_at_ns"]),
        status=ObservationStatus(str(value["status"])),
        card=card,
        confidence=float(confidence) if confidence is not None else None,
        model_version=str(value["model_version"]),
        calibration_version=str(value["calibration_version"]),
        stable_frames=int(value["stable_frames"]),
        quality_flags=tuple(str(item) for item in value.get("quality_flags", ())),
    )


__all__ = [
    "DEFAULT_SHOWDOWN_CARDS",
    "RecordedReplaySources",
    "ScriptedReplaySources",
    "StepClock",
    "default_replay_roster",
]
