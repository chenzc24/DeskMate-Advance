"""Evented four-player Hold'em state machine for the Stage 1 oracle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    CardObservation,
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    HandPhase,
    ObservationStatus,
    PlayerActionObservation,
    PlayerActionType,
    Rank,
    SEAT_ORDER,
    Seat,
    Street,
    Suit,
    VisionSlot,
    big_blind_seat,
    clockwise_order_after,
    first_to_act,
    next_button,
    small_blind_seat,
)

from .evaluator import HandRank, settle_showdown
from .event_log import EventLog, HandEvent
from .pots import OperatorAdjustment, Pot, build_pots
from .rules import FixedLimitRules


class SlotLifecycle(StrEnum):
    """Recoverable lifecycle values frozen by hand_snapshot schema 1.1."""

    EXPECTED_EMPTY = "expected_empty"
    DELIVERY_PENDING = "delivery_pending"
    PRESENT_FACE_DOWN = "present_face_down"
    REVEAL_PENDING = "reveal_pending"
    FACE_UP_UNCONFIRMED = "face_up_unconfirmed"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


BOARD_SLOTS: tuple[VisionSlot, ...] = (
    VisionSlot.BOARD_FLOP_1,
    VisionSlot.BOARD_FLOP_2,
    VisionSlot.BOARD_FLOP_3,
    VisionSlot.BOARD_TURN,
    VisionSlot.BOARD_RIVER,
)

HOLE_SLOTS: dict[Seat, tuple[VisionSlot, VisionSlot]] = {
    Seat.A: (VisionSlot.SEAT_A_HOLE_1, VisionSlot.SEAT_A_HOLE_2),
    Seat.B: (VisionSlot.SEAT_B_HOLE_1, VisionSlot.SEAT_B_HOLE_2),
    Seat.C: (VisionSlot.SEAT_C_HOLE_1, VisionSlot.SEAT_C_HOLE_2),
    Seat.D: (VisionSlot.SEAT_D_HOLE_1, VisionSlot.SEAT_D_HOLE_2),
}

STREET_BOARD_SLOTS: dict[Street, tuple[VisionSlot, ...]] = {
    Street.FLOP: BOARD_SLOTS[:3],
    Street.TURN: (VisionSlot.BOARD_TURN,),
    Street.RIVER: (VisionSlot.BOARD_RIVER,),
}


def _initial_slot_states() -> dict[VisionSlot, SlotLifecycle]:
    states = {slot: SlotLifecycle.EXPECTED_EMPTY for slot in VisionSlot}
    for slots in HOLE_SLOTS.values():
        for slot in slots:
            states[slot] = SlotLifecycle.PRESENT_FACE_DOWN
    return states


@dataclass(slots=True)
class PlayerState:
    stack_units: int
    street_commit_units: int = 0
    hand_commit_units: int = 0
    folded: bool = False
    all_in: bool = False

    def __post_init__(self) -> None:
        if min(
            self.stack_units, self.street_commit_units, self.hand_commit_units
        ) < 0:
            raise ValueError("player ledger values must be non-negative")


@dataclass(slots=True)
class HandState:
    hand_id: str
    state_version: int
    phase: HandPhase
    street: Street | None
    button: Seat
    small_blind_seat: Seat
    big_blind_seat: Seat
    acting_seat: Seat | None
    players: dict[Seat, PlayerState]
    legal_actions: tuple[PlayerActionType, ...] = ()
    current_bet_units: int = 0
    full_bets_this_street: int = 0
    acted_since_full_raise: set[Seat] = field(default_factory=set)
    raise_rights: set[Seat] = field(default_factory=set)
    pot_units: int = 0
    pots: tuple[Pot, ...] = ()
    board: tuple[CardIdentity, ...] = ()
    hole_cards: dict[Seat, tuple[CardIdentity, CardIdentity]] = field(
        default_factory=dict
    )
    slot_states: dict[VisionSlot, SlotLifecycle] = field(default_factory=dict)
    confirmed_cards: dict[VisionSlot, CardIdentity] = field(default_factory=dict)
    awards: dict[Seat, int] = field(default_factory=dict)
    paused_reason: str | None = None
    pending_command_id: str | None = None
    recovery_phase: HandPhase | None = None
    recovery_acting_seat: Seat | None = None
    recovery_pending_command_id: str | None = None
    rules_version: str = "1.3"

    def live_seats(self) -> tuple[Seat, ...]:
        return tuple(seat for seat in SEAT_ORDER if not self.players[seat].folded)

    def actionable_seats(self) -> tuple[Seat, ...]:
        return tuple(
            seat
            for seat in SEAT_ORDER
            if not self.players[seat].folded and not self.players[seat].all_in
        )

    def total_units(self) -> int:
        return sum(player.stack_units for player in self.players.values()) + self.pot_units


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_id: str
    hand_id: str
    expected_state_version: int
    seat: Seat
    action: PlayerActionType
    amount_units: int | None = None
    source: str = "simulator"


@dataclass(frozen=True, slots=True)
class ActionResult:
    accepted: bool
    reason: str
    state: HandState


@dataclass(frozen=True, slots=True)
class CardObservationResult:
    accepted: bool
    reason: str
    state: HandState


def _card_to_dict(card: CardIdentity) -> dict[str, str]:
    return {"rank": card.rank.value, "suit": card.suit.value}


def _card_from_dict(value: Mapping[str, str]) -> CardIdentity:
    return CardIdentity(Rank(value["rank"]), Suit(value["suit"]))


def state_to_dict(state: HandState) -> dict[str, Any]:
    return {
        "hand_id": state.hand_id,
        "state_version": state.state_version,
        "phase": state.phase.value,
        "street": state.street.value if state.street else None,
        "button": state.button.value,
        "small_blind_seat": state.small_blind_seat.value,
        "big_blind_seat": state.big_blind_seat.value,
        "acting_seat": state.acting_seat.value if state.acting_seat else None,
        "legal_actions": [action.value for action in state.legal_actions],
        "players": {
            seat.value: {
                "stack_units": player.stack_units,
                "street_commit_units": player.street_commit_units,
                "hand_commit_units": player.hand_commit_units,
                "folded": player.folded,
                "all_in": player.all_in,
            }
            for seat, player in state.players.items()
        },
        "current_bet_units": state.current_bet_units,
        "full_bets_this_street": state.full_bets_this_street,
        "acted_since_full_raise": sorted(
            seat.value for seat in state.acted_since_full_raise
        ),
        "raise_rights": sorted(seat.value for seat in state.raise_rights),
        "pot_units": state.pot_units,
        "pots": [
            {
                "pot_id": pot.pot_id,
                "amount_units": pot.amount_units,
                "eligible_seats": [seat.value for seat in pot.eligible_seats],
            }
            for pot in state.pots
        ],
        "board": [_card_to_dict(card) for card in state.board],
        "hole_cards": {
            seat.value: [_card_to_dict(card) for card in cards]
            for seat, cards in state.hole_cards.items()
        },
        "slot_states": {
            slot.value: lifecycle.value
            for slot, lifecycle in state.slot_states.items()
        },
        "confirmed_cards": {
            slot.value: _card_to_dict(card)
            for slot, card in state.confirmed_cards.items()
        },
        "awards": {seat.value: amount for seat, amount in state.awards.items()},
        "paused_reason": state.paused_reason,
        "pending_command_id": state.pending_command_id,
        "recovery_phase": (
            state.recovery_phase.value if state.recovery_phase else None
        ),
        "recovery_acting_seat": (
            state.recovery_acting_seat.value
            if state.recovery_acting_seat
            else None
        ),
        "recovery_pending_command_id": state.recovery_pending_command_id,
        "rules_version": state.rules_version,
    }


def state_from_dict(value: Mapping[str, Any]) -> HandState:
    players = {
        Seat(seat): PlayerState(**player)
        for seat, player in value["players"].items()
    }
    pots = tuple(
        Pot(
            pot["pot_id"],
            pot["amount_units"],
            tuple(Seat(seat) for seat in pot["eligible_seats"]),
        )
        for pot in value["pots"]
    )
    restored_slot_states = {
        VisionSlot(slot): SlotLifecycle(lifecycle)
        for slot, lifecycle in value.get("slot_states", {}).items()
    } or _initial_slot_states()
    return HandState(
        hand_id=value["hand_id"],
        state_version=value["state_version"],
        phase=HandPhase(value["phase"]),
        street=Street(value["street"]) if value["street"] else None,
        button=Seat(value["button"]),
        small_blind_seat=Seat(value["small_blind_seat"]),
        big_blind_seat=Seat(value["big_blind_seat"]),
        acting_seat=Seat(value["acting_seat"]) if value["acting_seat"] else None,
        players=players,
        legal_actions=tuple(PlayerActionType(item) for item in value["legal_actions"]),
        current_bet_units=value["current_bet_units"],
        full_bets_this_street=value["full_bets_this_street"],
        acted_since_full_raise={Seat(item) for item in value["acted_since_full_raise"]},
        raise_rights={Seat(item) for item in value["raise_rights"]},
        pot_units=value["pot_units"],
        pots=pots,
        board=tuple(_card_from_dict(card) for card in value["board"]),
        hole_cards={
            Seat(seat): tuple(_card_from_dict(card) for card in cards)  # type: ignore[arg-type]
            for seat, cards in value["hole_cards"].items()
        },
        slot_states=restored_slot_states,
        confirmed_cards={
            VisionSlot(slot): _card_from_dict(card)
            for slot, card in value.get("confirmed_cards", {}).items()
        },
        awards={Seat(seat): amount for seat, amount in value["awards"].items()},
        paused_reason=value["paused_reason"],
        pending_command_id=value["pending_command_id"],
        recovery_phase=(
            HandPhase(value["recovery_phase"])
            if value.get("recovery_phase")
            else None
        ),
        recovery_acting_seat=(
            Seat(value["recovery_acting_seat"])
            if value.get("recovery_acting_seat")
            else None
        ),
        recovery_pending_command_id=value.get("recovery_pending_command_id"),
        rules_version=value["rules_version"],
    )


def state_to_contract_snapshot(state: HandState) -> dict[str, Any]:
    """Return the exact recoverable snapshot shape frozen in schema 1.1."""

    internal = state_to_dict(state)
    if state.pots:
        pots = internal["pots"]
    elif state.pot_units:
        pots = [
            {
                "pot_id": "main",
                "amount_units": state.pot_units,
                "eligible_seats": [seat.value for seat in state.live_seats()],
            }
        ]
    else:
        pots = []
    return {
        "schema_version": "1.1",
        "rules_version": state.rules_version,
        "hand_id": state.hand_id,
        "state_version": state.state_version,
        "phase": state.phase.value,
        "street": state.street.value if state.street else None,
        "button": state.button.value,
        "small_blind_seat": state.small_blind_seat.value,
        "big_blind_seat": state.big_blind_seat.value,
        "acting_seat": state.acting_seat.value if state.acting_seat else None,
        "legal_actions": [action.value for action in state.legal_actions],
        "players": internal["players"],
        "pot_units": state.pot_units,
        "pots": pots,
        "slot_states": {
            slot.value: state.slot_states[slot].value for slot in VisionSlot
        },
        "confirmed_cards": [
            {"slot_id": slot.value, **_card_to_dict(state.confirmed_cards[slot])}
            for slot in VisionSlot
            if slot in state.confirmed_cards
        ],
        "pending_command_id": state.pending_command_id,
        "paused_reason": state.paused_reason,
    }


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    minimum_confidence: float = 0.90
    minimum_stable_frames: int = 3
    minimum_stable_duration_ms: int = 200


class ActionPromoter:
    def __init__(self, policy: PromotionPolicy | None = None) -> None:
        self.policy = policy or PromotionPolicy()
        self._seen_observations: set[str] = set()

    def promote(
        self, observation: PlayerActionObservation, state: HandState
    ) -> tuple[ActionRequest | None, str]:
        if observation.observation_id in self._seen_observations:
            return None, "duplicate_observation"
        self._seen_observations.add(observation.observation_id)
        if observation.hand_id != state.hand_id:
            return None, "wrong_hand"
        if observation.expected_state_version != state.state_version:
            return None, "stale_state_version"
        if observation.focus_seat is not state.acting_seat:
            return None, "non_current_seat"
        if observation.evidence_state is not ActionEvidenceState.CANDIDATE:
            return None, observation.evidence_state.value
        if observation.candidate_action is None or observation.confidence is None:
            return None, "incomplete_candidate"
        if observation.confidence < self.policy.minimum_confidence:
            return None, "low_confidence"
        if observation.stable_frames < self.policy.minimum_stable_frames:
            return None, "insufficient_stable_frames"
        if observation.stable_duration_ms < self.policy.minimum_stable_duration_ms:
            return None, "insufficient_stable_duration"
        return (
            ActionRequest(
                action_id=f"observation:{observation.observation_id}",
                hand_id=observation.hand_id,
                expected_state_version=observation.expected_state_version,
                seat=observation.focus_seat,
                action=observation.candidate_action,
                source=(
                    "voice_adapter"
                    if observation.model_version.startswith("player-action-vosk")
                    else (
                        "multimodal_adapter"
                        if observation.model_version.startswith("multimodal-action-fusion")
                        else "gesture_adapter"
                    )
                ),
            ),
            "candidate_promoted",
        )


class HandEngine:
    def __init__(
        self,
        rules: FixedLimitRules,
        state: HandState,
        log: EventLog | None = None,
        promoter: ActionPromoter | None = None,
    ) -> None:
        self.rules = rules
        self.state = state
        self.log = log or EventLog()
        self.promoter = promoter or ActionPromoter()
        self._seen_action_ids: set[str] = {
            str(event.payload["action_id"])
            for event in self.log.events
            if "action_id" in event.payload
        }
        self._seen_card_observation_ids: set[str] = {
            str(event.payload["observation_id"])
            for event in self.log.events
            if event.kind.startswith("card_observation")
            and "observation_id" in event.payload
        }
        self._seen_adjustment_ids: set[str] = {
            event.event_id
            for event in self.log.events
            if event.kind == "operator_adjustment"
        }
        completed_versions = [
            int(event.payload["device_state_version"])
            for event in self.log.events
            if event.kind == "dealer_command_completed"
            and "device_state_version" in event.payload
        ]
        self._last_dealer_device_state_version = max(completed_versions, default=-1)

    @classmethod
    def setup_session(
        cls,
        hand_id: str,
        button: Seat,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
        promotion_policy: PromotionPolicy | None = None,
    ) -> HandEngine:
        """Create an auditable pre-hand state without posting blinds."""

        if not hand_id.strip():
            raise ValueError("hand_id must not be empty")
        resolved_rules = rules or FixedLimitRules()
        starting = {
            seat: int((stacks or {}).get(seat, 80)) for seat in SEAT_ORDER
        }
        if any(stack < 0 for stack in starting.values()):
            raise ValueError("starting stacks must be non-negative")
        state = HandState(
            hand_id=hand_id,
            state_version=0,
            phase=HandPhase.SETUP,
            street=None,
            button=button,
            small_blind_seat=small_blind_seat(button),
            big_blind_seat=big_blind_seat(button),
            acting_seat=None,
            players={
                seat: PlayerState(stack) for seat, stack in starting.items()
            },
            slot_states={
                slot: SlotLifecycle.EXPECTED_EMPTY for slot in VisionSlot
            },
            rules_version=resolved_rules.rules_version,
        )
        engine = cls(
            resolved_rules,
            state,
            promoter=ActionPromoter(promotion_policy),
        )
        engine.log.append(
            kind="session_setup",
            event_id=f"{hand_id}:setup",
            before_version=-1,
            accepted=True,
            payload={"button": button.value},
            state=state,
        )
        return engine

    def begin_hand(self, event_id: str) -> HandState:
        """Post blinds and enter physical hole-card delivery.

        `start()` remains the pre-dealt Stage 1 oracle helper. This method is
        the production path and never assumes that private cards already exist.
        """

        if self.state.phase is not HandPhase.SETUP:
            raise ValueError("hand can only begin from setup")
        if not event_id.strip():
            raise ValueError("begin-hand event ID is required")
        if any(
            lifecycle is not SlotLifecycle.EXPECTED_EMPTY
            for lifecycle in self.state.slot_states.values()
        ):
            raise ValueError("all card slots must be empty before the hand begins")
        if any(
            player.stack_units < self.rules.big_blind_units
            for player in self.state.players.values()
        ):
            raise ValueError("every starting stack must cover the big blind")

        before = self.state.state_version
        self.state.phase = HandPhase.POSTING_BLINDS
        self.state.street = Street.PREFLOP
        self._contribute(self.state.small_blind_seat, self.rules.small_blind_units)
        self._contribute(self.state.big_blind_seat, self.rules.big_blind_units)
        self.state.current_bet_units = max(
            self.state.players[self.state.small_blind_seat].street_commit_units,
            self.state.players[self.state.big_blind_seat].street_commit_units,
        )
        self.state.full_bets_this_street = 1
        self.state.phase = HandPhase.DEALING_HOLE
        self.state.state_version += 1
        self.log.append(
            kind="hand_begun",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={
                "small_blind_seat": self.state.small_blind_seat.value,
                "big_blind_seat": self.state.big_blind_seat.value,
            },
            state=self.state,
        )
        return self.snapshot()

    def confirm_hole_dealt(self, event_id: str) -> HandState:
        """Open pre-flop action after all eight private slots are face down."""

        if self.state.phase is not HandPhase.DEALING_HOLE:
            raise ValueError("hole-card delivery is not pending")
        required_slots = tuple(
            slot for seat in SEAT_ORDER for slot in HOLE_SLOTS[seat]
        )
        if any(
            self.state.slot_states[slot] is not SlotLifecycle.PRESENT_FACE_DOWN
            for slot in required_slots
        ):
            raise ValueError("all hole-card slots must be present face down")

        before = self.state.state_version
        self.state.phase = HandPhase.AWAITING_ACTION
        actionables = self.state.actionable_seats()
        self.state.acting_seat = first_to_act(
            self.state.button, Street.PREFLOP, actionables
        )
        self.state.raise_rights = set(actionables)
        self.state.state_version += 1
        self._refresh_legal_actions()
        self.log.append(
            kind="hole_cards_confirmed",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={"hole_card_count": len(required_slots)},
            state=self.state,
        )
        return self.snapshot()

    @classmethod
    def start_predealt_fixture(
        cls,
        hand_id: str,
        button: Seat,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
    ) -> HandEngine:
        """Build the legacy pre-dealt oracle used only by deterministic tests."""
        if not hand_id.strip():
            raise ValueError("hand_id must not be empty")
        resolved_rules = rules or FixedLimitRules()
        starting = {
            seat: int((stacks or {}).get(seat, 80)) for seat in SEAT_ORDER
        }
        if any(stack < resolved_rules.big_blind_units for stack in starting.values()):
            raise ValueError("every starting stack must cover the big blind")
        players = {seat: PlayerState(stack) for seat, stack in starting.items()}
        sb = small_blind_seat(button)
        bb = big_blind_seat(button)
        state = HandState(
            hand_id=hand_id,
            state_version=0,
            phase=HandPhase.POSTING_BLINDS,
            street=Street.PREFLOP,
            button=button,
            small_blind_seat=sb,
            big_blind_seat=bb,
            acting_seat=None,
            players=players,
            slot_states=_initial_slot_states(),
            rules_version=resolved_rules.rules_version,
        )
        engine = cls(resolved_rules, state)
        engine._contribute(sb, resolved_rules.small_blind_units)
        engine._contribute(bb, resolved_rules.big_blind_units)
        state.current_bet_units = max(
            players[sb].street_commit_units, players[bb].street_commit_units
        )
        state.full_bets_this_street = 1
        state.phase = HandPhase.AWAITING_ACTION
        actionables = state.actionable_seats()
        state.acting_seat = first_to_act(button, Street.PREFLOP, actionables)
        state.raise_rights = set(actionables)
        engine._refresh_legal_actions()
        engine.log.append(
            kind="hand_started",
            event_id=f"{hand_id}:start",
            before_version=-1,
            accepted=True,
            payload={"button": button.value},
            state=state,
        )
        return engine

    @classmethod
    def start(
        cls,
        hand_id: str,
        button: Seat,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
    ) -> HandEngine:
        """Compatibility alias for tests; production must use setup/begin."""

        return cls.start_predealt_fixture(hand_id, button, stacks, rules)

    @classmethod
    def from_log(
        cls, rules: FixedLimitRules, log: EventLog
    ) -> HandEngine:
        return cls(rules, log.recover_state(), log=log)

    def snapshot(self) -> HandState:
        return copy.deepcopy(self.state)

    def record_dealer_command(self, command: DealerCommand) -> None:
        """Append a semantic command intent without pretending it completed."""

        if (
            self.state.pending_command_id is not None
            and self.state.pending_command_id != command.command_id
        ):
            raise ValueError("another dealer command is already pending")
        self.state.pending_command_id = command.command_id
        self.log.append(
            kind="dealer_command_issued",
            event_id=command.command_id,
            before_version=self.state.state_version,
            accepted=True,
            payload={
                "command": command.command.value,
                "target_slot": (
                    command.target_slot.value if command.target_slot else None
                ),
                "timeout_ms": command.timeout_ms,
            },
            state=self.state,
            observed_at_ns=command.issued_at_ns,
        )

    def record_dealer_ack(self, ack: DealerAck) -> None:
        """Append raw device evidence without accepting or clearing the command."""
        self.log.append(
            kind="dealer_ack_received",
            event_id=f"ack:{ack.command_id}:{ack.device_state_version}",
            before_version=self.state.state_version,
            accepted=False,
            payload={
                "command_id": ack.command_id,
                "command": ack.command.value,
                "target_slot": ack.target_slot.value if ack.target_slot else None,
                "status": ack.status.value,
                "device_state": ack.device_state.value,
                "device_state_version": ack.device_state_version,
                "device_reported_success": ack.status is DealerAckStatus.SUCCEEDED,
                "sensor_evidence": self._dealer_sensor_payload(ack),
                "error_code": ack.error_code.value if ack.error_code else None,
                "reason": ack.reason,
            },
            state=self.state,
            observed_at_ns=ack.observed_at_ns,
        )

    def record_dealer_completion(self, ack: DealerAck) -> None:
        """Commit a coordinator-correlated ACK with monotonic device evidence."""

        if ack.status is not DealerAckStatus.SUCCEEDED:
            raise ValueError("only a successful ACK can complete a command")
        if self.state.pending_command_id != ack.command_id:
            raise ValueError("ACK does not match the pending dealer command")
        if ack.device_state_version <= self._last_dealer_device_state_version:
            raise ValueError("dealer device state version is stale or non-monotonic")
        self.state.pending_command_id = None
        self._last_dealer_device_state_version = ack.device_state_version
        self.log.append(
            kind="dealer_command_completed",
            event_id=f"complete:{ack.command_id}:{ack.device_state_version}",
            before_version=self.state.state_version,
            accepted=True,
            payload={
                "command_id": ack.command_id,
                "command": ack.command.value,
                "target_slot": ack.target_slot.value if ack.target_slot else None,
                "device_state": ack.device_state.value,
                "device_state_version": ack.device_state_version,
                "sensor_evidence": self._dealer_sensor_payload(ack),
            },
            state=self.state,
            observed_at_ns=ack.observed_at_ns,
        )

    @staticmethod
    def _dealer_sensor_payload(ack: DealerAck) -> dict[str, bool | int | None]:
        evidence = ack.sensor_evidence
        return {
            "homed": evidence.homed,
            "at_target": evidence.at_target,
            "deck_present": evidence.deck_present,
            "exit_pulses": evidence.exit_pulses,
            "interlock_closed": evidence.interlock_closed,
            "emergency_stop": evidence.emergency_stop,
        }

    def mark_delivery_pending(
        self,
        event_id: str,
        slot: VisionSlot,
        observed_at_ns: int,
        *,
        face_down_by_default: bool = False,
    ) -> HandState:
        """Persist one sensor-valid dispense.

        Hole cards are dealt face down by the Core v1 mechanism contract, so a
        successful dispense acknowledgement can complete that logical slot
        without a second operator or vision confirmation. Face-up board cards
        remain delivery-pending until card perception confirms their identity.
        """

        if self.state.phase not in {HandPhase.DEALING_HOLE, HandPhase.DEALING_BOARD}:
            raise ValueError("card delivery is not active")
        lifecycle = self.state.slot_states[slot]
        if lifecycle is SlotLifecycle.DELIVERY_PENDING:
            return self.snapshot()
        if lifecycle is not SlotLifecycle.EXPECTED_EMPTY:
            raise ValueError("delivery target is not empty")
        before = self.state.state_version
        if face_down_by_default:
            if self.state.phase is not HandPhase.DEALING_HOLE:
                raise ValueError("face-down dispense default is only valid for hole cards")
            lifecycle = SlotLifecycle.PRESENT_FACE_DOWN
        else:
            lifecycle = SlotLifecycle.DELIVERY_PENDING
        self.state.slot_states[slot] = lifecycle
        self.state.state_version += 1
        self.log.append(
            kind="card_delivery_acknowledged",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={
                "slot_id": slot.value,
                "slot_lifecycle": lifecycle.value,
                "face_down_by_default": face_down_by_default,
            },
            state=self.state,
            observed_at_ns=observed_at_ns,
        )
        return self.snapshot()

    def _contribute(self, seat: Seat, requested: int) -> int:
        if requested < 0:
            raise ValueError("requested contribution cannot be negative")
        player = self.state.players[seat]
        paid = min(player.stack_units, requested)
        player.stack_units -= paid
        player.street_commit_units += paid
        player.hand_commit_units += paid
        player.all_in = player.stack_units == 0
        self.state.pot_units = sum(
            item.hand_commit_units for item in self.state.players.values()
        )
        return paid

    def _refresh_legal_actions(self) -> None:
        state = self.state
        seat = state.acting_seat
        if state.phase is not HandPhase.AWAITING_ACTION or seat is None:
            state.legal_actions = ()
            return
        player = state.players[seat]
        to_call = max(0, state.current_bet_units - player.street_commit_units)
        actions: list[PlayerActionType] = [PlayerActionType.FOLD]
        if to_call == 0:
            actions.append(PlayerActionType.CHECK)
            if player.stack_units > 0 and seat in state.raise_rights:
                if state.current_bet_units == 0:
                    actions.append(PlayerActionType.BET)
                elif state.full_bets_this_street < self.rules.max_full_bets_per_street:
                    actions.append(PlayerActionType.RAISE)
        else:
            actions.append(PlayerActionType.CALL)
            if (
                player.stack_units > to_call
                and seat in state.raise_rights
                and state.full_bets_this_street
                < self.rules.max_full_bets_per_street
            ):
                actions.append(PlayerActionType.RAISE)
        state.legal_actions = tuple(actions)

    def _record_rejection(self, event_id: str, reason: str, payload: Mapping[str, Any]) -> ActionResult:
        self.log.append(
            kind="action_rejected",
            event_id=event_id,
            before_version=self.state.state_version,
            accepted=False,
            payload={**payload, "reason": reason},
            state=self.state,
        )
        return ActionResult(False, reason, self.snapshot())

    def apply_observation(self, observation: PlayerActionObservation) -> ActionResult:
        request, reason = self.promoter.promote(observation, self.state)
        if request is None:
            return self._record_rejection(
                observation.observation_id,
                reason,
                {"observation_id": observation.observation_id},
            )
        return self.apply_action(request)

    def apply_action(self, request: ActionRequest) -> ActionResult:
        payload = {
            "action_id": request.action_id,
            "seat": request.seat.value,
            "action": request.action.value,
            "source": request.source,
        }
        if request.action_id in self._seen_action_ids:
            return ActionResult(False, "duplicate_action_id", self.snapshot())
        self._seen_action_ids.add(request.action_id)
        if request.hand_id != self.state.hand_id:
            return self._record_rejection(request.action_id, "wrong_hand", payload)
        if request.expected_state_version != self.state.state_version:
            return self._record_rejection(
                request.action_id, "stale_state_version", payload
            )
        if self.state.phase is not HandPhase.AWAITING_ACTION:
            return self._record_rejection(request.action_id, "not_awaiting_action", payload)
        if request.seat is not self.state.acting_seat:
            return self._record_rejection(request.action_id, "non_current_seat", payload)
        if request.amount_units is not None:
            return self._record_rejection(
                request.action_id, "fixed_limit_amount_must_be_null", payload
            )
        if request.action not in self.state.legal_actions:
            return self._record_rejection(request.action_id, "illegal_action", payload)

        before = self.state.state_version
        seat = request.seat
        player = self.state.players[seat]
        to_call = max(0, self.state.current_bet_units - player.street_commit_units)
        full_raise = False

        if request.action is PlayerActionType.FOLD:
            player.folded = True
        elif request.action is PlayerActionType.CHECK:
            pass
        elif request.action is PlayerActionType.CALL:
            self._contribute(seat, to_call)
        elif request.action in {PlayerActionType.BET, PlayerActionType.RAISE}:
            bet_size = self.rules.bet_size(self.state.street)  # type: ignore[arg-type]
            target = (
                bet_size
                if request.action is PlayerActionType.BET
                else self.state.current_bet_units + bet_size
            )
            required = target - player.street_commit_units
            paid = self._contribute(seat, required)
            new_commit = player.street_commit_units
            full_raise = paid == required
            self.state.current_bet_units = max(
                self.state.current_bet_units, new_commit
            )
            if full_raise:
                self.state.full_bets_this_street += 1

        self.state.raise_rights.discard(seat)
        if full_raise:
            actionables = set(self.state.actionable_seats())
            self.state.acted_since_full_raise = {seat}
            self.state.raise_rights = actionables - {seat}
        else:
            self.state.acted_since_full_raise.add(seat)

        if len(self.state.live_seats()) == 1:
            self._settle_uncontested(self.state.live_seats()[0])
        elif self._betting_round_complete():
            self._advance_after_betting_round()
        else:
            actionables = self.state.actionable_seats()
            self.state.acting_seat = clockwise_order_after(seat, actionables)[0]

        self.state.state_version += 1
        self._refresh_legal_actions()
        self.log.append(
            kind="action_applied",
            event_id=request.action_id,
            before_version=before,
            accepted=True,
            payload=payload,
            state=self.state,
        )
        return ActionResult(True, "accepted", self.snapshot())

    def _betting_round_complete(self) -> bool:
        actionables = set(self.state.actionable_seats())
        if not actionables:
            return True
        if len(actionables) == 1:
            only = next(iter(actionables))
            if self.state.players[only].street_commit_units >= self.state.current_bet_units:
                return True
        return actionables <= self.state.acted_since_full_raise and all(
            self.state.players[seat].street_commit_units
            == self.state.current_bet_units
            for seat in actionables
        )

    def _advance_after_betting_round(self) -> None:
        for player in self.state.players.values():
            player.street_commit_units = 0
        self.state.current_bet_units = 0
        self.state.full_bets_this_street = 0
        self.state.acted_since_full_raise.clear()
        self.state.raise_rights.clear()
        self.state.acting_seat = None
        self.state.legal_actions = ()
        if self.state.street is Street.RIVER:
            self.state.street = Street.SHOWDOWN
            self.state.phase = HandPhase.SHOWDOWN
            return
        next_street = {
            Street.PREFLOP: Street.FLOP,
            Street.FLOP: Street.TURN,
            Street.TURN: Street.RIVER,
        }[self.state.street]  # type: ignore[index]
        self.state.street = next_street
        self.state.phase = HandPhase.DEALING_BOARD

    def apply_card_observation(
        self, observation: CardObservation
    ) -> CardObservationResult:
        """Apply phase-gated visual evidence without allowing unknown to advance."""

        if observation.observation_id in self._seen_card_observation_ids:
            return CardObservationResult(
                False, "duplicate_observation", self.snapshot()
            )
        self._seen_card_observation_ids.add(observation.observation_id)
        before = self.state.state_version
        payload = {
            "observation_id": observation.observation_id,
            "slot_id": observation.slot_id.value,
            "status": observation.status.value,
        }

        active_slots: set[VisionSlot]
        if self.state.phase is HandPhase.DEALING_HOLE:
            active_slots = {
                slot for seat in SEAT_ORDER for slot in HOLE_SLOTS[seat]
            }
        elif self.state.phase is HandPhase.DEALING_BOARD:
            active_slots = set(STREET_BOARD_SLOTS[self.state.street])  # type: ignore[index]
        elif self.state.phase is HandPhase.SHOWDOWN:
            active_slots = set(BOARD_SLOTS)
            for seat in self.state.live_seats():
                active_slots.update(HOLE_SLOTS[seat])
        else:
            active_slots = set()
        if observation.slot_id not in active_slots:
            reason = "inactive_slot"
            self.log.append(
                kind="card_observation_rejected",
                event_id=observation.observation_id,
                before_version=before,
                accepted=False,
                payload={**payload, "reason": reason},
                state=self.state,
                observed_at_ns=observation.observed_at_ns,
            )
            return CardObservationResult(False, reason, self.snapshot())

        if observation.status in {
            ObservationStatus.UNKNOWN,
            ObservationStatus.OCCLUDED,
        }:
            reason = observation.status.value
            self.log.append(
                kind="card_observation_rejected",
                event_id=observation.observation_id,
                before_version=before,
                accepted=False,
                payload={**payload, "reason": reason},
                state=self.state,
                observed_at_ns=observation.observed_at_ns,
            )
            return CardObservationResult(False, reason, self.snapshot())

        lifecycle = {
            ObservationStatus.EMPTY: SlotLifecycle.EXPECTED_EMPTY,
            ObservationStatus.FACE_DOWN: SlotLifecycle.PRESENT_FACE_DOWN,
            ObservationStatus.FACE_UP_UNCONFIRMED: SlotLifecycle.FACE_UP_UNCONFIRMED,
        }.get(observation.status)
        reason = "observation_recorded"

        existing_card = self.state.confirmed_cards.get(observation.slot_id)
        existing_lifecycle = self.state.slot_states[observation.slot_id]
        if (
            existing_card is not None
            and observation.status is not ObservationStatus.CONFIRMED
        ):
            reason = "confirmed_slot_cannot_downgrade"
            self.log.append(
                kind="card_observation_rejected",
                event_id=observation.observation_id,
                before_version=before,
                accepted=False,
                payload={**payload, "reason": reason},
                state=self.state,
                observed_at_ns=observation.observed_at_ns,
            )
            return CardObservationResult(False, reason, self.snapshot())

        if observation.status is ObservationStatus.CONFIRMED:
            card = observation.card
            assert card is not None  # enforced by CardObservation
            if existing_card == card and existing_lifecycle is SlotLifecycle.CONFIRMED:
                self.log.append(
                    kind="card_observation_applied",
                    event_id=observation.observation_id,
                    before_version=before,
                    accepted=True,
                    payload={**payload, "reason": "card_already_confirmed"},
                    state=self.state,
                    observed_at_ns=observation.observed_at_ns,
                )
                return CardObservationResult(
                    True, "card_already_confirmed", self.snapshot()
                )
            if existing_card is not None and existing_card != card:
                self._capture_recovery_checkpoint()
                self.state.slot_states[observation.slot_id] = SlotLifecycle.CONFLICT
                self.state.phase = HandPhase.PAUSED_RECOVERY
                self.state.paused_reason = "slot_card_identity_changed"
                self.state.acting_seat = None
                self.state.legal_actions = ()
                self.state.state_version += 1
                reason = "slot_card_identity_changed"
                self.log.append(
                    kind="card_observation_conflict",
                    event_id=observation.observation_id,
                    before_version=before,
                    accepted=False,
                    payload={**payload, "reason": reason},
                    state=self.state,
                    observed_at_ns=observation.observed_at_ns,
                )
                return CardObservationResult(False, reason, self.snapshot())
            duplicate_slots = tuple(
                slot
                for slot, confirmed in self.state.confirmed_cards.items()
                if confirmed == card and slot is not observation.slot_id
            )
            if duplicate_slots:
                self._capture_recovery_checkpoint()
                self.state.slot_states[observation.slot_id] = SlotLifecycle.CONFLICT
                for slot in duplicate_slots:
                    self.state.slot_states[slot] = SlotLifecycle.CONFLICT
                self.state.phase = HandPhase.PAUSED_RECOVERY
                self.state.paused_reason = "duplicate_card_identity"
                self.state.acting_seat = None
                self.state.legal_actions = ()
                self.state.state_version += 1
                reason = "duplicate_card_identity"
                self.log.append(
                    kind="card_observation_conflict",
                    event_id=observation.observation_id,
                    before_version=before,
                    accepted=False,
                    payload={**payload, "reason": reason},
                    state=self.state,
                    observed_at_ns=observation.observed_at_ns,
                )
                return CardObservationResult(False, reason, self.snapshot())
            self.state.confirmed_cards[observation.slot_id] = card
            lifecycle = SlotLifecycle.CONFIRMED
            reason = "card_confirmed"

        assert lifecycle is not None
        changed = self.state.slot_states[observation.slot_id] is not lifecycle
        self.state.slot_states[observation.slot_id] = lifecycle
        if changed or observation.status is ObservationStatus.CONFIRMED:
            self.state.state_version += 1
        self.log.append(
            kind="card_observation_applied",
            event_id=observation.observation_id,
            before_version=before,
            accepted=True,
            payload=payload,
            state=self.state,
            observed_at_ns=observation.observed_at_ns,
        )
        return CardObservationResult(True, reason, self.snapshot())

    def confirm_board_dealt(self, event_id: str) -> HandState:
        if self.state.phase is not HandPhase.DEALING_BOARD:
            raise ValueError("no board street is pending")
        required_slots = STREET_BOARD_SLOTS[self.state.street]  # type: ignore[index]
        if any(
            self.state.slot_states[slot] is not SlotLifecycle.CONFIRMED
            for slot in required_slots
        ):
            raise ValueError("required board slots are not confirmed")
        before = self.state.state_version
        self.state.board = tuple(
            self.state.confirmed_cards[slot]
            for slot in BOARD_SLOTS
            if slot in self.state.confirmed_cards
        )
        self.state.phase = HandPhase.AWAITING_ACTION
        actionables = self.state.actionable_seats()
        if len(actionables) <= 1 and len(self.state.live_seats()) > 1:
            self._advance_after_betting_round()
        else:
            self.state.acting_seat = first_to_act(
                self.state.button,
                self.state.street,  # type: ignore[arg-type]
                actionables,
            )
            self.state.raise_rights = set(actionables)
        self.state.state_version += 1
        self._refresh_legal_actions()
        self.log.append(
            kind="board_confirmed",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={"street": self.state.street.value},  # type: ignore[union-attr]
            state=self.state,
        )
        return self.snapshot()

    def _settle_uncontested(self, winner: Seat) -> None:
        amount = self.state.pot_units
        self.state.players[winner].stack_units += amount
        self.state.awards = {winner: amount}
        for player in self.state.players.values():
            player.street_commit_units = 0
            player.hand_commit_units = 0
        self.state.pot_units = 0
        self.state.pots = ()
        self.state.phase = HandPhase.SETTLED
        self.state.acting_seat = None
        self.state.legal_actions = ()

    def settle_showdown(
        self,
        event_id: str,
        board: tuple[CardIdentity, ...],
        hole_cards: Mapping[Seat, tuple[CardIdentity, CardIdentity]],
    ) -> Mapping[Seat, HandRank]:
        if self.state.phase is not HandPhase.SHOWDOWN:
            raise ValueError("hand is not at showdown")
        live = set(self.state.live_seats())
        if set(hole_cards) != live:
            raise ValueError("showdown requires hole cards for every live seat")
        if len(board) != 5:
            raise ValueError("showdown requires five board cards")
        if self.state.board and tuple(board) != self.state.board:
            raise ValueError("showdown board differs from confirmed board slots")
        visible_cards = tuple(board) + tuple(
            card
            for seat in SEAT_ORDER
            for card in hole_cards.get(seat, ())
        )
        if len(visible_cards) != len(set(visible_cards)):
            raise ValueError("duplicate card identity at showdown")
        before = self.state.state_version
        built = build_pots(
            {seat: player.hand_commit_units for seat, player in self.state.players.items()},
            {seat for seat, player in self.state.players.items() if player.folded},
        )
        for seat, amount in built.returned_units.items():
            self.state.players[seat].stack_units += amount
        result = settle_showdown(built.pots, board, hole_cards, self.state.button)
        for seat, amount in result.awards.items():
            self.state.players[seat].stack_units += amount
        self.state.board = tuple(board)
        self.state.hole_cards = dict(hole_cards)
        for slot, card in zip(BOARD_SLOTS, board, strict=True):
            self.state.slot_states[slot] = SlotLifecycle.CONFIRMED
            self.state.confirmed_cards[slot] = card
        for seat, cards in hole_cards.items():
            for slot, card in zip(HOLE_SLOTS[seat], cards, strict=True):
                self.state.slot_states[slot] = SlotLifecycle.CONFIRMED
                self.state.confirmed_cards[slot] = card
        self.state.pots = built.pots
        self.state.awards = dict(result.awards)
        for player in self.state.players.values():
            player.street_commit_units = 0
            player.hand_commit_units = 0
        self.state.pot_units = 0
        self.state.phase = HandPhase.SETTLED
        self.state.acting_seat = None
        self.state.legal_actions = ()
        self.state.state_version += 1
        self.log.append(
            kind="showdown_settled",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={
                "winners_by_pot": {
                    pot: [seat.value for seat in winners]
                    for pot, winners in result.winners_by_pot.items()
                }
            },
            state=self.state,
        )
        return result.ranks

    def settle_confirmed_showdown(self, event_id: str) -> Mapping[Seat, HandRank]:
        """Settle a live hand only from identities confirmed in owned slots."""

        if self.state.phase is not HandPhase.SHOWDOWN:
            raise ValueError("hand is not at showdown")
        if any(
            self.state.slot_states[slot] is not SlotLifecycle.CONFIRMED
            for slot in BOARD_SLOTS
        ):
            raise ValueError("showdown requires five confirmed board slots")
        live = self.state.live_seats()
        if any(
            self.state.slot_states[slot] is not SlotLifecycle.CONFIRMED
            for seat in live
            for slot in HOLE_SLOTS[seat]
        ):
            raise ValueError("showdown requires two confirmed hole cards per live seat")
        board = tuple(self.state.confirmed_cards[slot] for slot in BOARD_SLOTS)
        hole_cards = {
            seat: tuple(
                self.state.confirmed_cards[slot] for slot in HOLE_SLOTS[seat]
            )
            for seat in live
        }
        return self.settle_showdown(event_id, board, hole_cards)  # type: ignore[arg-type]

    def apply_operator_adjustment(
        self, adjustment_id: str, adjustment: OperatorAdjustment
    ) -> HandState:
        """Apply a pre-hand balance correction as one append-only event."""

        if not adjustment_id.strip():
            raise ValueError("adjustment_id is required")
        if adjustment_id in self._seen_adjustment_ids:
            return self.snapshot()
        if self.state.phase is not HandPhase.SETUP:
            raise ValueError("operator adjustments are allowed only in setup")
        updated = (
            self.state.players[adjustment.seat].stack_units
            + adjustment.amount_units
        )
        if updated < 0:
            raise ValueError("operator adjustment cannot create a negative balance")
        before = self.state.state_version
        self.state.players[adjustment.seat].stack_units = updated
        self.state.state_version += 1
        self._seen_adjustment_ids.add(adjustment_id)
        self.log.append(
            kind="operator_adjustment",
            event_id=adjustment_id,
            before_version=before,
            accepted=True,
            payload={
                "seat": adjustment.seat.value,
                "amount_units": adjustment.amount_units,
                "operator_id": adjustment.operator_id,
                "reason": adjustment.reason,
            },
            state=self.state,
        )
        return self.snapshot()

    def pause(self, event_id: str, reason: str) -> HandState:
        if not reason.strip():
            raise ValueError("pause reason is required")
        before = self.state.state_version
        self._capture_recovery_checkpoint()
        self.state.phase = HandPhase.PAUSED_RECOVERY
        self.state.paused_reason = reason
        self.state.pending_command_id = None
        self.state.acting_seat = None
        self.state.legal_actions = ()
        self.state.state_version += 1
        self.log.append(
            kind="hand_paused",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={"reason": reason},
            state=self.state,
        )
        return self.snapshot()

    def resume_from_recovery(
        self,
        event_id: str,
        *,
        operator_id: str,
        reason: str,
        physical_state_confirmed: bool,
    ) -> HandState:
        """Resume only after an audited human confirms physical/software parity."""

        if self.state.phase is not HandPhase.PAUSED_RECOVERY:
            raise ValueError("hand is not paused for recovery")
        if not event_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("recovery event, operator and reason are required")
        if not physical_state_confirmed:
            raise ValueError("physical/software state parity must be confirmed")
        recovery_phase = self.state.recovery_phase
        if recovery_phase is None or recovery_phase is HandPhase.PAUSED_RECOVERY:
            raise ValueError("no resumable recovery checkpoint is available")
        if self.state.pending_command_id is not None:
            raise ValueError("cannot resume while a dealer command remains pending")
        if any(
            lifecycle is SlotLifecycle.CONFLICT
            for lifecycle in self.state.slot_states.values()
        ):
            raise ValueError("card conflicts require void/reconciliation, not retry")
        before = self.state.state_version
        self.state.phase = recovery_phase
        self.state.acting_seat = self.state.recovery_acting_seat
        old_pending = self.state.recovery_pending_command_id
        self.state.paused_reason = None
        self.state.recovery_phase = None
        self.state.recovery_acting_seat = None
        self.state.recovery_pending_command_id = None
        self.state.state_version += 1
        self._refresh_legal_actions()
        self.log.append(
            kind="hand_recovery_resumed",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={
                "operator_id": operator_id,
                "reason": reason,
                "physical_state_confirmed": True,
                "abandoned_command_id": old_pending,
                "restored_phase": recovery_phase.value,
            },
            state=self.state,
        )
        return self.snapshot()

    def reconcile_card_slot(
        self,
        event_id: str,
        *,
        slot: VisionSlot,
        operator_id: str,
        reason: str,
        physical_slot_empty: bool,
    ) -> HandState:
        """Audit a manual card removal so the restored lane can re-observe it."""

        if self.state.phase is not HandPhase.PAUSED_RECOVERY:
            raise ValueError("card reconciliation requires a paused hand")
        if not event_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("reconciliation event, operator and reason are required")
        if not physical_slot_empty:
            raise ValueError("reconciled card slot must be physically confirmed empty")
        if self.state.slot_states[slot] is not SlotLifecycle.CONFLICT:
            raise ValueError("only a conflicting slot may be reconciled")
        before = self.state.state_version
        removed = self.state.confirmed_cards.pop(slot, None)
        self.state.slot_states[slot] = SlotLifecycle.EXPECTED_EMPTY
        self.state.state_version += 1
        self.log.append(
            kind="card_slot_reconciled",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={
                "slot_id": slot.value,
                "operator_id": operator_id,
                "reason": reason,
                "physical_slot_empty": True,
                "removed_card": _card_to_dict(removed) if removed else None,
            },
            state=self.state,
        )
        return self.snapshot()

    def _capture_recovery_checkpoint(self) -> None:
        if self.state.phase is HandPhase.PAUSED_RECOVERY:
            return
        self.state.recovery_phase = self.state.phase
        self.state.recovery_acting_seat = self.state.acting_seat
        self.state.recovery_pending_command_id = self.state.pending_command_id

    def void(self, event_id: str, reason: str) -> HandState:
        before = self.state.state_version
        for player in self.state.players.values():
            player.stack_units += player.hand_commit_units
            player.street_commit_units = 0
            player.hand_commit_units = 0
        self.state.pot_units = 0
        self.state.phase = HandPhase.VOIDED
        self.state.paused_reason = reason
        self.state.pending_command_id = None
        self.state.recovery_phase = None
        self.state.recovery_acting_seat = None
        self.state.recovery_pending_command_id = None
        self.state.acting_seat = None
        self.state.legal_actions = ()
        self.state.state_version += 1
        self.log.append(
            kind="hand_voided",
            event_id=event_id,
            before_version=before,
            accepted=True,
            payload={"reason": reason, "redeal_button": self.state.button.value},
            state=self.state,
        )
        return self.snapshot()

    def next_button(self) -> Seat:
        if self.state.phase is HandPhase.VOIDED:
            return self.state.button
        if self.state.phase is not HandPhase.SETTLED:
            raise ValueError("button advances only after a settled hand")
        return next_button(self.state.button)
