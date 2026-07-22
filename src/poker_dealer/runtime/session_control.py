"""Semantic operator decisions around hands; never a game-rule authority."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import ControlIntent, ControlObservation, Seat, VisionSlot
from poker_dealer.game import SlotLifecycle

from .session_runtime import SessionRuntime


class SessionOperatorSignal(StrEnum):
    CONTINUE_WAITING = "continue_waiting"
    START_NEXT_HAND = "start_next_hand"
    RETRY_HAND = "retry_hand"
    HAND_VOIDED = "hand_voided"
    SESSION_ENDED = "session_ended"


@dataclass(frozen=True, slots=True)
class SessionOperatorOutcome:
    accepted: bool
    reason: str
    signal: SessionOperatorSignal = SessionOperatorSignal.CONTINUE_WAITING
    selected_seat: Seat | None = None
    selected_slot: VisionSlot | None = None


class SessionOperatorController:
    """Interpret common controls only at explicit session/recovery boundaries."""

    def __init__(
        self,
        session: SessionRuntime,
        *,
        operator_id: str,
        rebuy_to_units: int | None = None,
    ) -> None:
        if not operator_id.strip():
            raise ValueError("operator ID is required")
        target = (
            session.game_config.starting_stack_units
            if rebuy_to_units is None
            else rebuy_to_units
        )
        if target < session.game_config.minimum_stack_to_start_hand_units:
            raise ValueError("rebuy target must meet the minimum starting stack")
        self.session = session
        self.operator_id = operator_id
        self.rebuy_to_units = target
        self._selected_seat_index = 0
        self._selected_slot_index = 0

    @property
    def selected_low_stack_seat(self) -> Seat | None:
        seats = self.session.low_stack_seats
        if not seats:
            return None
        self._selected_seat_index %= len(seats)
        return seats[self._selected_seat_index]

    @property
    def selected_conflict_slot(self) -> VisionSlot | None:
        runtime = self.session.active_hand
        if runtime is None:
            return None
        slots = tuple(
            slot
            for slot in VisionSlot
            if runtime.engine.state.slot_states[slot] is SlotLifecycle.CONFLICT
        )
        if not slots:
            return None
        self._selected_slot_index %= len(slots)
        return slots[self._selected_slot_index]

    def accept(self, control: ControlObservation) -> SessionOperatorOutcome:
        if self.session.ended:
            return SessionOperatorOutcome(False, "session_already_ended")
        if self.session.active_hand is not None:
            return self._accept_recovery(control)
        return self._accept_between_hands(control)

    def _accept_between_hands(
        self, control: ControlObservation
    ) -> SessionOperatorOutcome:
        if control.intent in {
            ControlIntent.NEXT_OPTION,
            ControlIntent.PREVIOUS_OPTION,
        }:
            seats = self.session.low_stack_seats
            if not seats:
                return SessionOperatorOutcome(False, "no_rebuy_selection_required")
            delta = 1 if control.intent is ControlIntent.NEXT_OPTION else -1
            self._selected_seat_index = (self._selected_seat_index + delta) % len(seats)
            return SessionOperatorOutcome(
                True,
                "rebuy_seat_selected",
                selected_seat=self.selected_low_stack_seat,
            )
        if control.intent is ControlIntent.CONFIRM:
            if not self.session.table_cleared:
                self.session.confirm_table_cleared(
                    operator_id=self.operator_id,
                    reason="operator_confirmed_all_cards_returned",
                )
                return SessionOperatorOutcome(True, "table_cleared")
            seat = self.selected_low_stack_seat
            if seat is None:
                return SessionOperatorOutcome(False, "nothing_to_confirm")
            amount = self.rebuy_to_units - self.session.stacks[seat]
            if amount <= 0:
                return SessionOperatorOutcome(False, "selected_stack_not_below_target")
            self.session.adjust_stack(
                adjustment_id=f"rebuy:{control.observation_id}",
                seat=seat,
                amount_units=amount,
                operator_id=self.operator_id,
                reason="operator_confirmed_rebuy",
            )
            self._selected_seat_index = 0
            return SessionOperatorOutcome(
                True, "rebuy_applied", selected_seat=seat
            )
        if control.intent is ControlIntent.START:
            if not self.session.table_cleared:
                return SessionOperatorOutcome(False, "table_clearance_required")
            if self.session.low_stack_seats:
                return SessionOperatorOutcome(False, "rebuy_or_end_required")
            return SessionOperatorOutcome(
                True,
                "start_next_hand",
                SessionOperatorSignal.START_NEXT_HAND,
            )
        if control.intent in {ControlIntent.CLEAR, ControlIntent.CANCEL}:
            if not self.session.table_cleared:
                return SessionOperatorOutcome(False, "clear_table_before_session_end")
            self.session.end_session(
                operator_id=self.operator_id,
                reason="operator_requested_session_end",
            )
            return SessionOperatorOutcome(
                True,
                "session_ended",
                SessionOperatorSignal.SESSION_ENDED,
            )
        return SessionOperatorOutcome(False, "unsupported_between_hand_control")

    def _accept_recovery(
        self, control: ControlObservation
    ) -> SessionOperatorOutcome:
        runtime = self.session.active_hand
        assert runtime is not None
        if runtime.phase.value != "paused_recovery":
            return SessionOperatorOutcome(False, "active_hand_not_paused")
        if control.intent in {
            ControlIntent.NEXT_OPTION,
            ControlIntent.PREVIOUS_OPTION,
        }:
            conflicts = tuple(
                slot
                for slot in VisionSlot
                if runtime.engine.state.slot_states[slot]
                is SlotLifecycle.CONFLICT
            )
            if not conflicts:
                return SessionOperatorOutcome(False, "no_conflict_slot_to_select")
            delta = 1 if control.intent is ControlIntent.NEXT_OPTION else -1
            self._selected_slot_index = (self._selected_slot_index + delta) % len(
                conflicts
            )
            return SessionOperatorOutcome(
                True,
                "conflict_slot_selected",
                selected_slot=self.selected_conflict_slot,
            )
        if control.intent is ControlIntent.CONFIRM:
            slot = self.selected_conflict_slot
            if slot is None:
                return SessionOperatorOutcome(False, "no_conflict_slot_to_reconcile")
            self.session.reconcile_active_slot(
                decision_id=f"reconcile:{control.observation_id}",
                slot=slot,
                operator_id=self.operator_id,
                reason="operator_confirmed_slot_empty",
                slot_empty_confirmed=True,
            )
            self._selected_slot_index = 0
            return SessionOperatorOutcome(
                True, "slot_reconciled", selected_slot=slot
            )
        if control.intent is ControlIntent.START:
            if self.selected_conflict_slot is not None:
                return SessionOperatorOutcome(False, "reconcile_all_conflicts_first")
            self.session.retry_active_hand(
                decision_id=f"retry:{control.observation_id}",
                operator_id=self.operator_id,
                reason="operator_confirmed_state_parity",
                state_parity_confirmed=True,
            )
            return SessionOperatorOutcome(
                True,
                "hand_retry_started",
                SessionOperatorSignal.RETRY_HAND,
            )
        if control.intent in {ControlIntent.CLEAR, ControlIntent.CANCEL}:
            self.session.void_active_hand(
                decision_id=f"void:{control.observation_id}",
                operator_id=self.operator_id,
                reason="operator_voided_paused_hand",
            )
            return SessionOperatorOutcome(
                True,
                "hand_voided",
                SessionOperatorSignal.HAND_VOIDED,
            )
        return SessionOperatorOutcome(False, "unsupported_recovery_control")


__all__ = [
    "SessionOperatorController",
    "SessionOperatorOutcome",
    "SessionOperatorSignal",
]
