"""Multi-hand session authority for roster, stacks, button and recovery gates."""

from __future__ import annotations

from typing import Mapping

from poker_dealer.domain import HandPhase, SEAT_ORDER, Seat, VisionSlot, role_seats
from poker_dealer.game import CoreGameConfig, PromotionPolicy

from .hand_runtime import HandRuntime
from .registration import FrozenSessionRoster, RegisteredParticipant
from .session_log import SessionAuditEvent, SessionEventLog


class SessionRuntime:
    """Own continuity between hands; a `HandRuntime` still owns each hand."""

    def __init__(
        self,
        roster: FrozenSessionRoster,
        game_config: CoreGameConfig,
        *,
        stacks: Mapping[Seat, int] | None = None,
        log: SessionEventLog | None = None,
        action_promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        if len(roster.participants) != 4:
            raise ValueError("session requires four registered participants")
        if {item.seat for item in roster.participants} != set(SEAT_ORDER):
            raise ValueError("session roster must cover all four seats")
        self.roster = roster
        self.game_config = game_config
        self.stacks = dict(stacks or game_config.default_stacks())
        if set(self.stacks) != set(SEAT_ORDER) or min(self.stacks.values()) < 0:
            raise ValueError("session stacks must cover all seats and be non-negative")
        self.button = roster.button
        self.active_hand: HandRuntime | None = None
        self._table_cleared = True
        self._hand_ids: set[str] = set()
        self.log = log or SessionEventLog()
        self.action_promotion_policy = action_promotion_policy
        self._ended = False
        self._append(
            "session_started",
            {
                "session_id": roster.session_id,
                "roster_version": roster.roster_version,
                "button": self.button.value,
                "rules_version": game_config.rules.rules_version,
                "stacks": self._stack_payload(),
                "players": {
                    participant.seat.value: participant.participant_id
                    for participant in roster.participants
                },
            },
        )

    @property
    def events(self) -> tuple[SessionAuditEvent, ...]:
        return tuple(self.log.events)

    @property
    def table_cleared(self) -> bool:
        return self._table_cleared

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def low_stack_seats(self) -> tuple[Seat, ...]:
        minimum = self.game_config.minimum_stack_to_start_hand_units
        return tuple(seat for seat in SEAT_ORDER if self.stacks[seat] < minimum)

    @property
    def next_hand_number(self) -> int:
        return len(self._hand_ids) + 1

    def start_hand(self, hand_id: str) -> HandRuntime:
        if self._ended:
            raise ValueError("session has ended")
        if self.active_hand is not None:
            raise ValueError("the previous hand has not been closed")
        if not self._table_cleared:
            raise ValueError("table clearance must be confirmed before the next hand")
        if not hand_id.strip() or hand_id in self._hand_ids:
            raise ValueError("hand ID must be non-empty and unique in the session")
        below_minimum = {seat: self.stacks[seat] for seat in self.low_stack_seats}
        if below_minimum:
            raise ValueError(
                "all players must meet the configured minimum stack: "
                + ",".join(f"{seat.value}={stack}" for seat, stack in below_minimum.items())
            )
        hand_roster = self._roster_for_button(self.button)
        runtime = HandRuntime.from_roster(
            hand_id=hand_id,
            roster=hand_roster,
            stacks=self.stacks,
            rules=self.game_config.rules,
            action_promotion_policy=self.action_promotion_policy,
        )
        self.active_hand = runtime
        self._table_cleared = False
        self._hand_ids.add(hand_id)
        self._append(
            "hand_started",
            {
                "hand_id": hand_id,
                "button": self.button.value,
                "starting_stacks": self._stack_payload(),
            },
        )
        return runtime

    def close_terminal_hand(
        self,
        *,
        hand_log_path: str | None = None,
        hand_log_sha256: str | None = None,
        hand_log_check_passed: bool | None = None,
    ) -> None:
        runtime = self.active_hand
        if runtime is None:
            raise ValueError("no active hand")
        if runtime.phase not in {HandPhase.SETTLED, HandPhase.VOIDED}:
            raise ValueError("only a settled or voided hand can be closed")
        self.stacks = {
            seat: runtime.engine.state.players[seat].stack_units
            for seat in SEAT_ORDER
        }
        old_button = self.button
        self.button = runtime.engine.next_button()
        self._append(
            "hand_closed",
            {
                "hand_id": runtime.engine.state.hand_id,
                "terminal_phase": runtime.phase.value,
                "button_before": old_button.value,
                "button_after": self.button.value,
                "stacks": self._stack_payload(),
                "hand_log_path": hand_log_path,
                "hand_log_sha256": hand_log_sha256,
                "hand_log_check_passed": hand_log_check_passed,
            },
        )
        self.active_hand = None

    def confirm_table_cleared(
        self, *, operator_id: str, reason: str = "all_cards_manually_returned"
    ) -> None:
        if self._ended:
            raise ValueError("session has ended")
        if self.active_hand is not None:
            raise ValueError("cannot clear the table while a hand is active")
        if self._table_cleared:
            raise ValueError("table is already confirmed clear")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("table-clear operator and reason are required")
        self._table_cleared = True
        self._append(
            "table_cleared",
            {"operator_id": operator_id, "reason": reason},
        )

    def adjust_stack(
        self,
        *,
        adjustment_id: str,
        seat: Seat,
        amount_units: int,
        operator_id: str,
        reason: str,
    ) -> None:
        if self._ended:
            raise ValueError("session has ended")
        if self.active_hand is not None:
            raise ValueError("session stack adjustments are between-hand only")
        if not adjustment_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("adjustment ID, operator and reason are required")
        if any(
            event.kind == "stack_adjusted"
            and event.payload.get("adjustment_id") == adjustment_id
            for event in self.log.events
        ):
            return
        updated = self.stacks[seat] + amount_units
        if updated < 0:
            raise ValueError("stack adjustment cannot make a balance negative")
        self.stacks[seat] = updated
        self._append(
            "stack_adjusted",
            {
                "adjustment_id": adjustment_id,
                "seat": seat.value,
                "amount_units": amount_units,
                "operator_id": operator_id,
                "reason": reason,
                "stack_after": updated,
            },
        )

    def retry_active_hand(
        self,
        *,
        decision_id: str,
        operator_id: str,
        reason: str,
        state_parity_confirmed: bool,
    ) -> HandRuntime:
        runtime = self._paused_hand()
        runtime.resume_from_recovery(
            decision_id,
            operator_id=operator_id,
            reason=reason,
            physical_state_confirmed=state_parity_confirmed,
        )
        self._append(
            "recovery_decision",
            {
                "decision_id": decision_id,
                "hand_id": runtime.engine.state.hand_id,
                "decision": "retry",
                "operator_id": operator_id,
                "reason": reason,
                "state_parity_confirmed": state_parity_confirmed,
            },
        )
        return runtime

    def reconcile_active_slot(
        self,
        *,
        decision_id: str,
        slot: VisionSlot,
        operator_id: str,
        reason: str,
        slot_empty_confirmed: bool,
    ) -> HandRuntime:
        runtime = self._paused_hand()
        runtime.reconcile_card_slot(
            decision_id,
            slot=slot,
            operator_id=operator_id,
            reason=reason,
            physical_slot_empty=slot_empty_confirmed,
        )
        self._append(
            "recovery_decision",
            {
                "decision_id": decision_id,
                "hand_id": runtime.engine.state.hand_id,
                "decision": "reconcile_slot",
                "slot_id": slot.value,
                "operator_id": operator_id,
                "reason": reason,
                "slot_empty_confirmed": slot_empty_confirmed,
            },
        )
        return runtime

    def void_active_hand(
        self, *, decision_id: str, operator_id: str, reason: str
    ) -> HandRuntime:
        runtime = self.active_hand
        if runtime is None:
            raise ValueError("no active hand")
        if runtime.phase in {HandPhase.SETTLED, HandPhase.VOIDED}:
            raise ValueError("terminal hand cannot be voided again")
        if not decision_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("void decision, operator and reason are required")
        runtime.void(decision_id, reason)
        self._append(
            "recovery_decision",
            {
                "decision_id": decision_id,
                "hand_id": runtime.engine.state.hand_id,
                "decision": "void",
                "operator_id": operator_id,
                "reason": reason,
            },
        )
        return runtime

    def end_session(self, *, operator_id: str, reason: str) -> None:
        if self._ended:
            return
        if self.active_hand is not None:
            raise ValueError("cannot end session while a hand is active")
        if not self._table_cleared:
            raise ValueError("table must be confirmed clear before ending session")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("session-end operator and reason are required")
        self._ended = True
        self._append(
            "session_ended",
            {
                "operator_id": operator_id,
                "reason": reason,
                "hands": len(self._hand_ids),
                "final_button": self.button.value,
                "final_stacks": self._stack_payload(),
            },
        )

    def _roster_for_button(self, button: Seat) -> FrozenSessionRoster:
        roles_by_seat = {seat: role for role, seat in role_seats(button).items()}
        participants = tuple(
            RegisteredParticipant(
                participant.participant_id,
                participant.seat,
                roles_by_seat[participant.seat],
                participant.face_sample_count,
                participant.voice_enrolled,
                participant.simulated,
            )
            for participant in self.roster.participants
        )
        return FrozenSessionRoster(
            self.roster.session_id,
            button,
            self.roster.roster_version,
            participants,
        )

    def _append(self, kind: str, payload: Mapping[str, object]) -> None:
        self.log.append(kind, payload)

    def _paused_hand(self) -> HandRuntime:
        runtime = self.active_hand
        if runtime is None or runtime.phase is not HandPhase.PAUSED_RECOVERY:
            raise ValueError("active hand is not paused for recovery")
        return runtime

    def _stack_payload(self) -> dict[str, int]:
        return {seat.value: self.stacks[seat] for seat in SEAT_ORDER}


__all__ = ["SessionAuditEvent", "SessionRuntime"]
