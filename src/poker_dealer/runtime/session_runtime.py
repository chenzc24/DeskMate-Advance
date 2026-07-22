"""Multi-hand session authority for roster, stacks, button and recovery gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from poker_dealer.domain import HandPhase, SEAT_ORDER, Seat, role_seats
from poker_dealer.game import CoreGameConfig

from .hand_runtime import HandRuntime
from .registration import FrozenSessionRoster, RegisteredParticipant


@dataclass(frozen=True, slots=True)
class SessionAuditEvent:
    sequence: int
    kind: str
    payload: Mapping[str, object]


class SessionRuntime:
    """Own continuity between hands; a `HandRuntime` still owns each hand."""

    def __init__(
        self,
        roster: FrozenSessionRoster,
        game_config: CoreGameConfig,
        *,
        stacks: Mapping[Seat, int] | None = None,
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
        self._events: list[SessionAuditEvent] = []

    @property
    def events(self) -> tuple[SessionAuditEvent, ...]:
        return tuple(self._events)

    def start_hand(self, hand_id: str) -> HandRuntime:
        if self.active_hand is not None:
            raise ValueError("the previous hand has not been closed")
        if not self._table_cleared:
            raise ValueError("table clearance must be confirmed before the next hand")
        if not hand_id.strip() or hand_id in self._hand_ids:
            raise ValueError("hand ID must be non-empty and unique in the session")
        below_minimum = {
            seat: stack
            for seat, stack in self.stacks.items()
            if stack < self.game_config.minimum_stack_to_start_hand_units
        }
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
        )
        self.active_hand = runtime
        self._table_cleared = False
        self._hand_ids.add(hand_id)
        self._append("hand_started", {"hand_id": hand_id, "button": self.button.value})
        return runtime

    def close_terminal_hand(self) -> None:
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
                "stacks": {seat.value: self.stacks[seat] for seat in SEAT_ORDER},
            },
        )
        self.active_hand = None

    def confirm_table_cleared(
        self, *, operator_id: str, reason: str = "all_cards_manually_returned"
    ) -> None:
        if self.active_hand is not None:
            raise ValueError("cannot clear the table while a hand is active")
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
        if self.active_hand is not None:
            raise ValueError("session stack adjustments are between-hand only")
        if not adjustment_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("adjustment ID, operator and reason are required")
        if any(
            event.kind == "stack_adjusted"
            and event.payload.get("adjustment_id") == adjustment_id
            for event in self._events
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

    def _roster_for_button(self, button: Seat) -> FrozenSessionRoster:
        roles_by_seat = {seat: role for role, seat in role_seats(button).items()}
        participants = tuple(
            RegisteredParticipant(
                participant.participant_id,
                participant.seat,
                roles_by_seat[participant.seat],
                participant.face_sample_count,
                participant.voice_enrolled,
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
        self._events.append(SessionAuditEvent(len(self._events) + 1, kind, payload))


__all__ = ["SessionAuditEvent", "SessionRuntime"]
