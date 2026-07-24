"""OpenCV operator boundary for between-hand and recovery session controls."""

from __future__ import annotations

from dataclasses import dataclass
import time

from poker_dealer.domain import SEAT_ORDER, role_seats

from .live_perception import InteractiveOpenCVFrameSource
from .ports import ControlSource, FrameReadState
from .session_control import (
    SessionOperatorController,
    SessionOperatorOutcome,
    SessionOperatorSignal,
)
from .session_runtime import SessionRuntime


@dataclass(frozen=True, slots=True)
class LiveSessionBoundaryResult:
    signal: SessionOperatorSignal
    reason: str


class LiveSessionOperatorUI:
    def __init__(
        self,
        frame_source: InteractiveOpenCVFrameSource,
        control_source: ControlSource,
        *,
        state_observer: object | None = None,
        event_announcer: object | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.control_source = control_source
        self.state_observer = state_observer
        self.event_announcer = event_announcer

    def wait_for_decision(
        self,
        session: SessionRuntime,
        controller: SessionOperatorController,
        *,
        timeout_seconds: float | None = None,
        stop_after_clear: bool = False,
    ) -> LiveSessionBoundaryResult:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("session decision timeout must be positive")
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
        )
        last_reason = "waiting_for_operator"
        while deadline is None or time.monotonic_ns() < deadline:
            if self.state_observer is not None:
                publish = getattr(self.state_observer, "publish_session_state", None)
                if publish is not None:
                    publish(
                        session,
                        last_reason=last_reason,
                        stop_after_clear=stop_after_clear,
                        selected_seat=controller.selected_low_stack_seat,
                        selected_slot=controller.selected_conflict_slot,
                    )
            self.frame_source.set_status(
                *self._status_lines(
                    session, controller, last_reason, stop_after_clear=stop_after_clear
                )
            )
            read = self.frame_source.read()
            if read.state is FrameReadState.DISCONNECTED:
                raise RuntimeError(read.reason or "camera disconnected at session boundary")
            for control in self.control_source.poll_controls(read.observed_at_ns):
                outcome = controller.accept(control)
                last_reason = outcome.reason
                self._announce_outcome(session, outcome)
                if outcome.accepted and outcome.reason == "table_cleared" and stop_after_clear:
                    session.end_session(
                        operator_id=controller.operator_id,
                        reason="configured_hand_limit_reached",
                    )
                    return LiveSessionBoundaryResult(
                        SessionOperatorSignal.SESSION_ENDED,
                        "session_ended_after_table_clear",
                    )
                if outcome.accepted and outcome.signal is not SessionOperatorSignal.CONTINUE_WAITING:
                    return LiveSessionBoundaryResult(outcome.signal, outcome.reason)
        raise TimeoutError("session operator decision deadline expired")

    def _announce_outcome(
        self,
        session: SessionRuntime,
        outcome: SessionOperatorOutcome,
    ) -> None:
        if not outcome.accepted or self.event_announcer is None:
            return
        publish = getattr(self.event_announcer, "publish", None)
        if publish is None:
            return
        if outcome.reason == "table_cleared":
            publish(
                "next_hand_ready"
                if not session.low_stack_seats
                else "operator_adjustment"
            )
        elif outcome.reason == "rebuy_applied" and outcome.selected_seat is not None:
            roles = {
                seat: role.value for role, seat in role_seats(session.button).items()
            }
            publish(
                "rebuy_confirmed",
                role=roles[outcome.selected_seat],
            )
        elif outcome.reason == "slot_reconciled":
            publish("operator_adjustment")
        elif outcome.reason == "hand_retry_started":
            publish("recovery_resumed")
        elif outcome.reason == "hand_voided":
            publish("hand_voided")

    @staticmethod
    def _status_lines(
        session: SessionRuntime,
        controller: SessionOperatorController,
        last_reason: str,
        *,
        stop_after_clear: bool,
    ) -> tuple[str, ...]:
        runtime = session.active_hand
        stacks = " | ".join(
            f"{seat.value[-1].upper()}:{session.stacks[seat]}" for seat in SEAT_ORDER
        )
        if runtime is not None:
            selected = controller.selected_conflict_slot
            return (
                f"RECOVERY: {runtime.engine.state.paused_reason or 'unknown'}",
                "S retry after state check | X/Backspace void hand",
                (
                    f"N/P select conflict | C confirms empty: {selected.value}"
                    if selected is not None
                    else "No card conflict remains"
                ),
                last_reason,
            )
        low = controller.selected_low_stack_seat
        if not session.table_cleared:
            instruction = (
                "Return every card, then C/E/Enter ends session"
                if stop_after_clear
                else "Return every card, then C/E/Enter confirms table clear"
            )
        elif low is not None:
            instruction = (
                f"N/P select | C/E rebuy {low.value} to "
                f"{controller.rebuy_to_units} | X ends session"
            )
        else:
            instruction = "S starts next hand | X/Backspace ends session"
        return (
            f"SESSION: next Button {session.button.value} | hands {session.next_hand_number - 1}",
            stacks,
            instruction,
            last_reason,
        )


__all__ = ["LiveSessionBoundaryResult", "LiveSessionOperatorUI"]
