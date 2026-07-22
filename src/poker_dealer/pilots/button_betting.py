"""Pilot-only button action selector; never a production action-input path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    HandPhase,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import ActionRequest, ActionResult, HandEngine


class ButtonBettingPhase(StrEnum):
    SELECTING = "selecting"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ButtonBettingOutcome:
    accepted: bool
    reason: str
    selected_action: PlayerActionType | None
    action_result: ActionResult | None = None


class ButtonBettingRuntime:
    """Exercise rules/ledger directly, outside the product ``HandRuntime`` path."""

    def __init__(
        self, engine: HandEngine, *, allow_direct_engine_pilot: bool = False
    ) -> None:
        if not allow_direct_engine_pilot:
            raise ValueError(
                "direct button-to-engine submission is pilot-only; "
                "production controls must enter through HandRuntime/Part A"
            )
        self.engine = engine
        self.phase = ButtonBettingPhase.CLOSED
        self._selected_index = 0
        self._context: tuple[str, int, Seat] | None = None
        self._seen_controls: set[str] = set()
        self._last_device_versions: dict[tuple[str, str], int] = {}
        self._window_opened_at_ns = 0
        self.sync(window_opened_at_ns=0)

    @property
    def selected_action(self) -> PlayerActionType | None:
        if self.phase is ButtonBettingPhase.CLOSED:
            return None
        actions = self.engine.state.legal_actions
        return actions[self._selected_index] if actions else None

    def sync(self, *, window_opened_at_ns: int) -> None:
        state = self.engine.state
        if state.phase is not HandPhase.AWAITING_ACTION or state.acting_seat is None:
            self.phase = ButtonBettingPhase.CLOSED
            self._context = None
            self._selected_index = 0
            return
        self.phase = ButtonBettingPhase.SELECTING
        self._context = (state.hand_id, state.state_version, state.acting_seat)
        self._selected_index = 0
        self._window_opened_at_ns = window_opened_at_ns

    def accept_control(self, observation: ControlObservation) -> ButtonBettingOutcome:
        if observation.observation_id in self._seen_controls:
            return self._outcome(False, "duplicate_control")
        self._seen_controls.add(observation.observation_id)
        device_key = (observation.source.value, observation.control_id)
        previous_device_version = self._last_device_versions.get(device_key, -1)
        if observation.device_state_version <= previous_device_version:
            return self._outcome(False, "stale_device_state_version")
        self._last_device_versions[device_key] = observation.device_state_version
        if observation.observed_at_ns < self._window_opened_at_ns:
            return self._outcome(False, "control_precedes_action_window")
        state = self.engine.state
        current_context = (state.hand_id, state.state_version, state.acting_seat)
        if self.phase is not ButtonBettingPhase.SELECTING or self._context != current_context:
            return self._outcome(False, "action_window_context_changed")
        actions = state.legal_actions
        if not actions:
            return self._outcome(False, "no_legal_actions")
        if observation.intent is ControlIntent.NEXT_OPTION:
            self._selected_index = (self._selected_index + 1) % len(actions)
            return self._outcome(True, "selection_advanced")
        if observation.intent is ControlIntent.PREVIOUS_OPTION:
            self._selected_index = (self._selected_index - 1) % len(actions)
            return self._outcome(True, "selection_reversed")
        if observation.intent is ControlIntent.CANCEL:
            self._selected_index = 0
            return self._outcome(True, "selection_reset")
        if observation.intent is not ControlIntent.CONFIRM:
            return self._outcome(False, "unsupported_betting_control")
        action = actions[self._selected_index]
        assert state.acting_seat is not None
        result = self.engine.apply_action(
            ActionRequest(
                action_id=f"control:{observation.observation_id}",
                hand_id=state.hand_id,
                expected_state_version=state.state_version,
                seat=state.acting_seat,
                action=action,
                source=observation.source.value,
            )
        )
        if result.accepted:
            committed = action
            self.sync(window_opened_at_ns=observation.observed_at_ns)
            return ButtonBettingOutcome(True, result.reason, committed, result)
        return ButtonBettingOutcome(False, result.reason, action, result)

    def _outcome(self, accepted: bool, reason: str) -> ButtonBettingOutcome:
        return ButtonBettingOutcome(accepted, reason, self.selected_action)


__all__ = ["ButtonBettingOutcome", "ButtonBettingPhase", "ButtonBettingRuntime"]
