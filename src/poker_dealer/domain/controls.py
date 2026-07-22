"""Transport-neutral operator controls for laptop and future robot buttons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ControlIntent(StrEnum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    START = "start"
    CLEAR = "clear"
    NEXT_OPTION = "next_option"
    PREVIOUS_OPTION = "previous_option"


class ControlSource(StrEnum):
    LAPTOP_KEYBOARD = "laptop_keyboard"
    ROBOT_BUTTON = "robot_button"
    SIMULATOR = "simulator"


@dataclass(frozen=True, slots=True)
class ControlObservation:
    """One semantic button observation; it never mutates game state itself."""

    observation_id: str
    intent: ControlIntent
    source: ControlSource
    observed_at_ns: int
    control_id: str
    device_state_version: int

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.control_id.strip():
            raise ValueError("control observation and control IDs are required")
        if self.observed_at_ns < 0 or self.device_state_version < 0:
            raise ValueError("control timestamp/version must be non-negative")


class LaptopControlAdapter:
    """Map the retained laptop fallback keys onto semantic controls."""

    _KEYS = {
        ord("e"): ControlIntent.CONFIRM,
        13: ControlIntent.CONFIRM,
        ord("s"): ControlIntent.START,
        ord("x"): ControlIntent.CLEAR,
        ord("n"): ControlIntent.NEXT_OPTION,
        ord("p"): ControlIntent.PREVIOUS_OPTION,
        8: ControlIntent.CANCEL,
    }

    def __init__(self) -> None:
        self._sequence = 0

    def process_key(self, key: int, observed_at_ns: int) -> ControlObservation | None:
        intent = self._KEYS.get(key)
        if intent is None:
            return None
        self._sequence += 1
        return ControlObservation(
            observation_id=f"laptop-control:{self._sequence}:{observed_at_ns}",
            intent=intent,
            source=ControlSource.LAPTOP_KEYBOARD,
            observed_at_ns=observed_at_ns,
            control_id="keyboard",
            device_state_version=self._sequence,
        )


class RobotButtonAdapter:
    """Boundary for a future robot transport: short=confirm, long=cancel."""

    def process_press(
        self,
        *,
        event_id: str,
        button_id: str,
        observed_at_ns: int,
        device_state_version: int,
        long_press: bool = False,
        next_option: bool = False,
    ) -> ControlObservation:
        if long_press and next_option:
            raise ValueError("one robot press cannot be both cancel and next")
        intent = ControlIntent.CONFIRM
        if long_press:
            intent = ControlIntent.CANCEL
        elif next_option:
            intent = ControlIntent.NEXT_OPTION
        return ControlObservation(
            observation_id=event_id,
            intent=intent,
            source=ControlSource.ROBOT_BUTTON,
            observed_at_ns=observed_at_ns,
            control_id=button_id,
            device_state_version=device_state_version,
        )
