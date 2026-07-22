"""Semantic dealer commands independent of serial or motor details."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DealerCommandType(StrEnum):
    HOME = "home"
    ROTATE_TO = "rotate_to"
    DISPENSE_ONE = "dispense_one"
    STOP = "stop"
    GET_STATUS = "get_status"
    RESET_FAULT = "reset_fault"


class DealerTargetSlot(StrEnum):
    SEAT_A = "seat_a"
    SEAT_B = "seat_b"
    SEAT_C = "seat_c"
    SEAT_D = "seat_d"
    BOARD_FLOP_1 = "board_flop_1"
    BOARD_FLOP_2 = "board_flop_2"
    BOARD_FLOP_3 = "board_flop_3"
    BOARD_TURN = "board_turn"
    BOARD_RIVER = "board_river"


@dataclass(frozen=True, slots=True)
class DealerCommand:
    command_id: str
    issued_at_ns: int
    command: DealerCommandType
    target_slot: DealerTargetSlot | None = None
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("command_id must not be empty")
        if self.issued_at_ns < 0:
            raise ValueError("issued_at_ns must be non-negative")
        if self.command is DealerCommandType.ROTATE_TO and not self.target_slot:
            raise ValueError("rotate_to requires target_slot")
        if self.command is not DealerCommandType.ROTATE_TO and self.target_slot:
            raise ValueError("only rotate_to accepts target_slot")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")


class DealerAckStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class DealerDeviceState(StrEnum):
    BOOTING = "booting"
    NOT_HOMED = "not_homed"
    READY = "ready"
    MOVING = "moving"
    DISPENSING = "dispensing"
    FAULT = "fault"
    E_STOP = "e_stop"


class DealerErrorCode(StrEnum):
    NOT_HOMED = "not_homed"
    INVALID_TARGET = "invalid_target"
    DECK_EMPTY = "deck_empty"
    FEED_JAM = "feed_jam"
    DOUBLE_FEED = "double_feed"
    POSITION_TIMEOUT = "position_timeout"
    INTERLOCK_OPEN = "interlock_open"
    EMERGENCY_STOP = "emergency_stop"
    TRANSPORT_LOST = "transport_lost"
    PROTOCOL_ERROR = "protocol_error"


@dataclass(frozen=True, slots=True)
class DealerSensorEvidence:
    homed: bool | None
    at_target: bool | None
    deck_present: bool | None
    exit_pulses: int | None
    interlock_closed: bool | None
    emergency_stop: bool | None

    def __post_init__(self) -> None:
        if self.exit_pulses is not None and self.exit_pulses < 0:
            raise ValueError("exit_pulses must be non-negative")


@dataclass(frozen=True, slots=True)
class DealerAck:
    command_id: str
    command: DealerCommandType
    target_slot: DealerTargetSlot | None
    status: DealerAckStatus
    observed_at_ns: int
    device_state: DealerDeviceState
    device_state_version: int
    sensor_evidence: DealerSensorEvidence
    error_code: DealerErrorCode | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("command_id must not be empty")
        if self.command is DealerCommandType.ROTATE_TO and not self.target_slot:
            raise ValueError("rotate_to acknowledgement requires target_slot")
        if self.command is not DealerCommandType.ROTATE_TO and self.target_slot:
            raise ValueError("only rotate_to acknowledgement accepts target_slot")
        if self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be non-negative")
        if self.device_state_version < 0:
            raise ValueError("device_state_version must be non-negative")
        if self.status is DealerAckStatus.SUCCEEDED:
            if self.error_code is not None or self.reason is not None:
                raise ValueError("successful acknowledgements cannot carry an error")
            if self.command is DealerCommandType.HOME:
                self._require_safe_evidence(homed=True)
            elif self.command is DealerCommandType.ROTATE_TO:
                self._require_safe_evidence(homed=True, at_target=True)
            elif self.command is DealerCommandType.DISPENSE_ONE:
                self._require_safe_evidence(
                    homed=True, at_target=True, exit_pulses=1
                )
        elif self.error_code is None or not self.reason:
            raise ValueError(
                "non-success acknowledgements require error_code and reason"
            )

    def _require_safe_evidence(
        self,
        *,
        homed: bool,
        at_target: bool | None = None,
        exit_pulses: int | None = None,
    ) -> None:
        evidence = self.sensor_evidence
        if evidence.homed is not homed:
            raise ValueError("successful motion acknowledgement requires homed=true")
        if at_target is not None and evidence.at_target is not at_target:
            raise ValueError(
                "successful target action acknowledgement requires at_target=true"
            )
        if exit_pulses is not None and evidence.exit_pulses != exit_pulses:
            raise ValueError("successful dispense acknowledgement requires one exit pulse")
        if evidence.interlock_closed is not True:
            raise ValueError("successful motion acknowledgement requires closed interlock")
        if evidence.emergency_stop is not False:
            raise ValueError("successful motion acknowledgement requires inactive E-stop")
