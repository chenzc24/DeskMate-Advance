"""Dealer mapping for cocino_car's command-ACK-only `dispense_one` action."""

from __future__ import annotations

import time
from typing import Callable, Mapping

from poker_dealer.domain import (
    DealerAck,
    DealerAckStatus,
    DealerCompletionBasis,
    DealerCommand,
    DealerCommandType,
    DealerDeviceState,
    DealerErrorCode,
    DealerSensorEvidence,
    NavigationAck,
    NavigationAckStatus,
)
from poker_dealer.robotics.navigation.cocino_car import (
    CocinoCarClient,
    CocinoCarProtocolError,
)
from poker_dealer.robotics.navigation.port import NavigationUnavailableError

from .port import DealerHealth, DealerUnavailableError


_UNKNOWN_EVIDENCE = DealerSensorEvidence(
    homed=None,
    at_target=None,
    deck_present=None,
    exit_pulses=None,
    interlock_closed=None,
    emergency_stop=None,
)
_COMMAND_ACK_EVIDENCE = DealerSensorEvidence(
    homed=None,
    at_target=True,
    deck_present=None,
    exit_pulses=None,
    interlock_closed=None,
    emergency_stop=None,
)


class CocinoCarDealerAdapter:
    """Map `DISPENSE_ONE` using the explicitly selected command-ACK basis."""

    physical_motion = True

    def __init__(
        self,
        client: CocinoCarClient,
        *,
        device_id: str = "cocino-car-dealer-http-v1",
        poll_interval_seconds: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("dealer poll interval must be positive")
        self.client = client
        self._device_id = device_id
        self.poll_interval_seconds = poll_interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._opened = False
        self._command_ack_available = False
        self._reason = "adapter is not open"
        self._device_state_version = 0
        self._at_target = False
        self._acks: dict[str, DealerAck] = {}

    @property
    def device_id(self) -> str:
        return self._device_id

    def open(self) -> None:
        capabilities = self.client.capabilities()
        actions = set(capabilities.get("actions", ()))
        if capabilities.get("available") is not True or "dispense_one" not in actions:
            raise DealerUnavailableError("cocino_car dispense API is unavailable")
        self._command_ack_available = (
            capabilities.get("dispense_completion_evidence")
            == DealerCompletionBasis.ARDUINO_COMMAND_ACK_ONLY.value
            and capabilities.get("request_terminal_status") is True
        )
        if not self._command_ack_available:
            raise DealerUnavailableError(
                "cocino_car command-ACK terminal result is unavailable"
            )
        self._opened = True
        self._reason = None

    def execute(
        self, command: DealerCommand, observed_at_ns: int | None = None
    ) -> DealerAck:
        if not self._opened:
            raise DealerUnavailableError(self._reason or "dealer adapter is not open")
        remembered = self._acks.get(command.command_id)
        if remembered is not None:
            return remembered
        observed = time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        if command.command is DealerCommandType.STOP:
            self.client.action(f"{command.command_id}:stop", "stop")
            self._device_state_version += 1
            ack = DealerAck(
                command_id=command.command_id,
                command=command.command,
                target_slot=None,
                status=DealerAckStatus.SUCCEEDED,
                observed_at_ns=observed,
                device_state=DealerDeviceState.READY,
                device_state_version=self._device_state_version,
                sensor_evidence=_UNKNOWN_EVIDENCE,
            )
            return self._remember(ack)
        if command.command is not DealerCommandType.DISPENSE_ONE:
            return self._remember(self._failure(
                command,
                observed,
                DealerErrorCode.PROTOCOL_ERROR,
                f"cocino_car dealer does not implement {command.command.value}",
            ))
        if not self._at_target:
            return self._remember(self._failure(
                command,
                observed,
                DealerErrorCode.INVALID_TARGET,
                "successful navigation target ACK is required before dispensing",
            ))
        try:
            self.client.action(command.command_id, "dispense_one")
            result = self._wait_terminal(command)
        except TimeoutError as exc:
            return self._remember(
                self._failure(
                    command,
                    time.monotonic_ns(),
                    DealerErrorCode.POSITION_TIMEOUT,
                    str(exc),
                    status=DealerAckStatus.TIMED_OUT,
                )
            )
        except (
            CocinoCarProtocolError,
            NavigationUnavailableError,
            ValueError,
        ) as exc:
            return self._remember(
                self._failure(
                    command,
                    time.monotonic_ns(),
                    DealerErrorCode.PROTOCOL_ERROR,
                    str(exc),
                    status=DealerAckStatus.FAILED,
                )
            )
        self._device_state_version += 1
        ack = DealerAck(
            command_id=command.command_id,
            command=command.command,
            target_slot=None,
            status=DealerAckStatus.SUCCEEDED,
            observed_at_ns=(
                time.monotonic_ns() if observed_at_ns is None else observed_at_ns
            ),
            device_state=DealerDeviceState.READY,
            device_state_version=self._device_state_version,
            sensor_evidence=_COMMAND_ACK_EVIDENCE,
            completion_basis=DealerCompletionBasis.ARDUINO_COMMAND_ACK_ONLY,
        )
        del result
        return self._remember(ack)

    def _wait_terminal(self, command: DealerCommand) -> Mapping[str, object]:
        started = self.clock()
        while True:
            result = self.client.request_result(command.command_id)
            if result.get("action") != "dispense_one":
                raise CocinoCarProtocolError(
                    "dispense request-result action mismatch"
                )
            status = result.get("request_status")
            terminal = result.get("terminal")
            if status not in {
                "running",
                "succeeded",
                "failed",
                "cancelled",
            } or not isinstance(terminal, bool):
                raise CocinoCarProtocolError(
                    "dispense result lacks terminal lifecycle fields"
                )
            if terminal:
                if status == "succeeded":
                    return result
                raise CocinoCarProtocolError(
                    f"dispense request ended as {status}"
                )
            if (self.clock() - started) * 1000 >= command.timeout_ms:
                raise TimeoutError("dispense request did not reach terminal status")
            self.sleeper(self.poll_interval_seconds)

    def confirm_navigation_target(self, acknowledgement: NavigationAck) -> None:
        self._at_target = bool(
            acknowledgement.status is NavigationAckStatus.SUCCEEDED
            and acknowledgement.target_aligned
        )
        if not self._at_target:
            raise DealerUnavailableError(
                "dealer target cannot be armed from failed navigation"
            )

    def health(self) -> DealerHealth:
        return DealerHealth(
            device_id=self.device_id,
            available=self._opened and self._command_ack_available,
            opened=self._opened,
            physical_motion=True,
            reason=self._reason,
        )

    def close(self) -> None:
        self._opened = False
        self._at_target = False

    def _remember(self, ack: DealerAck) -> DealerAck:
        self._acks[ack.command_id] = ack
        return ack

    def _failure(
        self,
        command: DealerCommand,
        observed_at_ns: int,
        error_code: DealerErrorCode,
        reason: str,
        *,
        status: DealerAckStatus = DealerAckStatus.REJECTED,
    ) -> DealerAck:
        self._device_state_version += 1
        return DealerAck(
            command_id=command.command_id,
            command=command.command,
            target_slot=command.target_slot,
            status=status,
            observed_at_ns=observed_at_ns,
            device_state=DealerDeviceState.FAULT,
            device_state_version=self._device_state_version,
            sensor_evidence=_UNKNOWN_EVIDENCE,
            error_code=error_code,
            reason=reason,
        )


__all__ = ["CocinoCarDealerAdapter"]
