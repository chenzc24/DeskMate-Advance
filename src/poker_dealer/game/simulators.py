"""Deterministic Stage 1 simulators for action, card and dealer evidence."""

from __future__ import annotations

from dataclasses import dataclass

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    CardObservation,
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerDeviceState,
    DealerErrorCode,
    DealerSensorEvidence,
    ObservationStatus,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
    VisionSlot,
)

from .engine import HandState


@dataclass(frozen=True, slots=True)
class DealerFault:
    status: DealerAckStatus
    error_code: DealerErrorCode
    reason: str


class SimulatedDealer:
    def __init__(self) -> None:
        self._acks: dict[str, DealerAck] = {}
        self._faults: dict[str, DealerFault] = {}
        self.homed = False
        self.at_target = False
        self.dispensed_cards = 0

    def inject_fault(self, command_id: str, fault: DealerFault) -> None:
        self._faults[command_id] = fault

    def restart(self) -> None:
        """Simulate an MCU restart while retaining the command dedupe journal."""

        self.homed = False
        self.at_target = False

    def execute_and_deliver(
        self,
        commands: tuple[DealerCommand, ...],
        delivery_order: tuple[str, ...],
    ) -> tuple[DealerAck, ...]:
        """Execute in issue order, then deliver ACKs in a scripted order."""

        command_ids = tuple(command.command_id for command in commands)
        if len(set(command_ids)) != len(command_ids):
            raise ValueError("commands must have unique IDs")
        if len(delivery_order) != len(command_ids) or set(delivery_order) != set(
            command_ids
        ):
            raise ValueError("delivery_order must contain every command ID once")
        issued = {
            command.command_id: self.execute(command, command.issued_at_ns + 1)
            for command in commands
        }
        return tuple(issued[command_id] for command_id in delivery_order)

    def execute(self, command: DealerCommand, observed_at_ns: int = 1) -> DealerAck:
        if command.command_id in self._acks:
            return self._acks[command.command_id]

        fault = self._faults.get(command.command_id)
        if fault is not None:
            ack = DealerAck(
                command.command_id,
                command.command,
                command.target_slot,
                fault.status,
                observed_at_ns,
                DealerDeviceState.FAULT,
                len(self._acks) + 1,
                DealerSensorEvidence(
                    self.homed,
                    self.at_target,
                    True,
                    0,
                    True,
                    False,
                ),
                fault.error_code,
                fault.reason,
            )
            self._acks[command.command_id] = ack
            return ack

        if command.command is DealerCommandType.HOME:
            self.homed = True
            self.at_target = False
        elif command.command is DealerCommandType.ROTATE_TO:
            if not self.homed:
                return self._failed_not_homed(command, observed_at_ns)
            self.at_target = True
        elif command.command is DealerCommandType.DISPENSE_ONE:
            if not self.homed or not self.at_target:
                return self._failed_not_homed(command, observed_at_ns)
            self.dispensed_cards += 1

        evidence = DealerSensorEvidence(
            homed=self.homed,
            at_target=self.at_target,
            deck_present=True,
            exit_pulses=(
                1 if command.command is DealerCommandType.DISPENSE_ONE else 0
            ),
            interlock_closed=True,
            emergency_stop=False,
        )
        ack = DealerAck(
            command.command_id,
            command.command,
            command.target_slot,
            DealerAckStatus.SUCCEEDED,
            observed_at_ns,
            DealerDeviceState.READY,
            len(self._acks) + 1,
            evidence,
        )
        self._acks[command.command_id] = ack
        return ack

    def _failed_not_homed(
        self, command: DealerCommand, observed_at_ns: int
    ) -> DealerAck:
        ack = DealerAck(
            command.command_id,
            command.command,
            command.target_slot,
            DealerAckStatus.REJECTED,
            observed_at_ns,
            DealerDeviceState.NOT_HOMED,
            len(self._acks) + 1,
            DealerSensorEvidence(False, False, True, 0, True, False),
            DealerErrorCode.NOT_HOMED,
            "simulated dealer is not homed and at target",
        )
        self._acks[command.command_id] = ack
        return ack


class SimulatedCardPerception:
    def __init__(self) -> None:
        self.observations: dict[VisionSlot, CardObservation] = {}

    def emit(
        self,
        slot: VisionSlot,
        status: ObservationStatus,
        *,
        card: CardIdentity | None = None,
        confidence: float | None = None,
        observed_at_ns: int = 1,
        stable_frames: int = 3,
        quality_flags: tuple[str, ...] = (),
    ) -> CardObservation:
        observation = CardObservation(
            observation_id=f"sim-card-{slot.value}-{observed_at_ns}",
            slot_id=slot,
            observed_at_ns=observed_at_ns,
            status=status,
            card=card,
            confidence=confidence,
            model_version="simulated-card-v1",
            calibration_version="simulated-table-v1",
            stable_frames=stable_frames,
            quality_flags=quality_flags,
        )
        self.observations[slot] = observation
        return observation

    def confirmed_cards(self) -> dict[VisionSlot, CardIdentity]:
        confirmed = {
            slot: observation.card
            for slot, observation in self.observations.items()
            if observation.status is ObservationStatus.CONFIRMED
        }
        cards = tuple(card for card in confirmed.values() if card is not None)
        if len(cards) != len(set(cards)):
            raise ValueError("duplicate_card_identity")
        return {slot: card for slot, card in confirmed.items() if card is not None}


class SimulatedActionPerception:
    def __init__(self) -> None:
        self.sequence = 0

    def emit(
        self,
        state: HandState,
        evidence_state: ActionEvidenceState,
        *,
        action: PlayerActionType | None = None,
        focus_seat: Seat | None = None,
        confidence: float | None = None,
        stable_duration_ms: int = 400,
        stable_frames: int = 12,
        expected_state_version: int | None = None,
        observed_at_ns: int | None = None,
    ) -> PlayerActionObservation:
        self.sequence += 1
        observed = observed_at_ns or self.sequence * 1_000_000_000
        return PlayerActionObservation(
            observation_id=f"sim-action-{self.sequence}",
            hand_id=state.hand_id,
            expected_state_version=(
                state.state_version
                if expected_state_version is None
                else expected_state_version
            ),
            window_started_at_ns=max(0, observed - stable_duration_ms * 1_000_000),
            observed_at_ns=observed,
            focus_seat=focus_seat or state.acting_seat or state.button,
            evidence_state=evidence_state,
            candidate_action=action,
            confidence=confidence,
            stable_duration_ms=stable_duration_ms,
            stable_frames=stable_frames,
            model_version="simulated-action-v1",
            calibration_version="simulated-action-rois-v1",
        )
