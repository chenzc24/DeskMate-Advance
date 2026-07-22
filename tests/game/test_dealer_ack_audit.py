import pytest

from poker_dealer.domain import (
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerDeviceState,
    DealerSensorEvidence,
    Seat,
)
from poker_dealer.game import HandEngine


def _dispense_ack(command_id: str, version: int, *, deck_present: bool = True):
    return DealerAck(
        command_id=command_id,
        command=DealerCommandType.DISPENSE_ONE,
        target_slot=None,
        status=DealerAckStatus.SUCCEEDED,
        observed_at_ns=100 + version,
        device_state=DealerDeviceState.READY,
        device_state_version=version,
        sensor_evidence=DealerSensorEvidence(
            homed=True,
            at_target=True,
            deck_present=deck_present,
            exit_pulses=1,
            interlock_closed=True,
            emergency_stop=False,
        ),
    )


def test_successful_dispense_requires_deck_present() -> None:
    with pytest.raises(ValueError, match="deck_present"):
        _dispense_ack("dispense-1", 1, deck_present=False)


def test_raw_ack_is_not_accepted_until_correlated_completion() -> None:
    engine = HandEngine.setup_session("ack-audit", Seat.A)
    command = DealerCommand(
        "dispense-1", 10, DealerCommandType.DISPENSE_ONE
    )
    engine.record_dealer_command(command)
    ack = _dispense_ack(command.command_id, 1)
    engine.record_dealer_ack(ack)
    assert engine.state.pending_command_id == command.command_id
    assert engine.log.events[-1].kind == "dealer_ack_received"
    assert engine.log.events[-1].accepted is False
    assert engine.log.events[-1].payload["sensor_evidence"]["deck_present"] is True

    engine.record_dealer_completion(ack)
    assert engine.state.pending_command_id is None
    assert engine.log.events[-1].kind == "dealer_command_completed"
    assert engine.log.events[-1].accepted is True


def test_device_state_version_must_increase_between_completions() -> None:
    engine = HandEngine.setup_session("ack-version", Seat.A)
    first = DealerCommand("dispense-1", 10, DealerCommandType.DISPENSE_ONE)
    engine.record_dealer_command(first)
    engine.record_dealer_completion(_dispense_ack(first.command_id, 4))
    second = DealerCommand("dispense-2", 20, DealerCommandType.DISPENSE_ONE)
    engine.record_dealer_command(second)
    with pytest.raises(ValueError, match="non-monotonic"):
        engine.record_dealer_completion(_dispense_ack(second.command_id, 4))
