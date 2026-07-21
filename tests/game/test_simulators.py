from __future__ import annotations

import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerErrorCode,
    DealerTargetSlot,
    ObservationStatus,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    VisionSlot,
)
from poker_dealer.game import (
    DealerFault,
    HandEngine,
    SimulatedActionPerception,
    SimulatedCardPerception,
    SimulatedDealer,
)


def test_simulated_dealer_is_idempotent_and_fault_injectable() -> None:
    dealer = SimulatedDealer()
    rejected = dealer.execute(DealerCommand("r0", 1, DealerCommandType.ROTATE_TO, DealerTargetSlot.SEAT_A))
    assert rejected.status is DealerAckStatus.REJECTED
    dealer.execute(DealerCommand("home", 2, DealerCommandType.HOME))
    dealer.execute(DealerCommand("rotate", 3, DealerCommandType.ROTATE_TO, DealerTargetSlot.SEAT_A))
    command = DealerCommand("deal", 4, DealerCommandType.DISPENSE_ONE)
    first = dealer.execute(command)
    second = dealer.execute(command)
    assert first == second
    assert dealer.dispensed_cards == 1

    dealer.inject_fault(
        "jam", DealerFault(DealerAckStatus.FAILED, DealerErrorCode.FEED_JAM, "jam")
    )
    assert dealer.execute(DealerCommand("jam", 5, DealerCommandType.DISPENSE_ONE)).status is DealerAckStatus.FAILED


def test_all_ten_dealer_targets_and_out_of_order_ack_delivery() -> None:
    dealer = SimulatedDealer()
    dealer.execute(DealerCommand("home", 1, DealerCommandType.HOME))
    for index, target in enumerate(DealerTargetSlot, start=2):
        ack = dealer.execute(
            DealerCommand(
                f"rotate-{target.value}",
                index,
                DealerCommandType.ROTATE_TO,
                target,
            )
        )
        assert ack.status is DealerAckStatus.SUCCEEDED
        assert ack.target_slot is target

    scripted = SimulatedDealer()
    commands = (
        DealerCommand("ordered-home", 20, DealerCommandType.HOME),
        DealerCommand(
            "ordered-rotate",
            21,
            DealerCommandType.ROTATE_TO,
            DealerTargetSlot.BOARD_FLOP_1,
        ),
        DealerCommand("ordered-deal", 22, DealerCommandType.DISPENSE_ONE),
    )
    delivered = scripted.execute_and_deliver(
        commands, ("ordered-deal", "ordered-home", "ordered-rotate")
    )
    assert tuple(ack.command_id for ack in delivered) == (
        "ordered-deal",
        "ordered-home",
        "ordered-rotate",
    )
    assert all(ack.status is DealerAckStatus.SUCCEEDED for ack in delivered)
    assert scripted.dispensed_cards == 1


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        (DealerAckStatus.TIMED_OUT, DealerErrorCode.POSITION_TIMEOUT),
        (DealerAckStatus.FAILED, DealerErrorCode.FEED_JAM),
        (DealerAckStatus.FAILED, DealerErrorCode.DOUBLE_FEED),
        (DealerAckStatus.FAILED, DealerErrorCode.TRANSPORT_LOST),
        (DealerAckStatus.FAILED, DealerErrorCode.DECK_EMPTY),
        (DealerAckStatus.FAILED, DealerErrorCode.INTERLOCK_OPEN),
        (DealerAckStatus.FAILED, DealerErrorCode.EMERGENCY_STOP),
        (DealerAckStatus.FAILED, DealerErrorCode.PROTOCOL_ERROR),
    ],
)
def test_dealer_fault_matrix(
    status: DealerAckStatus, error_code: DealerErrorCode
) -> None:
    dealer = SimulatedDealer()
    dealer.inject_fault("fault", DealerFault(status, error_code, error_code.value))
    ack = dealer.execute(
        DealerCommand("fault", 1, DealerCommandType.DISPENSE_ONE)
    )
    assert ack.status is status
    assert ack.error_code is error_code
    assert dealer.dispensed_cards == 0


def test_restart_retains_dedupe_but_requires_home_for_new_motion() -> None:
    dealer = SimulatedDealer()
    dealer.execute(DealerCommand("home", 1, DealerCommandType.HOME))
    dealer.execute(
        DealerCommand(
            "rotate",
            2,
            DealerCommandType.ROTATE_TO,
            DealerTargetSlot.SEAT_A,
        )
    )
    command = DealerCommand("deal", 3, DealerCommandType.DISPENSE_ONE)
    original = dealer.execute(command)
    dealer.restart()
    assert dealer.execute(command) == original
    assert dealer.dispensed_cards == 1
    rejected = dealer.execute(
        DealerCommand(
            "new-rotate",
            4,
            DealerCommandType.ROTATE_TO,
            DealerTargetSlot.SEAT_B,
        )
    )
    assert rejected.status is DealerAckStatus.REJECTED
    assert rejected.error_code is DealerErrorCode.NOT_HOMED


def test_card_simulator_preserves_unknown_and_rejects_duplicates() -> None:
    simulator = SimulatedCardPerception()
    simulator.emit(VisionSlot.BOARD_FLOP_1, ObservationStatus.UNKNOWN)
    assert simulator.confirmed_cards() == {}
    ace = CardIdentity(Rank.ACE, Suit.SPADES)
    simulator.emit(VisionSlot.BOARD_FLOP_1, ObservationStatus.CONFIRMED, card=ace, confidence=0.99, observed_at_ns=2)
    simulator.emit(VisionSlot.SEAT_A_HOLE_1, ObservationStatus.CONFIRMED, card=ace, confidence=0.99, observed_at_ns=3)
    with pytest.raises(ValueError, match="duplicate"):
        simulator.confirmed_cards()


def test_action_simulator_carries_focus_and_state_version() -> None:
    engine = HandEngine.start("hand", Seat.A)
    observation = SimulatedActionPerception().emit(
        engine.state,
        ActionEvidenceState.CANDIDATE,
        action=PlayerActionType.CALL,
        confidence=0.99,
    )
    assert observation.focus_seat is Seat.D
    assert observation.expected_state_version == 0
