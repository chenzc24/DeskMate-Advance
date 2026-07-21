from __future__ import annotations

import numpy as np
import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    CardObservation,
    ColorSpace,
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerDeviceState,
    DealerErrorCode,
    DealerSensorEvidence,
    DealerTargetSlot,
    FramePacket,
    ObservationStatus,
    PlayerActionObservation,
    PlayerActionType,
    Rank,
    Suit,
    Seat,
    VisionSlot,
)


def test_action_observation_is_evidence_not_an_unscoped_action() -> None:
    observation = PlayerActionObservation(
        "action-obs-1",
        "hand-1",
        7,
        1_000,
        2_000,
        Seat.C,
        ActionEvidenceState.CANDIDATE,
        PlayerActionType.CALL,
        0.97,
        400,
        12,
        "action-test-v1",
        "roi-test-v1",
    )
    assert observation.focus_seat is Seat.C
    assert observation.candidate_action is PlayerActionType.CALL

    with pytest.raises(ValueError, match="requires an action"):
        PlayerActionObservation(
            "action-obs-2",
            "hand-1",
            7,
            1_000,
            2_000,
            Seat.C,
            ActionEvidenceState.CANDIDATE,
            None,
            0.97,
            400,
            12,
            "action-test-v1",
            "roi-test-v1",
        )

    with pytest.raises(ValueError, match="only candidate evidence"):
        PlayerActionObservation(
            "action-obs-3",
            "hand-1",
            7,
            1_000,
            2_000,
            Seat.C,
            ActionEvidenceState.AMBIGUOUS,
            PlayerActionType.CALL,
            0.40,
            200,
            6,
            "action-test-v1",
            "roi-test-v1",
        )


def test_frame_packet_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="declared dimensions"):
        FramePacket(0, 1, "table", 0, 3, 3, ColorSpace.BGR, 30.0, 0,
                    np.zeros((2, 3, 3), dtype=np.uint8))


def test_confirmed_card_observation_requires_identity() -> None:
    card = CardIdentity(Rank.ACE, Suit.SPADES)
    observation = CardObservation(
        "obs-1",
        VisionSlot.BOARD_FLOP_1,
        10,
        ObservationStatus.CONFIRMED,
        card,
        0.99,
        "test-v1",
        "cal-v1",
        3,
    )
    assert observation.card == card
    with pytest.raises(ValueError, match="require a card"):
        CardObservation(
            "obs-2",
            VisionSlot.BOARD_FLOP_1,
            10,
            ObservationStatus.CONFIRMED,
            None,
            0.99,
            "test-v1",
            "cal-v1",
            3,
        )
    with pytest.raises(ValueError, match="require confidence"):
        CardObservation(
            "obs-3",
            VisionSlot.BOARD_FLOP_1,
            10,
            ObservationStatus.CONFIRMED,
            card,
            None,
            "test-v1",
            "cal-v1",
            3,
        )


def test_rotate_command_requires_target_and_ack_failure_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires target_slot"):
        DealerCommand("cmd-1", 1, DealerCommandType.ROTATE_TO)
    command = DealerCommand(
        "cmd-2", 1, DealerCommandType.ROTATE_TO, DealerTargetSlot.BOARD_FLOP_1
    )
    assert command.target_slot is DealerTargetSlot.BOARD_FLOP_1
    with pytest.raises(ValueError, match="require error_code and reason"):
        DealerAck(
            "cmd-1",
            DealerCommandType.DISPENSE_ONE,
            None,
            DealerAckStatus.FAILED,
            10,
            DealerDeviceState.FAULT,
            2,
            DealerSensorEvidence(True, True, True, 0, True, False),
        )
    ack = DealerAck(
        "cmd-1",
        DealerCommandType.DISPENSE_ONE,
        None,
        DealerAckStatus.FAILED,
        10,
        DealerDeviceState.FAULT,
        2,
        DealerSensorEvidence(True, True, True, 0, True, False),
        DealerErrorCode.FEED_JAM,
        "exit sensor did not clear",
    )
    assert ack.error_code is DealerErrorCode.FEED_JAM


def test_successful_dispense_requires_physical_sensor_evidence() -> None:
    with pytest.raises(ValueError, match="one exit pulse"):
        DealerAck(
            "cmd-3",
            DealerCommandType.DISPENSE_ONE,
            None,
            DealerAckStatus.SUCCEEDED,
            20,
            DealerDeviceState.READY,
            3,
            DealerSensorEvidence(True, True, True, 0, True, False),
        )
    ack = DealerAck(
        "cmd-3",
        DealerCommandType.DISPENSE_ONE,
        None,
        DealerAckStatus.SUCCEEDED,
        20,
        DealerDeviceState.READY,
        3,
        DealerSensorEvidence(True, True, True, 1, True, False),
    )
    assert ack.status is DealerAckStatus.SUCCEEDED
