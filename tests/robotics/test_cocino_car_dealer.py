from __future__ import annotations

from typing import Any, Mapping

from poker_dealer.domain import (
    DealerAckStatus,
    DealerCompletionBasis,
    DealerCommand,
    DealerCommandType,
    DealerTargetSlot,
    NavigationAck,
    NavigationAckStatus,
    NavigationAction,
    RobotPoseNode,
)
from poker_dealer.robotics.dealer import CocinoCarDealerAdapter


class CommandAckV1Client:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "api_version": "1.0",
            "available": True,
            "actions": ["dispense_one", "stop"],
            "physical_card_exit_verified": False,
            "dispense_completion_evidence": "arduino_command_ack_only",
            "request_terminal_status": True,
        }

    def action(self, request_id: str, action: str, **parameters: object):
        del request_id, parameters
        self.actions.append(action)
        return {"action": action, "accepted": True}

    def request_result(self, request_id: str):
        return {
            "request_id": request_id,
            "action": "dispense_one",
            "request_status": "succeeded",
            "terminal": True,
        }


def _navigation_ack() -> NavigationAck:
    return NavigationAck(
        command_id="navigation",
        session_id="session",
        hand_id="hand",
        expected_state_version=1,
        action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
        target_slot=DealerTargetSlot.SEAT_A,
        status=NavigationAckStatus.SUCCEEDED,
        observed_at_ns=1,
        actual_pose=RobotPoseNode.INIT_TO_END,
        pose_version=1,
        pose_confidence=1.0,
        line_locked=True,
        endpoint_confirmed=True,
        target_aligned=True,
        stable_frames=3,
    )


def test_command_ack_v1_dispense_advances_without_faking_sensor_evidence() -> None:
    client = CommandAckV1Client()
    adapter = CocinoCarDealerAdapter(client)  # type: ignore[arg-type]
    adapter.open()
    adapter.confirm_navigation_target(_navigation_ack())
    ack = adapter.execute(
        DealerCommand(
            command_id="dispense",
            issued_at_ns=1,
            command=DealerCommandType.DISPENSE_ONE,
        ),
        observed_at_ns=2,
    )
    assert ack.status is DealerAckStatus.SUCCEEDED
    assert (
        ack.completion_basis
        is DealerCompletionBasis.ARDUINO_COMMAND_ACK_ONLY
    )
    assert ack.sensor_evidence.at_target is True
    assert ack.sensor_evidence.deck_present is None
    assert ack.sensor_evidence.exit_pulses is None
    assert client.actions == ["dispense_one"]
    assert adapter.health().available
