from __future__ import annotations

from typing import Any, Mapping

from poker_dealer.domain import (
    DealerTargetSlot,
    NavigationAction,
    NavigationAckStatus,
    NavigationCommand,
    NavigationErrorCode,
    RobotPoseNode,
)
from poker_dealer.robotics.navigation import (
    CocinoCarClient,
    CocinoCarNavigationAdapter,
    FaceCenterSample,
)


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Mapping[str, object] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((method, path, payload))
        if path.endswith("/capabilities"):
            return {
                "ok": True,
                "capabilities": {
                    "api_version": "1.0",
                    "available": True,
                    "request_terminal_status": True,
                    "actions": [
                        "follow_line_to_end",
                        "face_turn_start",
                        "face_turn_heartbeat",
                        "face_turn_stop",
                        "line_recenter_start",
                        "line_recenter_stop",
                        "preset_turn",
                        "dispense_one",
                        "stop",
                    ],
                },
            }
        if path.endswith("/status"):
            return {
                "ok": True,
                "status": {
                    "api_version": "1.0",
                    "available": True,
                    "gate_enabled": True,
                    "route": {"state": "MANUAL_COMPLETE"},
                    "robot": {},
                },
            }
        assert payload is not None
        action = str(payload["action"])
        return {
            "ok": True,
            "result": {
                "api_version": "1.0",
                "request_id": payload["request_id"],
                "action": action,
                "accepted": True,
                "state": "accepted",
            },
        }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.route_state = "MANUAL_COMPLETE"
        self.requests: dict[str, str] = {}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "api_version": "1.0",
            "available": True,
            "request_terminal_status": True,
            "actions": [
                "follow_line_to_end",
                "face_turn_start",
                "face_turn_heartbeat",
                "face_turn_stop",
                "line_recenter_start",
                "line_recenter_stop",
                "preset_turn",
                "stop",
            ],
        }

    def status(self) -> Mapping[str, Any]:
        return {
            "api_version": "1.0",
            "available": True,
            "gate_enabled": True,
            "route": {"state": self.route_state},
            "robot": {},
        }

    def action(
        self, request_id: str, action: str, **parameters: object
    ) -> Mapping[str, Any]:
        self.calls.append((request_id, action, dict(parameters)))
        self.requests[request_id] = action
        self.route_state = {
            "follow_line_to_end": "END_REACHED",
            "preset_turn": "MANUAL_COMPLETE",
            "line_recenter_start": "LINE_TURN_CENTERED",
            "face_turn_start": "FACE_CENTER_TURN",
            "face_turn_stop": "FACE_CENTERED_STOP",
            "stop": "MANUAL_COMPLETE",
        }.get(action, self.route_state)
        return {
            "api_version": "1.0",
            "request_id": request_id,
            "action": action,
            "accepted": True,
        }

    def request_result(self, request_id: str) -> Mapping[str, Any]:
        action = self.requests[request_id]
        return {
            "api_version": "1.0",
            "request_id": request_id,
            "action": action,
            "accepted": True,
            "request_status": "succeeded",
            "terminal": True,
        }


class CenteredFaceProbe:
    def observe_face_center(self) -> FaceCenterSample:
        return FaceCenterSample(True, True, 3, 2.0)


def _command(
    start: RobotPoseNode,
    target: RobotPoseNode,
    *,
    command_id: str = "nav-1",
    pose_version: int = 0,
) -> NavigationCommand:
    return NavigationCommand(
        command_id=command_id,
        session_id="session",
        hand_id="hand",
        expected_state_version=4,
        expected_pose_version=pose_version,
        issued_at_ns=1,
        action=NavigationAction.MOVE_AND_ALIGN_TO_TARGET,
        start_pose=start,
        target_pose=target,
        target_slot=DealerTargetSlot.SEAT_A,
        timeout_ms=5000,
        inter_motion_delay_ms=2500,
    )


def test_http_client_uses_formal_v1_paths_and_correlation() -> None:
    transport = FakeTransport()
    client = CocinoCarClient(transport)
    assert client.capabilities()["api_version"] == "1.0"
    result = client.action("request-1", "preset_turn", direction="RIGHT", degrees=90)
    assert result["request_id"] == "request-1"
    assert transport.calls[-1] == (
        "POST",
        "/api/robotics/v1/actions",
        {
            "request_id": "request-1",
            "action": "preset_turn",
            "direction": "RIGHT",
            "degrees": 90,
        },
    )


def test_adapter_expands_required_180_into_two_right_90_calls_with_gap() -> None:
    client = FakeClient()
    clock_value = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return clock_value[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock_value[0] += seconds

    adapter = CocinoCarNavigationAdapter(
        client,  # type: ignore[arg-type]
        initial_pose=RobotPoseNode.END_SMALL_BLIND,
        face_probe=CenteredFaceProbe(),
        clock=clock,
        sleeper=sleep,
    )
    adapter.open()
    ack = adapter.execute(
        _command(
            RobotPoseNode.END_SMALL_BLIND,
            RobotPoseNode.END_BIG_BLIND,
        )
    )
    assert ack.status is NavigationAckStatus.SUCCEEDED
    presets = [call for call in client.calls if call[1] == "preset_turn"]
    assert [call[2] for call in presets] == [
        {"direction": "RIGHT", "degrees": 90},
        {"direction": "RIGHT", "degrees": 90},
    ]
    assert any(seconds >= 2.5 for seconds in sleeps)


def test_adapter_runs_face_start_and_stop_only_after_center_evidence() -> None:
    client = FakeClient()
    adapter = CocinoCarNavigationAdapter(
        client,  # type: ignore[arg-type]
        initial_pose=RobotPoseNode.INIT_TO_END,
        face_probe=CenteredFaceProbe(),
    )
    adapter.open()
    ack = adapter.execute(
        _command(RobotPoseNode.INIT_TO_END, RobotPoseNode.INIT_BUTTON)
    )
    assert ack.status is NavigationAckStatus.SUCCEEDED
    assert ack.face_center_error_px == 2.0
    assert [call[1] for call in client.calls] == [
        "face_turn_start",
        "face_turn_stop",
    ]


def test_board_route_follows_for_one_second_then_issues_stop() -> None:
    client = FakeClient()
    sleeps: list[float] = []
    adapter = CocinoCarNavigationAdapter(
        client,  # type: ignore[arg-type]
        initial_pose=RobotPoseNode.INIT_TO_END,
        face_probe=CenteredFaceProbe(),
        sleeper=sleeps.append,
    )
    adapter.open()
    ack = adapter.execute(
        _command(RobotPoseNode.INIT_TO_END, RobotPoseNode.BOARD_TO_END)
    )
    assert ack.status is NavigationAckStatus.SUCCEEDED
    assert ack.actual_pose is RobotPoseNode.BOARD_TO_END
    assert ack.target_aligned
    assert [call[1] for call in client.calls] == [
        "follow_line_to_end",
        "stop",
    ]
    assert sleeps == [1.0]
