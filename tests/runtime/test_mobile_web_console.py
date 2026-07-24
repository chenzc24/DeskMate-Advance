from __future__ import annotations

import asyncio
from dataclasses import replace

from aiohttp import ClientSession
import numpy as np

from poker_dealer.domain import ControlIntent, ControlSource, Seat
from poker_dealer.runtime import HandRuntime
from poker_dealer.runtime.live_perception import RegistrationUiState
from poker_dealer.runtime.mobile_web_console import (
    CompositeControlSource,
    CompositeRuntimeEventSink,
    MobilePromptMirror,
    MobileWebConsole,
)


def _state(*, phase: str = "ready_for_face") -> RegistrationUiState:
    return RegistrationUiState(
        phase=phase,
        role="button",
        seat="seat_a",
        completed_roles=(),
        face_samples=0,
        face_target=5,
        voice_samples=0,
        voice_target=3,
        voice_active=False,
        prompt_playing=False,
        speech_enabled=True,
        alert_title=None,
        alert_detail=None,
    )


def test_semantic_commands_are_versioned_bounded_and_idempotent() -> None:
    console = MobileWebConsole(queue_limit=1)
    console._controller_id = "phone"
    console.publish_registration_status(_state())
    version = int(console.snapshot()["view_version"])

    first = console.submit_command(
        client_id="phone",
        command_id="one",
        intent="confirm",
        expected_view_version=version,
    )
    duplicate = console.submit_command(
        client_id="phone",
        command_id="one",
        intent="confirm",
        expected_view_version=version,
    )
    full = console.submit_command(
        client_id="phone",
        command_id="two",
        intent="clear",
        expected_view_version=version,
    )

    assert first == duplicate
    assert first["status"] == "queued"
    assert full["reason"] == "control_queue_full"
    observations = console.poll_controls(0)
    assert len(observations) == 1
    assert observations[0].intent is ControlIntent.CONFIRM
    assert observations[0].source is ControlSource.WEB_CONSOLE


def test_viewer_stale_and_unavailable_commands_fail_closed() -> None:
    console = MobileWebConsole()
    console._controller_id = "controller"
    console.publish_registration_status(_state(phase="capturing_face"))
    version = int(console.snapshot()["view_version"])

    viewer = console.submit_command(
        client_id="viewer",
        command_id="viewer-command",
        intent="clear",
        expected_view_version=version,
    )
    stale = console.submit_command(
        client_id="controller",
        command_id="stale-command",
        intent="clear",
        expected_view_version=version - 1,
    )
    unavailable = console.submit_command(
        client_id="controller",
        command_id="unavailable-command",
        intent="confirm",
        expected_view_version=version,
    )

    assert viewer["reason"] == "viewer_read_only"
    assert stale["reason"] == "stale_view"
    assert unavailable["reason"] == "intent_not_available"
    assert console.poll_controls(0) == ()


def test_state_contains_face_boxes_action_marker_and_memory_only_video() -> None:
    console = MobileWebConsole(frame_rate=1000)
    console.publish_registration_status(_state())
    console.publish_frame(
        np.zeros((100, 200, 3), dtype=np.uint8),
        observed_at_ns=1_000_000,
    )
    console.publish_face_detections(((20, 10, 40, 30),), status="FACE READY")
    console.publish_action_marker((0.35, 0.7, "call", 0.82))

    snapshot = console.snapshot()

    assert snapshot["video_ready"] is True
    assert snapshot["face_status"] == "FACE READY"
    assert snapshot["face_boxes"] == [
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.3}
    ]
    assert snapshot["action_marker"] == {
        "x": 0.35,
        "y": 0.7,
        "action": "call",
        "confidence": 0.82,
    }
    assert not hasattr(console, "recording_path")


def test_hand_action_stage_does_not_offer_e_confirmation() -> None:
    console = MobileWebConsole()
    console._controller_id = "phone"
    console._replace_state({"view": "hand", "phase": "awaiting_action"})
    snapshot = console.snapshot()

    assert snapshot["allowed_intents"] == ["cancel"]
    rejected = console.submit_command(
        client_id="phone",
        command_id="no-game-confirm",
        intent="confirm",
        expected_view_version=int(snapshot["view_version"]),
    )
    assert rejected["reason"] == "intent_not_available"


def test_audio_meter_updates_do_not_invalidate_operator_controls() -> None:
    console = MobileWebConsole()
    first = _state()
    console.publish_registration_status(first)
    version = console.snapshot()["view_version"]
    updated = replace(
        first,
        microphone_live=True,
        microphone_level=0.4,
        microphone_callback_blocks=20,
    )

    console.publish_registration_status(updated)

    assert console.snapshot()["view_version"] == version


def test_full_hand_state_exposes_ledger_without_private_cards() -> None:
    runtime = HandRuntime.new_hand(
        hand_id="ui-hand",
        session_id="ui-session",
        button=Seat.A,
        require_actor_binding=False,
        require_visual_settle=False,
        expected_player_by_seat={
            Seat.A: "participant_1",
            Seat.B: "participant_2",
            Seat.C: "participant_3",
            Seat.D: "participant_4",
        },
    )
    console = MobileWebConsole()

    console.publish_hand_state(runtime)
    state = console.snapshot()["state"]

    assert state["view"] == "hand"
    assert state["phase"] == "dealing_hole"
    assert state["pot_units"] == 3
    assert state["hole_cards"] == {}
    assert state["players_by_seat"]["seat_a"] == "participant_1"
    assert state["part_b_mode"] == "hole_deal"


def test_composite_adapters_preserve_controls_and_copy_events() -> None:
    class Source:
        def __init__(self, value: tuple[object, ...]) -> None:
            self.value = value

        def poll_controls(self, observed_at_ns: int) -> tuple[object, ...]:
            del observed_at_ns
            return self.value

    class Sink:
        def __init__(self) -> None:
            self.events: list[tuple[str, int, object]] = []

        def emit(self, kind: str, *, observed_at_ns: int, payload: object) -> None:
            self.events.append((kind, observed_at_ns, payload))

    assert CompositeControlSource(Source(("a",)), Source(("b",))).poll_controls(1) == (
        "a",
        "b",
    )
    first, second = Sink(), Sink()
    CompositeRuntimeEventSink(first, second).emit(
        "registration_control",
        observed_at_ns=9,
        payload={"accepted": True},
    )
    assert first.events == second.events


def test_http_and_websocket_console_round_trip() -> None:
    console = MobileWebConsole(port=0)
    console.start()

    async def exercise() -> None:
        async with ClientSession() as client:
            async with client.get(console.url + "healthz") as response:
                assert response.status == 200
                assert (await response.json())["audio_requested"] is False
            async with client.get(console.url) as response:
                assert response.status == 200
                assert "Player registration" in await response.text()
            async with client.get(console.url + "assets/app.js") as response:
                script = await response.text()
                assert "createCommandId()" in script
                assert "globalThis.crypto?.randomUUID" in script
                assert "runtimeFaceVisible" in script
                assert "VERIFIED · LISTENING" in script
                assert "Say one legal English action clearly" in script
                assert "message.action_marker" in script
            async with client.get(console.url + "assets/styles.css") as response:
                stylesheet = await response.text()
                assert response.status == 200
                assert "@media (orientation: landscape)" in stylesheet
                assert "grid-template-columns: minmax(0, 56fr)" in stylesheet
                assert ".face-box.verified" in stylesheet
                assert ".action-marker" in stylesheet
            websocket = await client.ws_connect(console.url + "ws")
            hello = await websocket.receive_json()
            state = await websocket.receive_json()
            assert hello["controller"] is True
            assert state["type"] == "state"
            await websocket.send_json(
                {
                    "type": "command",
                    "command_id": "round-trip",
                    "intent": "clear",
                    "expected_view_version": state["view_version"],
                }
            )
            acknowledgement = await websocket.receive_json()
            assert acknowledgement["status"] == "queued"
            await websocket.close()

    try:
        asyncio.run(exercise())
    finally:
        console.stop()


def test_runtime_result_is_correlated_to_web_command(monkeypatch) -> None:
    console = MobileWebConsole()
    messages: list[object] = []
    monkeypatch.setattr(console, "_broadcast", messages.append)

    console.emit(
        "registration_control",
        observed_at_ns=200,
        payload={
            "observation_id": "web-control:abc",
            "accepted": False,
            "reason": "four_roles_required",
        },
    )

    assert messages == [
        {
            "type": "control_result",
            "command_id": "abc",
            "observed_at_ns": 200,
            "accepted": False,
            "reason": "four_roles_required",
        }
    ]


def test_controller_connection_can_replay_the_current_prompt() -> None:
    console = MobileWebConsole()
    console._controller_id = "phone"
    console.publish_registration_status(_state())
    prompts: list[object] = []

    console.set_prompt_callback(prompts.append)

    assert len(prompts) == 1
    assert prompts[0]["state"]["phase"] == "ready_for_face"


def test_rendered_prompt_is_mirrored_without_changing_its_text(monkeypatch) -> None:
    class Primary:
        def __init__(self) -> None:
            self.items: list[object] = []

        def announce(self, announcement: object) -> None:
            self.items.append(announcement)

    class Announcement:
        text = "Button, press the E key."

    primary = Primary()
    console = MobileWebConsole()
    prompts: list[object] = []
    monkeypatch.setattr(console, "_broadcast", prompts.append)

    MobilePromptMirror(primary, console).announce(Announcement())

    assert len(primary.items) == 1
    assert prompts == [{"type": "prompt", "text": "Button, press the E key."}]
