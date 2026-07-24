"""Private mobile web console for shared registration video and controls."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit
import uuid

import numpy as np

from poker_dealer.domain import ControlIntent, ControlObservation, HandPhase, SEAT_ORDER
from poker_dealer.domain.controls import ControlSource as ControlSourceKind
from poker_dealer.game import state_to_dict

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover
    WSMsgType = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]


_ASSET_ROOT = Path(__file__).with_name("mobile_web_assets")
_SUPPORTED_INTENTS = {
    ControlIntent.CONFIRM,
    ControlIntent.CANCEL,
    ControlIntent.START,
    ControlIntent.CLEAR,
    ControlIntent.NEXT_OPTION,
    ControlIntent.PREVIOUS_OPTION,
}


class CompositeControlSource:
    """Poll multiple bounded semantic control sources without changing priority."""

    def __init__(self, *sources: object) -> None:
        self.sources = sources

    def poll_controls(self, observed_at_ns: int) -> tuple[ControlObservation, ...]:
        observations: list[ControlObservation] = []
        for source in self.sources:
            observations.extend(source.poll_controls(observed_at_ns))
        return tuple(observations)


class CompositeRuntimeEventSink:
    """Copy one runtime event to audit and live-view sinks."""

    def __init__(self, *sinks: object) -> None:
        self.sinks = sinks

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        for sink in self.sinks:
            sink.emit(kind, observed_at_ns=observed_at_ns, payload=payload)


class MobilePromptMirror:
    """Send the same rendered announcement to Windows audio and the phone UI."""

    def __init__(self, primary: object, console: MobileWebConsole) -> None:
        self.primary = primary
        self.console = console

    def announce(self, announcement: object) -> None:
        self.primary.announce(announcement)
        text = str(getattr(announcement, "text", "")).strip()
        if text:
            self.console.publish_prompt(text)


class MobileWebConsole:
    """Run an aiohttp console in a thread beside the synchronous camera loop."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        queue_limit: int = 32,
        jpeg_quality: int = 78,
        frame_rate: float = 10.0,
    ) -> None:
        if not host.strip() or not 0 <= port <= 65535:
            raise ValueError("web console host/port is invalid")
        if queue_limit <= 0 or not 30 <= jpeg_quality <= 95 or frame_rate <= 0:
            raise ValueError("web console bounds are invalid")
        self.host = host
        self.port = port
        self.queue_limit = queue_limit
        self.jpeg_quality = jpeg_quality
        self.frame_interval_ns = int(1_000_000_000 / frame_rate)
        self._lock = threading.RLock()
        self._controls: queue.Queue[ControlObservation] = queue.Queue(queue_limit)
        self._processed: OrderedDict[str, Mapping[str, object]] = OrderedDict()
        self._sequence = 0
        self._view_version = 0
        self._state: dict[str, object] = {
            "view": "registration",
            "phase": "starting",
            "role": "button",
            "seat": "",
            "completed_roles": [],
            "face_samples": 0,
            "face_target": 0,
            "voice_samples": 0,
            "voice_target": 0,
            "voice_active": False,
            "prompt_playing": False,
            "speech_enabled": False,
            "alert_title": None,
            "alert_detail": None,
            "microphone_live": False,
            "microphone_level": 0.0,
            "microphone_callback_blocks": 0,
        }
        self._face_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self._face_status: str | None = None
        self._frame_size = (0, 0)
        self._jpeg: bytes | None = None
        self._frame_version = 0
        self._last_encoded_at_ns: int | None = None
        self._quit_requested = False
        self._runtime_feedback: str | None = None
        self._last_prompt: str | None = None
        self._prompt_callback: Callable[[Mapping[str, object]], None] | None = None
        self._controller_id: str | None = None
        self._clients: dict[str, object] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}/"

    def start(self, *, timeout_seconds: float = 10.0) -> None:
        if web is None:
            raise RuntimeError(
                "mobile web console requires the optional 'web-console' dependency"
            )
        if self._thread is not None:
            raise RuntimeError("mobile web console is already started")
        self._thread = threading.Thread(
            target=self._thread_main,
            name="poker-dealer-mobile-web",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_seconds):
            raise RuntimeError("mobile web console startup timed out")
        if self._startup_error is not None:
            raise RuntimeError("mobile web console failed to start") from self._startup_error

    def stop(self, *, timeout_seconds: float = 10.0) -> None:
        loop = self._loop
        stop = self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)
        self._thread = None

    def publish_registration_status(self, state: object) -> None:
        if is_dataclass(state):
            value = asdict(state)
        elif isinstance(state, Mapping):
            value = dict(state)
        else:
            raise TypeError("registration state must be a dataclass or mapping")
        completed = value.get("completed_roles", ())
        value["completed_roles"] = list(completed) if isinstance(completed, Sequence) else []
        value["view"] = "registration"
        self._replace_state(value)

    def publish_hand_state(self, runtime: object) -> None:
        """Mirror authoritative hand state without granting the UI authority."""

        engine = getattr(runtime, "engine")
        value = state_to_dict(engine.state)
        phase = engine.state.phase
        if phase not in {HandPhase.SHOWDOWN, HandPhase.SETTLED}:
            value["hole_cards"] = {}
        value.update(
            {
                "view": "hand",
                "session_id": getattr(runtime, "session_id"),
                "part_a_phase": (
                    runtime.part_a.phase.value if runtime.part_a is not None else None
                ),
                "part_b_phase": (
                    runtime.part_b.phase.value if runtime.part_b is not None else None
                ),
                "part_b_mode": (
                    runtime.part_b.mode.value if runtime.part_b is not None else None
                ),
                "current_target": (
                    runtime.part_b.current_step.target.value
                    if runtime.part_b is not None
                    and runtime.part_b.current_step is not None
                    else (
                        runtime.part_a.focus_seat.value
                        if runtime.part_a is not None
                        and runtime.part_a.focus_seat is not None
                        else None
                    )
                ),
                "players_by_seat": {
                    seat.value: runtime.expected_player_by_seat.get(seat, "")
                    for seat in SEAT_ORDER
                },
            }
        )
        self._replace_state(value)

    def publish_session_state(
        self,
        session: object,
        *,
        last_reason: str = "waiting_for_operator",
        stop_after_clear: bool = False,
        selected_seat: object | None = None,
        selected_slot: object | None = None,
    ) -> None:
        runtime = session.active_hand
        value: dict[str, object] = {
            "view": "session_boundary",
            "phase": (
                "recovery"
                if runtime is not None
                else "session_ended"
                if session.ended
                else "table_clearance"
                if not session.table_cleared
                else "ready_next_hand"
            ),
            "session_id": session.roster.session_id,
            "button": session.button.value,
            "next_hand_number": session.next_hand_number,
            "stacks": {
                seat.value: session.stacks[seat] for seat in SEAT_ORDER
            },
            "low_stack_seats": [seat.value for seat in session.low_stack_seats],
            "table_cleared": session.table_cleared,
            "ended": session.ended,
            "last_reason": last_reason,
            "stop_after_clear": stop_after_clear,
            "selected_seat": (
                getattr(selected_seat, "value", None)
                if selected_seat is not None
                else None
            ),
            "selected_slot": (
                getattr(selected_slot, "value", None)
                if selected_slot is not None
                else None
            ),
            "paused_reason": (
                runtime.engine.state.paused_reason if runtime is not None else None
            ),
        }
        self._replace_state(value)

    def _replace_state(self, value: Mapping[str, object]) -> None:
        value = dict(value)
        with self._lock:
            if value == self._state:
                return
            previous_phase = str(self._state.get("phase", ""))
            previous_control_state = self._control_state_signature(self._state)
            next_control_state = self._control_state_signature(value)
            self._state = value
            if next_control_state != previous_control_state:
                self._view_version += 1
            should_prompt = (
                self._controller_id is not None
                and previous_phase == "starting"
                and str(value.get("phase", "")) != "starting"
            )
        self._notify_changed()
        if should_prompt:
            self._request_prompt()

    @staticmethod
    def _control_state_signature(state: Mapping[str, object]) -> tuple[object, ...]:
        return (
            state.get("view"),
            state.get("phase"),
            state.get("state_version"),
            state.get("acting_seat"),
            state.get("current_target"),
            state.get("role"),
            state.get("seat"),
            state.get("table_cleared"),
            state.get("last_reason"),
            state.get("selected_seat"),
            state.get("selected_slot"),
            bool(state.get("voice_active", False)),
        )

    def publish_face_detections(
        self,
        boxes: tuple[tuple[int, int, int, int], ...],
        *,
        status: str | None,
    ) -> None:
        with self._lock:
            if boxes == self._face_boxes and status == self._face_status:
                return
            self._face_boxes = boxes
            self._face_status = status
        self._notify_changed()

    def publish_frame(self, image: np.ndarray, *, observed_at_ns: int) -> None:
        if cv2 is None:
            return
        with self._lock:
            last = self._last_encoded_at_ns
            if last is not None and observed_at_ns - last < self.frame_interval_ns:
                return
            self._last_encoded_at_ns = observed_at_ns
        height, width = image.shape[:2]
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not encoded:
            return
        with self._lock:
            self._jpeg = bytes(buffer)
            self._frame_size = (width, height)
            self._frame_version += 1
        self._notify_changed()

    def request_quit(self) -> None:
        with self._lock:
            self._quit_requested = True
        self._notify_changed()

    def set_prompt_callback(
        self,
        callback: Callable[[Mapping[str, object]], None],
    ) -> None:
        with self._lock:
            self._prompt_callback = callback
            controller_connected = self._controller_id is not None
        if controller_connected:
            self._request_prompt()

    def publish_prompt(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._last_prompt = text.strip()
        self._broadcast({"type": "prompt", "text": self._last_prompt})

    def consume_quit_request(self) -> bool:
        with self._lock:
            if not self._quit_requested:
                return False
            self._quit_requested = False
            return True

    def poll_controls(self, observed_at_ns: int) -> tuple[ControlObservation, ...]:
        del observed_at_ns
        observations: list[ControlObservation] = []
        while True:
            try:
                observations.append(self._controls.get_nowait())
            except queue.Empty:
                return tuple(observations)

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        feedback: str | None = None
        if kind == "voice_enrollment_sample_rejected":
            feedback = (
                "Voice sample was too short. Repeat the current word, then pause."
            )
        elif kind == "voice_enrollment_sample_accepted":
            feedback = (
                f"Voice sample {payload.get('sample_number', 0)} accepted."
            )
        elif kind == "speaker_enrollment_completed":
            feedback = "Voice enrollment complete."
        elif kind == "audio_link_lost":
            feedback = "AudioRelay microphone disconnected."
        elif kind == "audio_link_restored":
            feedback = "AudioRelay microphone reconnected."
        if feedback is not None:
            with self._lock:
                self._runtime_feedback = feedback
            self._notify_changed()
        if kind not in {"registration_control", "runtime_control"}:
            return
        observation_id = str(payload.get("observation_id", ""))
        message = {
            "type": "control_result",
            "command_id": observation_id.removeprefix("web-control:"),
            "observed_at_ns": observed_at_ns,
            "accepted": (
                bool(payload.get("accepted", False))
                if kind == "registration_control"
                else True
            ),
            "reason": str(
                payload.get(
                    "reason",
                    "delivered_to_runtime" if kind == "runtime_control" else "",
                )
            ),
        }
        self._broadcast(message)

    def snapshot(self, *, controller: bool | None = None) -> dict[str, object]:
        with self._lock:
            width, height = self._frame_size
            state = dict(self._state)
            boxes = [
                {
                    "x": x / width if width else 0.0,
                    "y": y / height if height else 0.0,
                    "width": box_width / width if width else 0.0,
                    "height": box_height / height if height else 0.0,
                }
                for x, y, box_width, box_height in self._face_boxes
            ]
            result: dict[str, object] = {
                "type": "state",
                "view_version": self._view_version,
                "state": state,
                "face_boxes": boxes,
                "face_status": self._face_status,
                "video_ready": self._jpeg is not None,
                "runtime_feedback": self._runtime_feedback,
                "last_prompt": self._last_prompt,
                "allowed_intents": [
                    intent.value for intent in self._allowed_intents_locked()
                ],
            }
            if controller is not None:
                result["controller"] = controller
            return result

    def submit_command(
        self,
        *,
        client_id: str,
        command_id: str,
        intent: str,
        expected_view_version: int,
    ) -> Mapping[str, object]:
        if not command_id.strip() or len(command_id) > 128:
            return self._command_ack(command_id, "rejected", "invalid_command_id")
        with self._lock:
            previous = self._processed.get(command_id)
            if previous is not None:
                return previous
            if client_id != self._controller_id:
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "viewer_read_only"),
                )
            try:
                parsed_intent = ControlIntent(intent)
            except ValueError:
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "unsupported_intent"),
                )
            if parsed_intent not in _SUPPORTED_INTENTS:
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "unsupported_intent"),
                )
            if expected_view_version != self._view_version:
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "stale_view"),
                )
            if parsed_intent not in self._allowed_intents_locked():
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "intent_not_available"),
                )
            self._sequence += 1
            observation = ControlObservation(
                observation_id=f"web-control:{command_id}",
                intent=parsed_intent,
                source=ControlSourceKind.WEB_CONSOLE,
                observed_at_ns=time.monotonic_ns(),
                control_id=f"mobile-web:{client_id}",
                device_state_version=self._sequence,
            )
            try:
                self._controls.put_nowait(observation)
            except queue.Full:
                return self._remember(
                    command_id,
                    self._command_ack(command_id, "rejected", "control_queue_full"),
                )
            return self._remember(
                command_id,
                self._command_ack(command_id, "queued", "awaiting_runtime"),
            )

    def _allowed_intents_locked(self) -> tuple[ControlIntent, ...]:
        view = str(self._state.get("view", "registration"))
        phase = str(self._state.get("phase", ""))
        if view == "hand":
            if phase == HandPhase.AWAITING_ACTION.value:
                return (ControlIntent.CONFIRM, ControlIntent.CANCEL)
            return ()
        if view == "session_boundary":
            if phase == "recovery":
                return (
                    ControlIntent.START,
                    ControlIntent.CLEAR,
                    ControlIntent.CONFIRM,
                    ControlIntent.NEXT_OPTION,
                    ControlIntent.PREVIOUS_OPTION,
                )
            if phase == "table_clearance":
                return (ControlIntent.CONFIRM,)
            if phase == "ready_next_hand":
                return (
                    ControlIntent.START,
                    ControlIntent.CLEAR,
                    ControlIntent.CONFIRM,
                    ControlIntent.NEXT_OPTION,
                    ControlIntent.PREVIOUS_OPTION,
                )
            return ()
        voice_active = bool(self._state.get("voice_active", False))
        allowed: list[ControlIntent] = []
        if phase == "ready_for_face" and not voice_active:
            allowed.append(ControlIntent.CONFIRM)
        if phase != "started":
            allowed.append(ControlIntent.CLEAR)
        if phase == "ready_to_start" and not voice_active:
            allowed.append(ControlIntent.START)
        return tuple(allowed)

    @staticmethod
    def _command_ack(
        command_id: str, status: str, reason: str
    ) -> dict[str, object]:
        return {
            "type": "command_ack",
            "command_id": command_id,
            "status": status,
            "reason": reason,
        }

    def _remember(
        self, command_id: str, ack: Mapping[str, object]
    ) -> Mapping[str, object]:
        self._processed[command_id] = ack
        while len(self._processed) > 256:
            self._processed.popitem(last=False)
        return ack

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()

    async def _serve(self) -> None:
        assert web is not None
        self._loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()
        self._stop = asyncio.Event()

        async def security_headers(request: object, response: object) -> None:
            del request
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; "
                "script-src 'self'; style-src 'self'; connect-src 'self' ws: wss:; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            )
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"

        app = web.Application(client_max_size=4096)
        app.on_response_prepare.append(security_headers)
        app.router.add_get("/", self._index)
        app.router.add_get("/assets/{name}", self._asset)
        app.router.add_get("/healthz", self._health)
        app.router.add_get("/video.mjpeg", self._video)
        app.router.add_get("/ws", self._websocket)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
            server = getattr(site, "_server", None)
            sockets = getattr(server, "sockets", ()) if server is not None else ()
            if sockets:
                self.port = int(sockets[0].getsockname()[1])
            self._ready.set()
            await self._stop.wait()
        finally:
            await runner.cleanup()

    async def _index(self, request: object):
        del request
        return web.FileResponse(_ASSET_ROOT / "index.html")

    async def _asset(self, request: object):
        name = request.match_info["name"]
        if name not in {"app.js", "styles.css"}:
            raise web.HTTPNotFound()
        return web.FileResponse(_ASSET_ROOT / name)

    async def _health(self, request: object):
        del request
        return web.json_response(
            {
                "status": "ok",
                "service": "poker-dealer-mobile-web-console",
                "frames_saved": False,
                "audio_requested": False,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _video(self, request: object):
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Content-Type-Options": "nosniff",
            },
        )
        await response.prepare(request)
        last_version = -1
        try:
            while self._stop is not None and not self._stop.is_set():
                with self._lock:
                    version = self._frame_version
                    jpeg = self._jpeg
                if jpeg is not None and version != last_version:
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    last_version = version
                await asyncio.sleep(0.05)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response

    async def _websocket(self, request: object):
        origin = request.headers.get("Origin")
        if origin:
            origin_host = urlsplit(origin).netloc.casefold()
            request_host = request.host.casefold()
            if origin_host != request_host:
                raise web.HTTPForbidden(reason="cross-origin websocket denied")
        websocket = web.WebSocketResponse(
            heartbeat=20.0,
            max_msg_size=4096,
            compress=False,
        )
        await websocket.prepare(request)
        client_id = uuid.uuid4().hex
        with self._lock:
            is_controller = self._controller_id is None
            if is_controller:
                self._controller_id = client_id
            self._clients[client_id] = websocket
        await websocket.send_json(
            {
                "type": "hello",
                "client_id": client_id,
                "controller": is_controller,
            }
        )
        await websocket.send_json(self.snapshot(controller=is_controller))
        if is_controller:
            self._request_prompt()
        try:
            async for message in websocket:
                if message.type is WSMsgType.TEXT:
                    await self._handle_ws_message(client_id, websocket, message.data)
                elif message.type in {
                    WSMsgType.ERROR,
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                }:
                    break
        finally:
            next_controller: tuple[str, object] | None = None
            with self._lock:
                self._clients.pop(client_id, None)
                if self._controller_id == client_id:
                    self._controller_id = next(iter(self._clients), None)
                    if self._controller_id is not None:
                        next_controller = (
                            self._controller_id,
                            self._clients[self._controller_id],
                        )
            if next_controller is not None:
                await next_controller[1].send_json(
                    {"type": "controller_changed", "controller": True}
                )
        return websocket

    async def _handle_ws_message(
        self, client_id: str, websocket: object, raw: str
    ) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json(
                self._command_ack("", "rejected", "invalid_json")
            )
            return
        if not isinstance(message, dict) or message.get("type") != "command":
            await websocket.send_json(
                self._command_ack("", "rejected", "invalid_message")
            )
            return
        intent = str(message.get("intent", ""))
        command_id = str(message.get("command_id", ""))
        expected = message.get("expected_view_version")
        if intent == "quit":
            with self._lock:
                allowed = client_id == self._controller_id
            ack = self._command_ack(
                command_id,
                "accepted" if allowed else "rejected",
                "operator_exit_requested" if allowed else "viewer_read_only",
            )
            if allowed:
                self.request_quit()
            await websocket.send_json(ack)
            return
        if intent == "repeat_prompt":
            with self._lock:
                allowed = client_id == self._controller_id
            ack = self._command_ack(
                command_id,
                "accepted" if allowed else "rejected",
                "prompt_replayed" if allowed else "viewer_read_only",
            )
            if allowed:
                self._request_prompt()
            await websocket.send_json(ack)
            return
        if not isinstance(expected, int) or isinstance(expected, bool):
            await websocket.send_json(
                self._command_ack(
                    command_id, "rejected", "invalid_view_version"
                )
            )
            return
        ack = self.submit_command(
            client_id=client_id,
            command_id=command_id,
            intent=intent,
            expected_view_version=expected,
        )
        await websocket.send_json(ack)
        if ack.get("reason") == "stale_view":
            await websocket.send_json(self.snapshot(controller=True))

    def _notify_changed(self) -> None:
        loop = self._loop
        changed = self._changed
        if loop is not None and changed is not None:
            loop.call_soon_threadsafe(changed.set)
        self._broadcast(self.snapshot())

    def _request_prompt(self) -> None:
        with self._lock:
            callback = self._prompt_callback
            snapshot = self.snapshot(controller=True)
        if callback is not None:
            try:
                callback(snapshot)
            except Exception:
                return

    def _broadcast(self, payload: Mapping[str, object]) -> None:
        loop = self._loop
        if loop is None:
            return

        async def send() -> None:
            with self._lock:
                clients = list(self._clients.items())
                controller_id = self._controller_id
            for client_id, websocket in clients:
                message = dict(payload)
                if message.get("type") == "state":
                    message["controller"] = client_id == controller_id
                try:
                    await websocket.send_json(message)
                except (ConnectionResetError, RuntimeError):
                    continue

        asyncio.run_coroutine_threadsafe(send(), loop)


__all__ = [
    "CompositeControlSource",
    "CompositeRuntimeEventSink",
    "MobilePromptMirror",
    "MobileWebConsole",
]
