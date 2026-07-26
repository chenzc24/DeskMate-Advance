"""Read-only integration client for cocino_car's robotics HTTP API v1.0."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from poker_dealer.domain import (
    NavigationAck,
    NavigationAckStatus,
    NavigationCommand,
    NavigationErrorCode,
    RobotPoseNode,
)

from .port import NavigationHealth, NavigationUnavailableError
from .table_route import (
    CarApiAction,
    RoutePrimitive,
    TableRoutePlanner,
    TurnDirection,
)


COCINO_CAR_API_VERSION = "1.0"
_REQUEST_ID_SAFE = re.compile(r"[^A-Za-z0-9._:-]+")
_ROUTE_FAILURE_STATES = {
    "FACE_TURN_HEARTBEAT_TIMEOUT",
    "FACE_TURN_SEARCH_TIMEOUT",
    "LINE_TURN_SEARCH_TIMEOUT",
    "MANUAL_ALIGNMENT_TIMEOUT",
}


class CocinoCarProtocolError(RuntimeError):
    """Raised when the remote facade violates its declared v1.0 contract."""


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """Small standard-library JSON transport with bounded requests."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 3.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive")
        base = base_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("cocino_car base_url must use http or https")
        self.base_url = base
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CocinoCarProtocolError(
                f"cocino_car HTTP {exc.code}: {detail[:300]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise NavigationUnavailableError(
                f"cocino_car transport unavailable: {type(exc).__name__}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CocinoCarProtocolError("cocino_car returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise CocinoCarProtocolError("cocino_car response must be an object")
        return decoded


class CocinoCarClient:
    """Typed access to `/api/robotics/v1/*`; it never enables the motor gate."""

    def __init__(self, transport: JsonTransport) -> None:
        self.transport = transport

    def capabilities(self) -> Mapping[str, Any]:
        body = self._ok(self.transport.request("GET", "/api/robotics/v1/capabilities"))
        value = body.get("capabilities")
        return self._versioned_object(value, "capabilities")

    def status(self) -> Mapping[str, Any]:
        body = self._ok(self.transport.request("GET", "/api/robotics/v1/status"))
        value = body.get("status")
        return self._versioned_object(value, "status")

    def action(
        self, request_id: str, action: str, **parameters: object
    ) -> Mapping[str, Any]:
        payload = {"request_id": request_id, "action": action, **parameters}
        try:
            response = self.transport.request(
                "POST", "/api/robotics/v1/actions", payload
            )
        except NavigationUnavailableError as original:
            # The POST may have reached the Pi. Resolve the same request ID
            # without reissuing physical motion.
            try:
                resolved = self.request_result(request_id)
            except (NavigationUnavailableError, CocinoCarProtocolError):
                raise original
            if resolved.get("action") != action:
                raise CocinoCarProtocolError(
                    "resolved action response correlation mismatch"
                )
            return resolved
        body = self._ok(response)
        result = body.get("result")
        value = self._versioned_object(result, "action result")
        if value.get("request_id") != request_id or value.get("action") != action:
            raise CocinoCarProtocolError("action response correlation mismatch")
        if value.get("accepted") is not True:
            raise CocinoCarProtocolError("action was not accepted")
        return value

    def request_result(self, request_id: str) -> Mapping[str, Any]:
        body = self._ok(
            self.transport.request(
                "GET", f"/api/robotics/v1/requests/{request_id}"
            )
        )
        result = body.get("result")
        value = self._versioned_object(result, "request result")
        if value.get("request_id") != request_id:
            raise CocinoCarProtocolError("request-result correlation mismatch")
        return value

    @staticmethod
    def _ok(body: Mapping[str, Any]) -> Mapping[str, Any]:
        if body.get("ok") is not True:
            raise CocinoCarProtocolError(str(body.get("error") or "request rejected"))
        return body

    @staticmethod
    def _versioned_object(value: object, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise CocinoCarProtocolError(f"{label} must be an object")
        if value.get("api_version") != COCINO_CAR_API_VERSION:
            raise CocinoCarProtocolError(
                f"{label} API version is not {COCINO_CAR_API_VERSION}"
            )
        return value


@dataclass(frozen=True, slots=True)
class FaceCenterSample:
    detected: bool
    centered: bool
    stable_frames: int
    center_error_px: float | None = None

    def __post_init__(self) -> None:
        if self.stable_frames < 0:
            raise ValueError("face stable_frames must be non-negative")
        if self.center_error_px is not None and self.center_error_px < 0:
            raise ValueError("face center error must be non-negative")
        if self.centered and (not self.detected or self.stable_frames <= 0):
            raise ValueError("centered face requires detected stable evidence")


class FaceCenterProbe(Protocol):
    def observe_face_center(self) -> FaceCenterSample: ...


class CocinoCarNavigationAdapter:
    """Execute one semantic navigation command through safe API primitives.

    The adapter tracks canonical pose locally because cocino_car v1.0 does not
    expose pose or pose-version evidence. A process restart therefore requires
    an operator-confirmed initial pose.
    """

    physical_motion = True

    def __init__(
        self,
        client: CocinoCarClient,
        *,
        device_id: str = "cocino-car-http-v1",
        initial_pose: RobotPoseNode,
        initial_pose_version: int = 0,
        face_probe: FaceCenterProbe | None = None,
        planner: TableRoutePlanner | None = None,
        poll_interval_seconds: float = 0.1,
        heartbeat_interval_seconds: float = 1.0,
        board_follow_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if initial_pose is RobotPoseNode.UNKNOWN:
            raise ValueError("physical navigation requires an operator-confirmed pose")
        if initial_pose_version < 0:
            raise ValueError("initial pose version must be non-negative")
        if (
            poll_interval_seconds <= 0
            or heartbeat_interval_seconds <= 0
            or board_follow_seconds <= 0
        ):
            raise ValueError(
                "poll, heartbeat and board-follow intervals must be positive"
            )
        self.client = client
        self._device_id = device_id
        self._pose = initial_pose
        self._pose_version = initial_pose_version
        self.face_probe = face_probe
        self.planner = planner or TableRoutePlanner()
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.board_follow_seconds = board_follow_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._opened = False
        self._reason: str | None = None
        self._acks: dict[str, NavigationAck] = {}
        self._last_motion_completed_at: float | None = None

    @property
    def device_id(self) -> str:
        return self._device_id

    def open(self) -> None:
        capabilities = self.client.capabilities()
        required = {
            "follow_line_to_end",
            "face_turn_start",
            "face_turn_heartbeat",
            "face_turn_stop",
            "line_recenter_start",
            "preset_turn",
            "stop",
        }
        actions = {
            str(item) for item in capabilities.get("actions", ()) if isinstance(item, str)
        }
        missing = sorted(required - actions)
        if (
            capabilities.get("available") is not True
            or capabilities.get("request_terminal_status") is not True
            or missing
        ):
            raise NavigationUnavailableError(
                "cocino_car robotics API unavailable"
                + (
                    "; request terminal status is unavailable"
                    if capabilities.get("request_terminal_status") is not True
                    else ""
                )
                + (f"; missing actions: {missing}" if missing else "")
            )
        status = self.client.status()
        if status.get("gate_enabled") is not True:
            raise NavigationUnavailableError(
                "cocino_car motor gate is disabled; operator must enable it"
            )
        self._opened = True
        self._reason = None

    def execute(
        self, command: NavigationCommand, observed_at_ns: int | None = None
    ) -> NavigationAck:
        if not self._opened:
            raise NavigationUnavailableError(self._reason or "adapter is not open")
        remembered = self._acks.get(command.command_id)
        if remembered is not None:
            return remembered
        observed = time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        if (
            command.start_pose is not self._pose
            or command.expected_pose_version != self._pose_version
        ):
            return self._remember(
                self._failure(
                    command,
                    observed,
                    NavigationAckStatus.REJECTED,
                    NavigationErrorCode.POSE_UNKNOWN,
                    "expected pose/version does not match adapter pose",
                )
            )
        if command.target_pose is RobotPoseNode.UNKNOWN:
            return self._remember(
                self._failure(
                    command,
                    observed,
                    NavigationAckStatus.REJECTED,
                    NavigationErrorCode.TARGET_NOT_FOUND,
                    "state machine did not resolve a canonical target pose",
                )
            )
        try:
            route = self.planner.plan(self._pose, command.target_pose)
            stable_frames = 1
            center_error: float | None = None
            for index, primitive in enumerate(route):
                self._wait_inter_motion_gap(command.inter_motion_delay_ms)
                sample = self._execute_primitive(
                    command, primitive, index, command.timeout_ms
                )
                self._pose = primitive.target_pose
                self._last_motion_completed_at = self.clock()
                if sample is not None:
                    stable_frames = max(1, sample.stable_frames)
                    center_error = sample.center_error_px
            self._pose = command.target_pose
            self._pose_version += 1
            ack = NavigationAck(
                command_id=command.command_id,
                session_id=command.session_id,
                hand_id=command.hand_id,
                expected_state_version=command.expected_state_version,
                action=command.action,
                target_slot=command.target_slot,
                status=NavigationAckStatus.SUCCEEDED,
                observed_at_ns=(
                    time.monotonic_ns() if observed_at_ns is None else observed_at_ns
                ),
                actual_pose=self._pose,
                pose_version=self._pose_version,
                pose_confidence=0.75,
                line_locked=self._pose in _LINE_POSES,
                endpoint_confirmed=self._pose in _ENDPOINT_POSES,
                target_aligned=True,
                stable_frames=stable_frames,
                face_center_error_px=center_error,
            )
        except TimeoutError as exc:
            self._best_effort_stop(command.command_id)
            ack = self._failure(
                command,
                time.monotonic_ns(),
                NavigationAckStatus.TIMED_OUT,
                NavigationErrorCode.ALIGNMENT_TIMEOUT,
                str(exc),
            )
        except (NavigationUnavailableError, CocinoCarProtocolError, ValueError) as exc:
            self._best_effort_stop(command.command_id)
            code = (
                NavigationErrorCode.BOARD_MARKER_NOT_FOUND
                if "follow_line_to_board" in str(exc)
                else NavigationErrorCode.PROTOCOL_ERROR
            )
            ack = self._failure(
                command,
                time.monotonic_ns(),
                NavigationAckStatus.FAILED,
                code,
                str(exc),
            )
        return self._remember(ack)

    def health(self) -> NavigationHealth:
        return NavigationHealth(
            device_id=self.device_id,
            available=True,
            opened=self._opened,
            physical_motion=True,
            pose=self._pose,
            pose_version=self._pose_version,
            reason=self._reason,
        )

    def close(self) -> None:
        if self._opened:
            self._best_effort_stop("close")
        self._opened = False
        self._reason = None

    def _execute_primitive(
        self,
        command: NavigationCommand,
        primitive: RoutePrimitive,
        index: int,
        timeout_ms: int,
    ) -> FaceCenterSample | None:
        prefix = self._request_id(command.command_id, index)
        if primitive.action is CarApiAction.FOLLOW_LINE_TO_BOARD:
            self.client.action(f"{prefix}:board:follow", "follow_line_to_end")
            self.sleeper(self.board_follow_seconds)
            stop_id = f"{prefix}:board:stop"
            self.client.action(stop_id, "stop")
            self._wait_request_success(stop_id, "stop", timeout_ms)
            return None
        if primitive.action is CarApiAction.FOLLOW_LINE_TO_END:
            request_id = f"{prefix}:follow"
            self.client.action(request_id, "follow_line_to_end")
            self._wait_request_success(
                request_id, "follow_line_to_end", timeout_ms
            )
            return None
        if primitive.action is CarApiAction.LINE_RECENTER:
            assert primitive.direction is not None
            request_id = f"{prefix}:line"
            self.client.action(
                request_id,
                "line_recenter_start",
                direction=primitive.direction.value,
            )
            self._wait_request_success(
                request_id, "line_recenter_start", timeout_ms
            )
            return None
        if primitive.action is CarApiAction.PRESET_TURN:
            assert primitive.direction is not None and primitive.degrees is not None
            count = 2 if primitive.degrees == 180 else 1
            for turn_index in range(count):
                if turn_index:
                    self._wait_inter_motion_gap(command.inter_motion_delay_ms)
                request_id = f"{prefix}:preset:{turn_index + 1}"
                self.client.action(
                    request_id,
                    "preset_turn",
                    direction=primitive.direction.value,
                    degrees=90,
                )
                self._wait_request_success(request_id, "preset_turn", timeout_ms)
                self._last_motion_completed_at = self.clock()
            return None
        assert primitive.action is CarApiAction.FACE_TURN
        assert primitive.direction is not None
        if self.face_probe is None:
            raise CocinoCarProtocolError(
                "face_turn requires a PC FaceCenterProbe; none is configured"
            )
        reset = getattr(self.face_probe, "reset", None)
        if reset is not None:
            reset()
        self.client.action(
            f"{prefix}:face:start",
            "face_turn_start",
            direction=primitive.direction.value,
        )
        started = self.clock()
        next_heartbeat = started + self.heartbeat_interval_seconds
        heartbeat_index = 0
        while True:
            sample = self.face_probe.observe_face_center()
            if sample.centered:
                request_id = f"{prefix}:face:stop"
                self.client.action(request_id, "face_turn_stop")
                self._wait_request_success(request_id, "face_turn_stop", timeout_ms)
                return sample
            now = self.clock()
            if (now - started) * 1000 >= timeout_ms:
                raise TimeoutError("face centering timed out")
            route_state = self._route_state(self.client.status())
            self._raise_route_failure(route_state)
            if now >= next_heartbeat:
                heartbeat_index += 1
                self.client.action(
                    f"{prefix}:face:heartbeat:{heartbeat_index}",
                    "face_turn_heartbeat",
                )
                next_heartbeat = now + self.heartbeat_interval_seconds
            self.sleeper(self.poll_interval_seconds)

    def _wait_request_success(
        self,
        request_id: str,
        action: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        started = self.clock()
        while True:
            result = self.client.request_result(request_id)
            if result.get("action") != action:
                raise CocinoCarProtocolError(
                    "request-result action correlation mismatch"
                )
            request_status = result.get("request_status")
            terminal = result.get("terminal")
            if request_status not in {
                "running",
                "succeeded",
                "failed",
                "cancelled",
            } or not isinstance(terminal, bool):
                raise CocinoCarProtocolError(
                    "request result lacks terminal lifecycle fields"
                )
            if terminal:
                if request_status == "succeeded":
                    return result
                raise CocinoCarProtocolError(
                    f"cocino_car {action} ended as {request_status}"
                )
            if (self.clock() - started) * 1000 >= timeout_ms:
                raise TimeoutError(
                    f"request {request_id} did not reach a terminal result"
                )
            self.sleeper(self.poll_interval_seconds)

    def _wait_route_state(
        self, expected: set[str], timeout_ms: int
    ) -> Mapping[str, Any]:
        started = self.clock()
        while True:
            status = self.client.status()
            state = self._route_state(status)
            if state in expected:
                return status
            self._raise_route_failure(state)
            if (self.clock() - started) * 1000 >= timeout_ms:
                raise TimeoutError(
                    f"route state timeout waiting for {sorted(expected)}; got {state}"
                )
            self.sleeper(self.poll_interval_seconds)

    def _wait_inter_motion_gap(self, delay_ms: int) -> None:
        if self._last_motion_completed_at is None or delay_ms <= 0:
            return
        remaining = delay_ms / 1000 - (self.clock() - self._last_motion_completed_at)
        if remaining > 0:
            self.sleeper(remaining)

    @staticmethod
    def _route_state(status: Mapping[str, Any]) -> str:
        route = status.get("route")
        if not isinstance(route, Mapping):
            raise CocinoCarProtocolError("status.route must be an object")
        state = route.get("state")
        if not isinstance(state, str) or not state:
            raise CocinoCarProtocolError("status.route.state must be a string")
        return state

    @staticmethod
    def _raise_route_failure(state: str) -> None:
        if state in _ROUTE_FAILURE_STATES or any(
            marker in state for marker in ("ERROR", "FAULT", "LOST", "TIMEOUT")
        ):
            raise CocinoCarProtocolError(f"cocino_car route failed in state {state}")

    def _failure(
        self,
        command: NavigationCommand,
        observed_at_ns: int,
        status: NavigationAckStatus,
        error_code: NavigationErrorCode,
        reason: str,
    ) -> NavigationAck:
        return NavigationAck(
            command_id=command.command_id,
            session_id=command.session_id,
            hand_id=command.hand_id,
            expected_state_version=command.expected_state_version,
            action=command.action,
            target_slot=command.target_slot,
            status=status,
            observed_at_ns=observed_at_ns,
            actual_pose=self._pose,
            pose_version=self._pose_version,
            pose_confidence=0.0,
            line_locked=self._pose in _LINE_POSES,
            endpoint_confirmed=self._pose in _ENDPOINT_POSES,
            target_aligned=False,
            stable_frames=1,
            error_code=error_code,
            reason=reason[:500] or "navigation failed",
        )

    def _remember(self, ack: NavigationAck) -> NavigationAck:
        self._acks[ack.command_id] = ack
        return ack

    def _best_effort_stop(self, command_id: str) -> None:
        try:
            self.client.action(self._request_id(command_id, 999), "stop")
        except Exception:
            pass

    @staticmethod
    def _request_id(command_id: str, index: int) -> str:
        safe = _REQUEST_ID_SAFE.sub("-", command_id).strip("-") or "navigation"
        suffix = f":p{index}"
        # Primitive-specific suffixes are appended by callers; retain headroom
        # under cocino_car's 160-character request-id limit.
        return f"{safe[: 120 - len(suffix)]}{suffix}"


_LINE_POSES = {
    RobotPoseNode.INIT_TO_END,
    RobotPoseNode.INIT_TO_INIT,
    RobotPoseNode.BOARD_TO_END,
    RobotPoseNode.END_TO_END,
    RobotPoseNode.END_TO_INIT,
}
_ENDPOINT_POSES = {
    RobotPoseNode.INIT_TO_END,
    RobotPoseNode.INIT_TO_INIT,
    RobotPoseNode.INIT_BUTTON,
    RobotPoseNode.INIT_UTG,
    RobotPoseNode.END_TO_END,
    RobotPoseNode.END_TO_INIT,
    RobotPoseNode.END_SMALL_BLIND,
    RobotPoseNode.END_BIG_BLIND,
}


__all__ = [
    "COCINO_CAR_API_VERSION",
    "CocinoCarClient",
    "CocinoCarNavigationAdapter",
    "CocinoCarProtocolError",
    "FaceCenterProbe",
    "FaceCenterSample",
    "JsonTransport",
    "UrllibJsonTransport",
]
