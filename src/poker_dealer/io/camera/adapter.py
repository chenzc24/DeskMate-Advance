"""OpenCV camera adapter for physical and virtual Windows cameras."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Event, Thread, current_thread
import time
from typing import Any, Protocol
from urllib.parse import urlsplit

import numpy as np

from poker_dealer.domain.frame import ColorSpace, FramePacket

try:
    import cv2
except ImportError:  # Allows domain tests and clear runtime diagnostics.
    cv2 = None  # type: ignore[assignment]


class CaptureDevice(Protocol):
    """The small subset of cv2.VideoCapture used by this adapter."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


CaptureSource = int | str
CaptureFactory = Callable[[CaptureSource, int], CaptureDevice]
Clock = Callable[[], int]


class CameraError(RuntimeError):
    """Raised when the camera adapter cannot be used as requested."""


class CameraReadStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Stable camera selection and capture preferences."""

    device_index: int = 0
    stream_url: str | None = None
    source_id: str = "table_camera"
    backend: str = "dshow"
    width: int | None = 1280
    height: int | None = 720
    fps: float | None = 30.0
    disconnect_after_failures: int = 3
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 2000
    reconnect_attempts: int = 5
    reconnect_backoff_ms: int = 250

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if self.stream_url is not None:
            if not self.stream_url.strip():
                raise ValueError("stream_url must not be blank")
            parsed = urlsplit(self.stream_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("stream_url must be an absolute HTTP(S) URL")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("stream_url must not contain embedded credentials")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("backend must be one of: auto, dshow, msmf")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive when set")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive when set")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive when set")
        if self.disconnect_after_failures <= 0:
            raise ValueError("disconnect_after_failures must be positive")
        if self.open_timeout_ms <= 0:
            raise ValueError("open_timeout_ms must be positive")
        if self.read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms must be positive")
        if self.reconnect_attempts <= 0:
            raise ValueError("reconnect_attempts must be positive")
        if self.reconnect_backoff_ms < 0:
            raise ValueError("reconnect_backoff_ms must be non-negative")

    @property
    def is_network_stream(self) -> bool:
        return self.stream_url is not None

    @property
    def capture_source(self) -> CaptureSource:
        return self.stream_url if self.stream_url is not None else self.device_index


@dataclass(frozen=True, slots=True)
class CameraRead:
    """A successful frame or an explicit missing/disconnected observation."""

    status: CameraReadStatus
    observed_at_ns: int
    frame: FramePacket | None
    consecutive_failures: int
    reason: str | None = None


class OpenCVCamera:
    """Bounded synchronous camera source with explicit failure semantics."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        capture_factory: CaptureFactory | None = None,
        clock_ns: Clock = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory
        self._clock_ns = clock_ns
        self._capture: CaptureDevice | None = None
        self._sequence_id = 0
        self._consecutive_failures = 0
        self._negotiated: dict[str, int | float | str] | None = None
        self._network_condition = Condition()
        self._network_stop = Event()
        self._network_thread: Thread | None = None
        self._network_latest_image: np.ndarray[Any, Any] | None = None
        self._network_latest_at_ns = 0
        self._network_latest_source_sequence = -1
        self._network_delivered_source_sequence = -1
        self._network_source_failures = 0
        self._network_last_reason: str | None = None
        self._network_reconnecting = False
        self._network_terminal_failure = False
        self._network_reconnects = 0

    @property
    def is_open(self) -> bool:
        capture_open = self._capture is not None and self._capture.isOpened()
        if not self.config.is_network_stream:
            return capture_open
        thread_alive = self._network_thread is not None and self._network_thread.is_alive()
        return capture_open or (
            thread_alive
            and not self._network_terminal_failure
            and not self._network_stop.is_set()
        )

    @property
    def network_reconnects(self) -> int:
        with self._network_condition:
            return self._network_reconnects

    @property
    def network_reconnecting(self) -> bool:
        with self._network_condition:
            return self._network_reconnecting

    def open(self) -> OpenCVCamera:
        if self.is_open:
            return self

        capture = self._create_capture()
        if not capture.isOpened():
            capture.release()
            if self.config.is_network_stream:
                raise CameraError(
                    "Cannot open network camera stream with the FFmpeg backend. "
                    "Confirm the private-network route and MJPEG service are available."
                )
            raise CameraError(
                "Cannot open camera index "
                f"{self.config.device_index} with backend {self.config.backend!r}. "
                "Confirm the device is enabled, close other camera apps, then "
                "run the camera probe."
            )

        self._capture = capture
        self._apply_preferences(capture)
        self._negotiated = self._read_negotiated_properties(capture)
        self._sequence_id = 0
        self._consecutive_failures = 0
        if self.config.is_network_stream:
            with self._network_condition:
                self._network_latest_image = None
                self._network_latest_at_ns = 0
                self._network_latest_source_sequence = -1
                self._network_delivered_source_sequence = -1
                self._network_source_failures = 0
                self._network_last_reason = None
                self._network_reconnecting = False
                self._network_terminal_failure = False
                self._network_reconnects = 0
            self._network_stop.clear()
            self._network_thread = Thread(
                target=self._network_reader,
                name=f"camera-reader-{self.config.source_id}",
                daemon=True,
            )
            self._network_thread.start()
        return self

    def read(self) -> CameraRead:
        if self.config.is_network_stream:
            if self._network_thread is None:
                raise CameraError("camera is not open")
            return self._read_network_frame()
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise CameraError("camera is not open")

        ok, raw_image = capture.read()
        observed_at_ns = self._clock_ns()
        if not ok or raw_image is None:
            return self._failed_read(observed_at_ns, "capture_read_failed")

        owned_image = self._owned_image(raw_image)
        if owned_image is None:
            return self._failed_read(observed_at_ns, "unexpected_frame_format")
        nominal_fps = max(0.0, float(capture.get(self._property("CAP_PROP_FPS", 5))))
        return self._successful_read(
            owned_image,
            observed_at_ns=observed_at_ns,
            nominal_fps=nominal_fps,
            dropped_before=self._consecutive_failures,
        )

    def close(self) -> None:
        self._network_stop.set()
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        with self._network_condition:
            self._network_condition.notify_all()
        thread, self._network_thread = self._network_thread, None
        if thread is not None and thread is not current_thread():
            thread.join(
                timeout=max(
                    1.0,
                    self.config.open_timeout_ms / 1000 + 0.5,
                    self.config.read_timeout_ms / 1000 + 0.5,
                )
            )
        self._negotiated = None

    def __enter__(self) -> OpenCVCamera:
        return self.open()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def negotiated_properties(self) -> dict[str, int | float | str]:
        """Return properties reported by the active backend for diagnostics."""

        if (
            not self.config.is_network_stream
            and (self._capture is None or not self._capture.isOpened())
        ):
            raise CameraError("camera is not open")
        if self._negotiated is None:
            raise CameraError("camera properties are unavailable")
        properties = dict(self._negotiated)
        if self.config.is_network_stream:
            properties["reconnect_count"] = self.network_reconnects
            properties["reconnecting"] = self.network_reconnecting
        return properties

    def _read_network_frame(self) -> CameraRead:
        deadline = time.monotonic() + self.config.read_timeout_ms / 1000
        with self._network_condition:
            while (
                self._network_latest_source_sequence
                <= self._network_delivered_source_sequence
                and not self._network_terminal_failure
                and not self._network_stop.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._network_condition.wait(timeout=remaining)

            source_sequence = self._network_latest_source_sequence
            if source_sequence > self._network_delivered_source_sequence:
                image = self._network_latest_image
                observed_at_ns = self._network_latest_at_ns
                dropped_before = max(
                    0,
                    source_sequence - self._network_delivered_source_sequence - 1,
                )
                self._network_delivered_source_sequence = source_sequence
                assert image is not None
                nominal_fps = float(
                    (self._negotiated or {}).get("nominal_fps", 0.0)
                )
                return self._successful_read(
                    image,
                    observed_at_ns=observed_at_ns,
                    nominal_fps=nominal_fps,
                    dropped_before=dropped_before,
                )

            if self._network_terminal_failure or self._network_stop.is_set():
                self._consecutive_failures += 1
                return CameraRead(
                    status=CameraReadStatus.DISCONNECTED,
                    observed_at_ns=self._clock_ns(),
                    frame=None,
                    consecutive_failures=self._consecutive_failures,
                    reason=self._network_last_reason or "network_reconnect_exhausted",
                )
            self._consecutive_failures += 1
            reason = (
                "network_reconnecting"
                if self._network_reconnecting
                else self._network_last_reason or "network_read_timeout"
            )
            return CameraRead(
                status=CameraReadStatus.MISSING,
                observed_at_ns=self._clock_ns(),
                frame=None,
                consecutive_failures=self._consecutive_failures,
                reason=reason,
            )

    def _network_reader(self) -> None:
        capture = self._capture
        if capture is None:
            return
        while not self._network_stop.is_set():
            try:
                ok, raw_image = capture.read()
            except Exception:  # pragma: no cover - backend-specific exception path
                ok, raw_image = False, None
                reason = "capture_read_exception"
            else:
                reason = "capture_read_failed"
            observed_at_ns = self._clock_ns()
            if self._network_stop.is_set():
                return
            owned_image = self._owned_image(raw_image) if ok else None
            should_reconnect = False
            with self._network_condition:
                if owned_image is None:
                    self._network_source_failures += 1
                    self._network_last_reason = (
                        "unexpected_frame_format" if ok else reason
                    )
                    self._network_condition.notify_all()
                    if (
                        self._network_source_failures
                        >= self.config.disconnect_after_failures
                    ):
                        self._network_reconnecting = True
                        should_reconnect = True
                    if not should_reconnect:
                        continue
                else:
                    self._network_source_failures = 0
                    self._network_last_reason = None
                    self._network_latest_source_sequence += 1
                    self._network_latest_at_ns = observed_at_ns
                    self._network_latest_image = owned_image
                    self._network_condition.notify_all()
            if should_reconnect:
                replacement = self._reconnect_network_capture(capture)
                if replacement is None:
                    return
                capture = replacement

    def _reconnect_network_capture(
        self, failed_capture: CaptureDevice
    ) -> CaptureDevice | None:
        failed_capture.release()
        with self._network_condition:
            if self._capture is failed_capture:
                self._capture = None
            self._network_condition.notify_all()

        for attempt in range(self.config.reconnect_attempts):
            if attempt and self._network_stop.wait(
                self.config.reconnect_backoff_ms / 1000
            ):
                return None
            if self._network_stop.is_set():
                return None
            try:
                replacement = self._create_capture()
            except Exception:  # pragma: no cover - backend-specific open failure
                with self._network_condition:
                    self._network_last_reason = "network_reopen_exception"
                    self._network_condition.notify_all()
                continue
            if replacement.isOpened():
                self._apply_preferences(replacement)
                negotiated = self._read_negotiated_properties(replacement)
                with self._network_condition:
                    if self._network_stop.is_set():
                        replacement.release()
                        return None
                    self._capture = replacement
                    self._negotiated = negotiated
                    self._network_source_failures = 0
                    self._network_last_reason = None
                    self._network_reconnecting = False
                    self._network_terminal_failure = False
                    self._network_reconnects += 1
                    self._network_condition.notify_all()
                return replacement
            replacement.release()

        with self._network_condition:
            self._network_reconnecting = False
            self._network_terminal_failure = True
            self._network_last_reason = "network_reconnect_exhausted"
            self._network_condition.notify_all()
        return None

    def _successful_read(
        self,
        image: np.ndarray[Any, Any],
        *,
        observed_at_ns: int,
        nominal_fps: float,
        dropped_before: int,
    ) -> CameraRead:
        self._consecutive_failures = 0
        height, width = image.shape[:2]
        packet = FramePacket(
            sequence_id=self._sequence_id,
            captured_at_ns=observed_at_ns,
            source_id=self.config.source_id,
            device_index=self.config.device_index,
            width=width,
            height=height,
            color_space=ColorSpace.BGR,
            nominal_fps=max(0.0, nominal_fps),
            dropped_before=dropped_before,
            image=image,
        )
        self._sequence_id += 1
        return CameraRead(
            status=CameraReadStatus.OK,
            observed_at_ns=observed_at_ns,
            frame=packet,
            consecutive_failures=0,
        )

    @staticmethod
    def _owned_image(raw_image: object) -> np.ndarray[Any, Any] | None:
        if raw_image is None:
            return None
        image = np.asarray(raw_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return None
        owned_image = np.ascontiguousarray(image).copy()
        owned_image.setflags(write=False)
        return owned_image

    def _failed_read(self, observed_at_ns: int, reason: str) -> CameraRead:
        self._consecutive_failures += 1
        status = (
            CameraReadStatus.DISCONNECTED
            if self._consecutive_failures >= self.config.disconnect_after_failures
            else CameraReadStatus.MISSING
        )
        return CameraRead(
            status=status,
            observed_at_ns=observed_at_ns,
            frame=None,
            consecutive_failures=self._consecutive_failures,
            reason=reason,
        )

    def _apply_preferences(self, capture: CaptureDevice) -> None:
        preferences = [("CAP_PROP_BUFFERSIZE", 38, 1)]
        if not self.config.is_network_stream:
            preferences[:0] = [
                ("CAP_PROP_FRAME_WIDTH", 3, self.config.width),
                ("CAP_PROP_FRAME_HEIGHT", 4, self.config.height),
                ("CAP_PROP_FPS", 5, self.config.fps),
            ]
        for name, fallback, value in preferences:
            if value is not None:
                capture.set(self._property(name, fallback), float(value))

    def _read_negotiated_properties(
        self, capture: CaptureDevice
    ) -> dict[str, int | float | str]:
        properties: dict[str, int | float | str] = {
            "device_index": self.config.device_index,
            "source_id": self.config.source_id,
            "backend": "ffmpeg" if self.config.is_network_stream else self.config.backend,
            "width": int(capture.get(self._property("CAP_PROP_FRAME_WIDTH", 3))),
            "height": int(capture.get(self._property("CAP_PROP_FRAME_HEIGHT", 4))),
            "nominal_fps": float(capture.get(self._property("CAP_PROP_FPS", 5))),
        }
        if self.config.is_network_stream:
            properties.update(
                {
                    "source_kind": "network_mjpeg",
                    "latest_frame_buffer": 1,
                    "open_timeout_ms": self.config.open_timeout_ms,
                    "read_timeout_ms": self.config.read_timeout_ms,
                    "reconnect_attempts": self.config.reconnect_attempts,
                    "reconnect_backoff_ms": self.config.reconnect_backoff_ms,
                }
            )
        return properties

    def _create_capture(self) -> CaptureDevice:
        source = self.config.capture_source
        if self._capture_factory is None:
            return self._open_opencv_capture(source, self._backend_code())
        return self._capture_factory(source, self._backend_code())

    def _backend_code(self) -> int:
        if self.config.is_network_stream:
            return self._property("CAP_FFMPEG", 1900)
        fallback = {"auto": 0, "dshow": 700, "msmf": 1400}
        names = {"auto": "CAP_ANY", "dshow": "CAP_DSHOW", "msmf": "CAP_MSMF"}
        return self._property(names[self.config.backend], fallback[self.config.backend])

    @staticmethod
    def _property(name: str, fallback: int) -> int:
        return int(getattr(cv2, name, fallback)) if cv2 is not None else fallback

    def _open_opencv_capture(
        self, source: CaptureSource, backend_code: int
    ) -> CaptureDevice:
        if cv2 is None:
            raise CameraError(
                "OpenCV is not installed. Install this project with "
                "`python -m pip install -e .` before opening a camera."
            )
        if self.config.is_network_stream:
            params = [
                self._property("CAP_PROP_OPEN_TIMEOUT_MSEC", 53),
                self.config.open_timeout_ms,
                self._property("CAP_PROP_READ_TIMEOUT_MSEC", 54),
                self.config.read_timeout_ms,
            ]
            return cv2.VideoCapture(source, backend_code, params)
        return cv2.VideoCapture(source, backend_code)
