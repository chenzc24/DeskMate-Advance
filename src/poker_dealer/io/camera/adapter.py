"""OpenCV camera adapter for physical and virtual Windows cameras."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Any, Protocol

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


CaptureFactory = Callable[[int, int], CaptureDevice]
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
    source_id: str = "table_camera"
    backend: str = "dshow"
    width: int | None = 1280
    height: int | None = 720
    fps: float | None = 30.0
    disconnect_after_failures: int = 3

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
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

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> OpenCVCamera:
        if self.is_open:
            return self

        factory = self._capture_factory or self._opencv_capture_factory()
        capture = factory(self.config.device_index, self._backend_code())
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                "Cannot open camera index "
                f"{self.config.device_index} with backend {self.config.backend!r}. "
                "Confirm the device is enabled, close other camera apps, then "
                "run the camera probe."
            )

        self._capture = capture
        self._apply_preferences(capture)
        self._sequence_id = 0
        self._consecutive_failures = 0
        return self

    def read(self) -> CameraRead:
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise CameraError("camera is not open")

        ok, raw_image = capture.read()
        observed_at_ns = self._clock_ns()
        if not ok or raw_image is None:
            return self._failed_read(observed_at_ns, "capture_read_failed")

        image = np.asarray(raw_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return self._failed_read(observed_at_ns, "unexpected_frame_format")

        dropped_before = self._consecutive_failures
        self._consecutive_failures = 0
        owned_image = np.ascontiguousarray(image).copy()
        owned_image.setflags(write=False)
        height, width = owned_image.shape[:2]
        nominal_fps = max(0.0, float(capture.get(self._property("CAP_PROP_FPS", 5))))
        packet = FramePacket(
            sequence_id=self._sequence_id,
            captured_at_ns=observed_at_ns,
            source_id=self.config.source_id,
            device_index=self.config.device_index,
            width=width,
            height=height,
            color_space=ColorSpace.BGR,
            nominal_fps=nominal_fps,
            dropped_before=dropped_before,
            image=owned_image,
        )
        self._sequence_id += 1
        return CameraRead(
            status=CameraReadStatus.OK,
            observed_at_ns=observed_at_ns,
            frame=packet,
            consecutive_failures=0,
        )

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> OpenCVCamera:
        return self.open()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def negotiated_properties(self) -> dict[str, int | float | str]:
        """Return properties reported by the active backend for diagnostics."""

        capture = self._capture
        if capture is None or not capture.isOpened():
            raise CameraError("camera is not open")
        return {
            "device_index": self.config.device_index,
            "source_id": self.config.source_id,
            "backend": self.config.backend,
            "width": int(capture.get(self._property("CAP_PROP_FRAME_WIDTH", 3))),
            "height": int(capture.get(self._property("CAP_PROP_FRAME_HEIGHT", 4))),
            "nominal_fps": float(capture.get(self._property("CAP_PROP_FPS", 5))),
        }

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
        preferences = (
            ("CAP_PROP_FRAME_WIDTH", 3, self.config.width),
            ("CAP_PROP_FRAME_HEIGHT", 4, self.config.height),
            ("CAP_PROP_FPS", 5, self.config.fps),
            ("CAP_PROP_BUFFERSIZE", 38, 1),
        )
        for name, fallback, value in preferences:
            if value is not None:
                capture.set(self._property(name, fallback), float(value))

    def _backend_code(self) -> int:
        fallback = {"auto": 0, "dshow": 700, "msmf": 1400}
        names = {"auto": "CAP_ANY", "dshow": "CAP_DSHOW", "msmf": "CAP_MSMF"}
        return self._property(names[self.config.backend], fallback[self.config.backend])

    @staticmethod
    def _property(name: str, fallback: int) -> int:
        return int(getattr(cv2, name, fallback)) if cv2 is not None else fallback

    @staticmethod
    def _opencv_capture_factory() -> CaptureFactory:
        if cv2 is None:
            raise CameraError(
                "OpenCV is not installed. Install this project with "
                "`python -m pip install -e .` before opening a camera."
            )
        return cv2.VideoCapture
