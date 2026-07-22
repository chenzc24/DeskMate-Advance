from __future__ import annotations

from collections.abc import Iterator
import queue
import time

import numpy as np
import pytest

from poker_dealer.io.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)


class FakeCapture:
    def __init__(self, frames: list[tuple[bool, object]], *, opened: bool = True) -> None:
        self.opened = opened
        self.frames: Iterator[tuple[bool, object]] = iter(frames)
        self.properties: dict[int, float] = {3: 640.0, 4: 480.0, 5: 25.0}
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, object]:
        return next(self.frames)

    def set(self, prop_id: int, value: float) -> bool:
        self.properties[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        return self.properties.get(prop_id, 0.0)

    def release(self) -> None:
        self.released = True


class StreamingFakeCapture(FakeCapture):
    def __init__(self) -> None:
        super().__init__([])
        self.pending: queue.Queue[tuple[bool, object]] = queue.Queue()
        self.read_count = 0

    def read(self) -> tuple[bool, object]:
        while not self.released:
            try:
                item = self.pending.get(timeout=0.02)
            except queue.Empty:
                continue
            self.read_count += 1
            return item
        return False, None

    def push(self, ok: bool, frame: object) -> None:
        self.pending.put((ok, frame))


def _wait_for_reads(capture: StreamingFakeCapture, count: int) -> None:
    deadline = time.monotonic() + 1.0
    while capture.read_count < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert capture.read_count >= count


def test_open_applies_preferences_and_reports_properties() -> None:
    capture = FakeCapture([(True, np.zeros((720, 1280, 3), dtype=np.uint8))])
    camera = OpenCVCamera(
        CameraConfig(device_index=2, width=1280, height=720, fps=30),
        capture_factory=lambda index, backend: capture,
    )

    with camera:
        properties = camera.negotiated_properties()

    assert properties == {
        "device_index": 2,
        "source_id": "table_camera",
        "backend": "dshow",
        "width": 1280,
        "height": 720,
        "nominal_fps": 30.0,
    }
    assert capture.properties[38] == 1.0
    assert capture.released


def test_read_preserves_monotonic_time_and_records_recovered_gap() -> None:
    source = np.zeros((2, 3, 3), dtype=np.uint8)
    capture = FakeCapture([(False, None), (True, source)])
    timestamps = iter([100, 200])
    camera = OpenCVCamera(
        CameraConfig(width=None, height=None, fps=None),
        capture_factory=lambda index, backend: capture,
        clock_ns=lambda: next(timestamps),
    ).open()

    missing = camera.read()
    recovered = camera.read()

    assert missing.status is CameraReadStatus.MISSING
    assert missing.observed_at_ns == 100
    assert recovered.status is CameraReadStatus.OK
    assert recovered.frame is not None
    assert recovered.frame.captured_at_ns == 200
    assert recovered.frame.sequence_id == 0
    assert recovered.frame.dropped_before == 1
    assert recovered.frame.image.flags.writeable is False
    source[0, 0, 0] = 255
    assert recovered.frame.image[0, 0, 0] == 0


def test_repeated_failures_become_disconnected() -> None:
    capture = FakeCapture([(False, None), (False, None)])
    camera = OpenCVCamera(
        CameraConfig(disconnect_after_failures=2),
        capture_factory=lambda index, backend: capture,
    ).open()

    assert camera.read().status is CameraReadStatus.MISSING
    assert camera.read().status is CameraReadStatus.DISCONNECTED


def test_unexpected_frame_is_missing_instead_of_crossing_boundary() -> None:
    capture = FakeCapture([(True, np.zeros((2, 3), dtype=np.uint8))])
    camera = OpenCVCamera(
        CameraConfig(), capture_factory=lambda index, backend: capture
    ).open()

    result = camera.read()

    assert result.status is CameraReadStatus.MISSING
    assert result.frame is None
    assert result.reason == "unexpected_frame_format"


def test_failed_open_releases_device() -> None:
    capture = FakeCapture([], opened=False)
    camera = OpenCVCamera(
        CameraConfig(device_index=3), capture_factory=lambda index, backend: capture
    )

    with pytest.raises(CameraError, match="Cannot open camera index 3"):
        camera.open()

    assert capture.released


def test_network_stream_uses_url_and_delivers_only_latest_frame() -> None:
    capture = StreamingFakeCapture()
    opened_with: list[tuple[object, int]] = []
    camera = OpenCVCamera(
        CameraConfig(
            stream_url="http://robot.local:5000/video_feed",
            source_id="robot_mjpeg_stream",
            backend="auto",
            width=None,
            height=None,
            fps=None,
            read_timeout_ms=250,
        ),
        capture_factory=lambda source, backend: (
            opened_with.append((source, backend)) or capture
        ),
    ).open()

    capture.push(True, np.zeros((2, 3, 3), dtype=np.uint8))
    first = camera.read()
    for value in (1, 2, 3):
        capture.push(True, np.full((2, 3, 3), value, dtype=np.uint8))
    _wait_for_reads(capture, 4)
    latest = camera.read()
    properties = camera.negotiated_properties()
    camera.close()

    assert opened_with[0][0] == "http://robot.local:5000/video_feed"
    assert first.status is CameraReadStatus.OK
    assert latest.status is CameraReadStatus.OK
    assert latest.frame is not None
    assert latest.frame.image[0, 0, 0] == 3
    assert latest.frame.dropped_before == 2
    assert properties["source_kind"] == "network_mjpeg"
    assert properties["latest_frame_buffer"] == 1
    assert properties["backend"] == "ffmpeg"
    assert capture.released


def test_network_stream_failures_become_disconnected_without_guessing() -> None:
    capture = StreamingFakeCapture()
    camera = OpenCVCamera(
        CameraConfig(
            stream_url="http://robot.local:5000/video_feed",
            backend="auto",
            width=None,
            height=None,
            fps=None,
            disconnect_after_failures=3,
            read_timeout_ms=100,
            reconnect_attempts=1,
            reconnect_backoff_ms=0,
        ),
        capture_factory=lambda source, backend: capture,
    ).open()
    for _ in range(3):
        capture.push(False, None)
    _wait_for_reads(capture, 3)

    disconnected = camera.read()
    camera.close()

    assert disconnected.status is CameraReadStatus.DISCONNECTED
    assert disconnected.frame is None
    assert disconnected.reason == "network_reconnect_exhausted"


def test_network_stream_reconnects_without_losing_frame_contract() -> None:
    first_capture = StreamingFakeCapture()
    second_capture = StreamingFakeCapture()
    captures: queue.Queue[StreamingFakeCapture] = queue.Queue()
    captures.put(first_capture)
    captures.put(second_capture)
    camera = OpenCVCamera(
        CameraConfig(
            stream_url="http://robot.local:5000/video_feed",
            source_id="robot_mjpeg_stream",
            backend="auto",
            width=None,
            height=None,
            fps=None,
            disconnect_after_failures=3,
            read_timeout_ms=500,
            reconnect_attempts=2,
            reconnect_backoff_ms=0,
        ),
        capture_factory=lambda source, backend: captures.get_nowait(),
    ).open()
    first_capture.push(True, np.zeros((2, 3, 3), dtype=np.uint8))
    first = camera.read()
    for _ in range(3):
        first_capture.push(False, None)
    second_capture.push(True, np.full((2, 3, 3), 9, dtype=np.uint8))

    recovered = camera.read()
    properties = camera.negotiated_properties()
    camera.close()

    assert first.status is CameraReadStatus.OK
    assert recovered.status is CameraReadStatus.OK
    assert recovered.frame is not None
    assert recovered.frame.sequence_id == 1
    assert recovered.frame.source_id == "robot_mjpeg_stream"
    assert recovered.frame.image[0, 0, 0] == 9
    assert camera.network_reconnects == 1
    assert properties["reconnect_count"] == 1
    assert first_capture.released
    assert second_capture.released


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_index": -1},
        {"backend": "invalid"},
        {"width": 0},
        {"height": 0},
        {"fps": 0},
        {"disconnect_after_failures": 0},
        {"open_timeout_ms": 0},
        {"read_timeout_ms": 0},
        {"reconnect_attempts": 0},
        {"reconnect_backoff_ms": -1},
        {"stream_url": ""},
        {"stream_url": "rtsp://robot.local/video"},
        {"stream_url": "http://user:secret@robot.local/video"},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)
