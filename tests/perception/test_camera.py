from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from deskmate_advance.perception.camera import (
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
        "source_id": "laptop_camera",
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_index": -1},
        {"backend": "invalid"},
        {"width": 0},
        {"height": 0},
        {"fps": 0},
        {"disconnect_after_failures": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)
