from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from deskmate_advance.perception.audio import (
    AudioReadStatus,
    MicrophoneConfig,
    MicrophoneError,
    SoundDeviceMicrophone,
)


class FakeInputStream:
    def __init__(self, reads: list[tuple[object, bool] | Exception]) -> None:
        self._reads: Iterator[tuple[object, bool] | Exception] = iter(reads)
        self.active = False
        self.samplerate = 16_000.0
        self.channels = 1
        self.blocksize = 1_600
        self.latency = 0.02
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True

    def read(self, frames: int) -> tuple[Any, bool]:
        item = next(self._reads)
        if isinstance(item, Exception):
            raise item
        return item


def test_open_passes_bounded_capture_settings() -> None:
    stream = FakeInputStream([])
    received: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeInputStream:
        received.update(kwargs)
        return stream

    config = MicrophoneConfig(device_index=1, block_duration_ms=100)
    microphone = SoundDeviceMicrophone(config, stream_factory=factory)

    with microphone:
        properties = microphone.negotiated_properties()

    assert received == {
        "device": 1,
        "samplerate": 16_000,
        "channels": 1,
        "dtype": "float32",
        "blocksize": 1_600,
        "latency": "low",
    }
    assert properties["block_frames"] == 1_600
    assert stream.closed


def test_read_returns_owned_timestamped_packet() -> None:
    source = np.zeros((1_600, 1), dtype=np.float32)
    stream = FakeInputStream([(source, False)])
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(),
        stream_factory=lambda **kwargs: stream,
        clock_ns=lambda: 123_456,
    ).open()

    result = microphone.read()

    assert result.status is AudioReadStatus.OK
    assert result.observed_at_ns == 123_456
    assert result.packet is not None
    assert result.packet.sequence_id == 0
    assert result.packet.sample_rate_hz == 16_000
    assert result.packet.sample_count == 1_600
    assert result.packet.samples.flags.writeable is False
    source[0, 0] = 1.0
    assert result.packet.samples[0, 0] == 0.0


def test_overflow_returns_degraded_packet() -> None:
    samples = np.zeros((1_600, 1), dtype=np.float32)
    stream = FakeInputStream([(samples, True)])
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(), stream_factory=lambda **kwargs: stream
    ).open()

    result = microphone.read()

    assert result.status is AudioReadStatus.DEGRADED
    assert result.reason == "input_overflow"
    assert result.packet is not None
    assert result.packet.input_overflowed


def test_malformed_audio_becomes_missing() -> None:
    wrong_shape = np.zeros((800, 1), dtype=np.float32)
    stream = FakeInputStream([(wrong_shape, False)])
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(), stream_factory=lambda **kwargs: stream
    ).open()

    result = microphone.read()

    assert result.status is AudioReadStatus.MISSING
    assert result.packet is None
    assert result.reason == "unexpected_audio_format"


def test_repeated_read_failures_become_disconnected() -> None:
    stream = FakeInputStream([RuntimeError("one"), RuntimeError("two")])
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(disconnect_after_failures=2),
        stream_factory=lambda **kwargs: stream,
    ).open()

    assert microphone.read().status is AudioReadStatus.MISSING
    assert microphone.read().status is AudioReadStatus.DISCONNECTED


def test_read_before_open_is_rejected() -> None:
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(), stream_factory=lambda **kwargs: FakeInputStream([])
    )

    with pytest.raises(MicrophoneError, match="microphone is not open"):
        microphone.read()


def test_failed_open_is_wrapped_and_closes_stream() -> None:
    class FailingStream(FakeInputStream):
        def start(self) -> None:
            raise RuntimeError("busy")

    stream = FailingStream([])
    microphone = SoundDeviceMicrophone(
        MicrophoneConfig(), stream_factory=lambda **kwargs: stream
    )

    with pytest.raises(MicrophoneError, match="Cannot open microphone index 1"):
        microphone.open()

    assert stream.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_index": -1},
        {"source_id": ""},
        {"sample_rate_hz": 0},
        {"channel_count": 0},
        {"block_duration_ms": 0},
        {"latency": "medium"},
        {"latency": 0.0},
        {"disconnect_after_failures": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MicrophoneConfig(**kwargs)
