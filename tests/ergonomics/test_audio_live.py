from __future__ import annotations

from collections.abc import Callable
from collections import deque
import gc
from threading import Event, Lock
import time
import weakref

import numpy as np
import pytest

from deskmate_advance.domain.audio import AudioPacket
from deskmate_advance.features.ergonomics.audio_live import AudioLevelPoller
from deskmate_advance.perception.audio import (
    AudioRead,
    AudioReadStatus,
    MicrophoneConfig,
)
from deskmate_advance.perception.ergonomics import ObservationState


def _packet_read(
    observed_at_ns: int,
    *,
    level: float = 0.25,
    status: AudioReadStatus = AudioReadStatus.OK,
) -> AudioRead:
    samples = np.full((1_600, 1), level, dtype=np.float32)
    packet = AudioPacket(
        sequence_id=0,
        captured_at_ns=observed_at_ns,
        source_id="test_microphone",
        device_index=1,
        sample_rate_hz=16_000,
        channel_count=1,
        sample_count=1_600,
        input_overflowed=status is AudioReadStatus.DEGRADED,
        samples=samples,
    )
    return AudioRead(
        status=status,
        observed_at_ns=observed_at_ns,
        packet=packet,
        consecutive_failures=0,
        reason="input_overflow" if status is AudioReadStatus.DEGRADED else None,
    )


class FakeMicrophone:
    def __init__(
        self,
        reads: list[AudioRead],
        *,
        open_error: Exception | None = None,
    ) -> None:
        self.config = MicrophoneConfig(source_id="test_microphone")
        self._reads = deque(reads)
        self._lock = Lock()
        self._closed = Event()
        self.open_error = open_error
        self.open_count = 0
        self.close_count = 0

    def open(self) -> FakeMicrophone:
        self.open_count += 1
        if self.open_error is not None:
            raise self.open_error
        return self

    def read(self) -> AudioRead:
        with self._lock:
            if self._reads:
                return self._reads.popleft()
        self._closed.wait(timeout=2.0)
        raise RuntimeError("closed")

    def close(self) -> None:
        self.close_count += 1
        self._closed.set()


def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("background audio poller did not reach expected state")


def test_poller_exposes_latest_scalar_level_and_exact_window() -> None:
    microphone = FakeMicrophone([_packet_read(200_000_000, level=0.25)])
    poller = AudioLevelPoller(microphone.config, microphone=microphone)

    with poller:
        _wait_until(lambda: poller.latest_observation is not None)
        snapshot = poller.snapshot()

    assert snapshot.status is AudioReadStatus.OK
    assert snapshot.error is None
    assert snapshot.observed_at_ns == 200_000_000
    assert snapshot.observation is not None
    assert snapshot.observation.state is ObservationState.VALID
    assert snapshot.observation.window_started_at_ns == 100_000_000
    assert snapshot.observation.window_ended_at_ns == 200_000_000
    assert snapshot.observation.rms == pytest.approx(0.25)
    assert microphone.open_count == 1
    assert microphone.close_count >= 1


def test_degraded_packet_keeps_level_but_exposes_overflow() -> None:
    microphone = FakeMicrophone(
        [_packet_read(200_000_000, status=AudioReadStatus.DEGRADED)]
    )
    poller = AudioLevelPoller(microphone.config, microphone=microphone).start()
    try:
        _wait_until(lambda: poller.latest_status is AudioReadStatus.DEGRADED)
        snapshot = poller.snapshot()
    finally:
        poller.close()

    assert snapshot.observation is not None
    assert snapshot.observation.valid
    assert snapshot.error == "input_overflow"


def test_missing_read_replaces_prior_level_instead_of_becoming_quiet() -> None:
    missing = AudioRead(
        status=AudioReadStatus.MISSING,
        observed_at_ns=300_000_000,
        packet=None,
        consecutive_failures=1,
        reason="capture_read_failed:RuntimeError",
    )
    microphone = FakeMicrophone([_packet_read(200_000_000), missing])
    poller = AudioLevelPoller(
        microphone.config,
        microphone=microphone,
        failure_backoff_seconds=0,
    ).start()
    try:
        _wait_until(
            lambda: poller.latest_error == "capture_read_failed:RuntimeError"
        )
        snapshot = poller.snapshot()
    finally:
        poller.close()

    assert snapshot.status is AudioReadStatus.MISSING
    assert snapshot.observation is None
    assert snapshot.observed_at_ns == 300_000_000


def test_non_increasing_read_timestamp_is_unknown_not_reused_level() -> None:
    microphone = FakeMicrophone(
        [_packet_read(200_000_000), _packet_read(200_000_000)]
    )
    poller = AudioLevelPoller(microphone.config, microphone=microphone).start()
    try:
        _wait_until(
            lambda: poller.latest_error == "non_increasing_audio_read_timestamp"
        )
        snapshot = poller.snapshot()
    finally:
        poller.close()

    assert snapshot.status is AudioReadStatus.MISSING
    assert snapshot.observation is None


def test_open_error_is_a_disconnected_snapshot() -> None:
    microphone = FakeMicrophone([], open_error=RuntimeError("unavailable"))
    poller = AudioLevelPoller(microphone.config, microphone=microphone).start()
    try:
        _wait_until(
            lambda: poller.latest_error == "microphone_open_failed:RuntimeError"
        )
        snapshot = poller.snapshot()
    finally:
        poller.close()

    assert snapshot.status is AudioReadStatus.DISCONNECTED
    assert snapshot.observation is None
    assert snapshot.observed_at_ns is None


def test_unexpected_read_error_is_a_disconnected_snapshot() -> None:
    class ReadErrorMicrophone(FakeMicrophone):
        def read(self) -> AudioRead:
            raise OSError("device removed")

    microphone = ReadErrorMicrophone([])
    poller = AudioLevelPoller(microphone.config, microphone=microphone).start()
    try:
        _wait_until(
            lambda: poller.latest_error == "microphone_read_failed:OSError"
        )
        snapshot = poller.snapshot()
    finally:
        poller.close()

    assert snapshot.status is AudioReadStatus.DISCONNECTED
    assert snapshot.observation is None


def test_start_and_close_are_idempotent() -> None:
    microphone = FakeMicrophone([_packet_read(200_000_000)])
    poller = AudioLevelPoller(microphone.config, microphone=microphone)

    assert poller.start() is poller
    assert poller.start() is poller
    _wait_until(lambda: poller.latest_observation is not None)
    poller.close()
    poller.close()

    assert not poller.is_running
    assert microphone.open_count == 1


def test_poller_does_not_retain_audio_samples_after_level_calculation() -> None:
    read = _packet_read(200_000_000)
    assert read.packet is not None
    samples_reference = weakref.ref(read.packet.samples)
    microphone = FakeMicrophone([read])
    poller = AudioLevelPoller(microphone.config, microphone=microphone).start()
    _wait_until(lambda: poller.latest_observation is not None)

    del read
    gc.collect()
    try:
        assert samples_reference() is None
        assert poller.snapshot().observation is not None
    finally:
        poller.close()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("failure_backoff_seconds", -0.1),
        ("failure_backoff_seconds", float("nan")),
        ("close_timeout_seconds", 0.0),
    ],
)
def test_invalid_poller_timing_is_rejected(keyword: str, value: float) -> None:
    config = MicrophoneConfig(source_id="test_microphone")

    with pytest.raises(ValueError):
        AudioLevelPoller(config, **{keyword: value})
