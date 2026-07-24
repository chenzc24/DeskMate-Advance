from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from poker_dealer.runtime import AudioInputHealth, StreamingPcm16Resampler
from poker_dealer.runtime.live_perception import LivePerceptionSession


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: dict[str, object],
    ) -> None:
        del observed_at_ns
        self.events.append((kind, payload))


def test_native_44100_pcm_is_continuously_resampled_to_16000() -> None:
    source_rate = 44_100
    target_rate = 16_000
    time_axis = np.arange(source_rate, dtype=np.float64) / source_rate
    source = np.rint(np.sin(2 * np.pi * 440 * time_axis) * 12_000).astype("<i2")
    resampler = StreamingPcm16Resampler(source_rate, target_rate)

    outputs = []
    for start in range(0, len(source), 3_307):
        outputs.append(resampler.process(source[start : start + 3_307].tobytes()))
    result = np.frombuffer(b"".join(outputs), dtype="<i2")

    assert abs(len(result) - target_rate) <= 1
    frequencies = np.fft.rfftfreq(len(result), d=1 / target_rate)
    peak_hz = frequencies[np.argmax(np.abs(np.fft.rfft(result)))]
    assert abs(peak_hz - 440) < 2


def test_resampler_identity_rate_preserves_pcm_bytes() -> None:
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype="<i2").tobytes()
    assert StreamingPcm16Resampler(16_000, 16_000).process(pcm) == pcm


def test_audio_health_detects_callback_staleness_and_status() -> None:
    now = [1_000_000_000]
    health = AudioInputHealth(clock_ns=lambda: now[0])
    health.record_callback(
        11_025, "", rms_level=0.10, peak_level=0.25
    )
    snapshot = health.snapshot()
    assert snapshot.callback_blocks == 1
    assert snapshot.callback_frames == 11_025
    assert snapshot.rms_level == 0.10
    assert snapshot.peak_level == 0.25
    assert not snapshot.is_stale(now[0], 2_000)

    now[0] += 2_100_000_000
    assert health.snapshot().is_stale(now[0], 2_000)
    health.record_callback(11_025, "input overflow")
    snapshot = health.snapshot()
    assert snapshot.status_events == 1
    assert snapshot.last_status == "input overflow"


def test_live_session_restarts_stale_audio_and_reports_recovery() -> None:
    now = [1_000_000_000]
    sink = RecordingEventSink()
    session = object.__new__(LivePerceptionSession)
    session.config = SimpleNamespace(speech_enabled=True)
    session.speech_config = SimpleNamespace(audio={"sample_rate_hz": 16_000})
    session._audio_capture_rate_hz = 44_100
    session._audio_health = AudioInputHealth(clock_ns=lambda: now[0])
    session._audio_stream = SimpleNamespace(active=True)
    session._audio_event_sink = sink
    session._audio_disconnected = False
    session._audio_unavailable_reported = False
    session._audio_last_reconnect_attempt_ns = None
    session._audio_last_status_events = 0
    session._audio_stale_after_ms = 2_000
    session._audio_reconnect_cooldown_ms = 2_000
    restarts = []
    session._restart_audio_stream = lambda: restarts.append(True)

    now[0] += 2_100_000_000
    assert not session._audio_input_is_healthy(now[0])
    assert restarts == [True]
    assert sink.events[0][0] == "audio_link_lost"

    session._audio_health.record_callback(11_025, "")
    assert session._audio_input_is_healthy(now[0])
    assert sink.events[-1][0] == "audio_link_restored"
