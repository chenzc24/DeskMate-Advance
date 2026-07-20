"""Bounded synchronous microphone adapter backed by python-sounddevice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Any, Protocol

import numpy as np

from deskmate_advance.domain.audio import AudioPacket

try:
    import sounddevice as sd
except ImportError:  # Allows fake-stream tests and clear runtime diagnostics.
    sd = None  # type: ignore[assignment]


class InputStream(Protocol):
    """The small subset of sounddevice.InputStream used by this adapter."""

    active: bool
    samplerate: float
    channels: int
    blocksize: int
    latency: float

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def read(self, frames: int) -> tuple[Any, bool]: ...


StreamFactory = Callable[..., InputStream]
Clock = Callable[[], int]


class MicrophoneError(RuntimeError):
    """Raised when the microphone adapter cannot be used as requested."""


class AudioReadStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    MISSING = "missing"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class MicrophoneConfig:
    """Stable laptop microphone selection and capture parameters."""

    device_index: int = 1
    source_id: str = "intel_smart_sound_microphone_array"
    sample_rate_hz: int = 16_000
    channel_count: int = 1
    block_duration_ms: int = 100
    latency: str | float = "low"
    disconnect_after_failures: int = 3

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if self.block_duration_ms <= 0:
            raise ValueError("block_duration_ms must be positive")
        if isinstance(self.latency, str) and self.latency not in {"low", "high"}:
            raise ValueError("string latency must be 'low' or 'high'")
        if isinstance(self.latency, (int, float)) and self.latency <= 0:
            raise ValueError("numeric latency must be positive")
        if self.disconnect_after_failures <= 0:
            raise ValueError("disconnect_after_failures must be positive")

    @property
    def block_frames(self) -> int:
        return max(1, round(self.sample_rate_hz * self.block_duration_ms / 1000))


@dataclass(frozen=True, slots=True)
class AudioRead:
    """A complete block or an explicit degraded/missing input observation."""

    status: AudioReadStatus
    observed_at_ns: int
    packet: AudioPacket | None
    consecutive_failures: int
    reason: str | None = None


class SoundDeviceMicrophone:
    """Synchronous microphone source with bounded blocks and failure states."""

    def __init__(
        self,
        config: MicrophoneConfig,
        *,
        stream_factory: StreamFactory | None = None,
        clock_ns: Clock = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._stream_factory = stream_factory
        self._clock_ns = clock_ns
        self._stream: InputStream | None = None
        self._sequence_id = 0
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def open(self) -> SoundDeviceMicrophone:
        if self.is_open:
            return self

        factory = self._stream_factory or self._sounddevice_stream_factory()
        if self._stream_factory is None:
            self._check_runtime_settings()
        try:
            stream = factory(
                device=self.config.device_index,
                samplerate=self.config.sample_rate_hz,
                channels=self.config.channel_count,
                dtype="float32",
                blocksize=self.config.block_frames,
                latency=self.config.latency,
            )
            stream.start()
        except Exception as error:
            if "stream" in locals():
                stream.close()
            raise MicrophoneError(
                f"Cannot open microphone index {self.config.device_index}. "
                "Confirm Windows microphone permission and close exclusive "
                f"audio applications ({type(error).__name__})."
            ) from error

        self._stream = stream
        self._sequence_id = 0
        self._consecutive_failures = 0
        return self

    def read(self) -> AudioRead:
        stream = self._stream
        if stream is None or not stream.active:
            raise MicrophoneError("microphone is not open")

        try:
            raw_samples, overflowed = stream.read(self.config.block_frames)
        except Exception as error:
            return self._failed_read(
                self._clock_ns(), f"capture_read_failed:{type(error).__name__}"
            )

        observed_at_ns = self._clock_ns()
        samples = np.asarray(raw_samples)
        expected_shape = (self.config.block_frames, self.config.channel_count)
        if samples.dtype != np.float32 or samples.shape != expected_shape:
            return self._failed_read(observed_at_ns, "unexpected_audio_format")

        self._consecutive_failures = 0
        owned_samples = np.ascontiguousarray(samples).copy()
        owned_samples.setflags(write=False)
        packet = AudioPacket(
            sequence_id=self._sequence_id,
            captured_at_ns=observed_at_ns,
            source_id=self.config.source_id,
            device_index=self.config.device_index,
            sample_rate_hz=self.config.sample_rate_hz,
            channel_count=self.config.channel_count,
            sample_count=self.config.block_frames,
            input_overflowed=bool(overflowed),
            samples=owned_samples,
        )
        self._sequence_id += 1
        return AudioRead(
            status=(AudioReadStatus.DEGRADED if overflowed else AudioReadStatus.OK),
            observed_at_ns=observed_at_ns,
            packet=packet,
            consecutive_failures=0,
            reason="input_overflow" if overflowed else None,
        )

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
        finally:
            stream.close()

    def __enter__(self) -> SoundDeviceMicrophone:
        return self.open()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def negotiated_properties(self) -> dict[str, int | float | str]:
        stream = self._stream
        if stream is None or not stream.active:
            raise MicrophoneError("microphone is not open")
        return {
            "device_index": self.config.device_index,
            "source_id": self.config.source_id,
            "sample_rate_hz": int(stream.samplerate),
            "channel_count": int(stream.channels),
            "block_frames": int(stream.blocksize),
            "latency_seconds": float(stream.latency),
        }

    def _failed_read(self, observed_at_ns: int, reason: str) -> AudioRead:
        self._consecutive_failures += 1
        status = (
            AudioReadStatus.DISCONNECTED
            if self._consecutive_failures >= self.config.disconnect_after_failures
            else AudioReadStatus.MISSING
        )
        return AudioRead(
            status=status,
            observed_at_ns=observed_at_ns,
            packet=None,
            consecutive_failures=self._consecutive_failures,
            reason=reason,
        )

    def _check_runtime_settings(self) -> None:
        if sd is None:
            raise MicrophoneError(
                "sounddevice is not installed. Install the resolved project "
                "dependencies before opening a microphone."
            )
        try:
            sd.check_input_settings(
                device=self.config.device_index,
                samplerate=self.config.sample_rate_hz,
                channels=self.config.channel_count,
                dtype="float32",
            )
        except Exception as error:
            raise MicrophoneError(
                "Microphone does not accept the configured sample rate, channel "
                f"count, or dtype ({type(error).__name__})."
            ) from error

    @staticmethod
    def _sounddevice_stream_factory() -> StreamFactory:
        if sd is None:
            raise MicrophoneError(
                "sounddevice is not installed. Install the resolved project "
                "dependencies before opening a microphone."
            )
        return sd.InputStream
