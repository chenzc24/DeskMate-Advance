"""Bounded background audio-level polling for the Part A live runtime.

Only scalar level observations cross this boundary.  The poller never queues or
retains microphone packets, sample arrays, recordings, or an error history.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event, Lock, Thread, current_thread
from typing import Protocol

from deskmate_advance.perception.audio import (
    AudioRead,
    AudioReadStatus,
    MicrophoneConfig,
    SoundDeviceMicrophone,
)
from deskmate_advance.perception.ergonomics import (
    AudioLevelCalculator,
    AudioLevelObservation,
    ObservationState,
)


class MicrophoneSource(Protocol):
    """Small injectable microphone interface used by the level poller."""

    config: MicrophoneConfig

    def open(self) -> object: ...

    def read(self) -> AudioRead: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioLevelSnapshot:
    """Atomic latest-only view of microphone health and scalar evidence."""

    status: AudioReadStatus
    observed_at_ns: int | None
    observation: AudioLevelObservation | None
    error: str | None


class AudioLevelPoller:
    """Poll one bounded microphone block at a time on a background thread.

    A missing, disconnected, malformed, or non-monotonic read replaces the
    latest observation with ``None`` (or an invalid calculator observation).
    Consequently callers cannot accidentally reuse an earlier quiet level as
    evidence that the current environment is quiet.
    """

    def __init__(
        self,
        config: MicrophoneConfig,
        *,
        microphone: MicrophoneSource | None = None,
        calculator: AudioLevelCalculator | None = None,
        failure_backoff_seconds: float = 0.05,
        close_timeout_seconds: float = 2.0,
    ) -> None:
        if (
            not math.isfinite(failure_backoff_seconds)
            or failure_backoff_seconds < 0
        ):
            raise ValueError("failure_backoff_seconds must be finite and non-negative")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be finite and positive")

        self.config = config
        self._microphone = microphone or SoundDeviceMicrophone(config)
        self._calculator = calculator or AudioLevelCalculator()
        self._failure_backoff_seconds = failure_backoff_seconds
        self._close_timeout_seconds = close_timeout_seconds
        self._stop = Event()
        self._snapshot_lock = Lock()
        self._lifecycle_lock = Lock()
        self._thread: Thread | None = None
        self._latest = AudioLevelSnapshot(
            status=AudioReadStatus.MISSING,
            observed_at_ns=None,
            observation=None,
            error="not_started",
        )
        self._last_read_ns: int | None = None
        self._last_packet_ns: int | None = None

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def latest_status(self) -> AudioReadStatus:
        return self.snapshot().status

    @property
    def latest_observation(self) -> AudioLevelObservation | None:
        return self.snapshot().observation

    @property
    def latest_error(self) -> str | None:
        return self.snapshot().error

    def snapshot(self) -> AudioLevelSnapshot:
        """Return one internally consistent latest-only scalar snapshot."""

        with self._snapshot_lock:
            return self._latest

    def start(self) -> AudioLevelPoller:
        """Start polling if needed; repeated calls while running are harmless."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop.clear()
            self._last_read_ns = None
            self._last_packet_ns = None
            self._publish(
                AudioLevelSnapshot(
                    status=AudioReadStatus.MISSING,
                    observed_at_ns=None,
                    observation=None,
                    error="starting",
                )
            )
            self._thread = Thread(
                target=self._run,
                name="part-a-audio-level",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop polling and close the source; repeated cleanup is harmless."""

        with self._lifecycle_lock:
            thread = self._thread
            self._stop.set()

        # Closing from the caller can unblock a device read.  The production
        # adapter is idempotent; the worker also closes in its finally block.
        try:
            self._microphone.close()
        except Exception as error:  # Preserve health as data; do not log history.
            self._publish_error(
                AudioReadStatus.DISCONNECTED,
                observed_at_ns=self.snapshot().observed_at_ns,
                reason=f"microphone_close_failed:{type(error).__name__}",
            )

        if thread is not None and thread is not current_thread():
            thread.join(timeout=self._close_timeout_seconds)
            if thread.is_alive():
                self._publish_error(
                    AudioReadStatus.DISCONNECTED,
                    observed_at_ns=self.snapshot().observed_at_ns,
                    reason="audio_level_thread_did_not_stop",
                )

        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def __enter__(self) -> AudioLevelPoller:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _run(self) -> None:
        try:
            try:
                self._microphone.open()
            except Exception as error:
                self._publish_error(
                    AudioReadStatus.DISCONNECTED,
                    observed_at_ns=None,
                    reason=f"microphone_open_failed:{type(error).__name__}",
                )
                return

            while not self._stop.is_set():
                try:
                    read = self._microphone.read()
                except Exception as error:
                    if not self._stop.is_set():
                        self._publish_error(
                            AudioReadStatus.DISCONNECTED,
                            observed_at_ns=None,
                            reason=f"microphone_read_failed:{type(error).__name__}",
                        )
                    return
                failed = read.status in {
                    AudioReadStatus.MISSING,
                    AudioReadStatus.DISCONNECTED,
                }
                self._consume(read)
                # Do not retain a packet/sample array while the next blocking
                # device read is in progress.
                del read
                if (
                    failed
                    and self._stop.wait(self._failure_backoff_seconds)
                ):
                    break
        finally:
            try:
                self._microphone.close()
            except Exception as error:
                self._publish_error(
                    AudioReadStatus.DISCONNECTED,
                    observed_at_ns=self.snapshot().observed_at_ns,
                    reason=f"microphone_close_failed:{type(error).__name__}",
                )

    def _consume(self, read: AudioRead) -> None:
        observed_at_ns = read.observed_at_ns
        if observed_at_ns < 0 or (
            self._last_read_ns is not None and observed_at_ns <= self._last_read_ns
        ):
            self._publish_error(
                AudioReadStatus.MISSING,
                observed_at_ns=(observed_at_ns if observed_at_ns >= 0 else None),
                reason="non_increasing_audio_read_timestamp",
            )
            return
        self._last_read_ns = observed_at_ns

        if read.status in {AudioReadStatus.MISSING, AudioReadStatus.DISCONNECTED}:
            self._publish_error(
                read.status,
                observed_at_ns=observed_at_ns,
                reason=read.reason or f"audio_read_{read.status.value}",
            )
            return

        packet = read.packet
        if packet is None:
            self._publish_error(
                AudioReadStatus.MISSING,
                observed_at_ns=observed_at_ns,
                reason="audio_packet_missing_for_success_status",
            )
            return
        if packet.captured_at_ns > observed_at_ns or (
            self._last_packet_ns is not None
            and packet.captured_at_ns <= self._last_packet_ns
        ):
            self._publish_error(
                AudioReadStatus.MISSING,
                observed_at_ns=observed_at_ns,
                reason="invalid_audio_packet_timestamp",
            )
            return
        self._last_packet_ns = packet.captured_at_ns

        duration_ns = max(
            1,
            round(packet.sample_count * 1_000_000_000 / packet.sample_rate_hz),
        )
        window_started_at_ns = packet.captured_at_ns - duration_ns
        if window_started_at_ns < 0:
            self._publish_error(
                AudioReadStatus.MISSING,
                observed_at_ns=observed_at_ns,
                reason="invalid_audio_window_timestamp",
            )
            return

        try:
            observation = self._calculator.observe(
                packet.samples,
                source_id=packet.source_id,
                window_started_at_ns=window_started_at_ns,
                window_ended_at_ns=packet.captured_at_ns,
                sample_rate_hz=packet.sample_rate_hz,
            )
        except Exception as error:
            self._publish_error(
                AudioReadStatus.MISSING,
                observed_at_ns=observed_at_ns,
                reason=f"audio_level_failed:{type(error).__name__}",
            )
            return

        if observation.state is not ObservationState.VALID:
            self._publish(
                AudioLevelSnapshot(
                    status=AudioReadStatus.MISSING,
                    observed_at_ns=observed_at_ns,
                    observation=observation,
                    error=observation.reason or "invalid_audio_level",
                )
            )
            return
        self._publish(
            AudioLevelSnapshot(
                status=read.status,
                observed_at_ns=observed_at_ns,
                observation=observation,
                error=(read.reason if read.status is AudioReadStatus.DEGRADED else None),
            )
        )

    def _publish_error(
        self,
        status: AudioReadStatus,
        *,
        observed_at_ns: int | None,
        reason: str,
    ) -> None:
        self._publish(
            AudioLevelSnapshot(
                status=status,
                observed_at_ns=observed_at_ns,
                observation=None,
                error=reason,
            )
        )

    def _publish(self, snapshot: AudioLevelSnapshot) -> None:
        with self._snapshot_lock:
            self._latest = snapshot
