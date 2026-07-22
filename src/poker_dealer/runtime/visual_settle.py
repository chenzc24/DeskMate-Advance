"""Camera-frame settling gate after a successful physical rotation ACK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from poker_dealer.domain import FramePacket


class VisualSettleState(StrEnum):
    IDLE = "idle"
    FLUSHING = "flushing"
    STABILIZING = "stabilizing"
    SETTLED = "settled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class VisualSettlePolicy:
    minimum_new_frames: int = 2
    stable_frames: int = 4
    maximum_mean_absdiff: float = 4.0
    timeout_ms: int = 3000
    sample_width: int = 160
    sample_height: int = 90

    def __post_init__(self) -> None:
        if self.minimum_new_frames < 1 or self.stable_frames < 1:
            raise ValueError("visual settle frame counts must be positive")
        if self.maximum_mean_absdiff < 0 or self.timeout_ms <= 0:
            raise ValueError("visual settle thresholds are invalid")
        if self.sample_width <= 0 or self.sample_height <= 0:
            raise ValueError("visual settle sample dimensions must be positive")


@dataclass(frozen=True, slots=True)
class VisualSettleObservation:
    state: VisualSettleState
    sequence_id: int
    new_frames: int
    stable_frames: int
    mean_absdiff: float | None
    reason: str


class VisualSettleGate:
    def __init__(self, policy: VisualSettlePolicy | None = None) -> None:
        self.policy = policy or VisualSettlePolicy()
        self.state = VisualSettleState.IDLE
        self._started_at_ns: int | None = None
        self._sequence_watermark = -1
        self._camera_epoch = 0
        self._new_frames = 0
        self._stable_frames = 0
        self._previous: np.ndarray | None = None

    def begin(
        self,
        *,
        started_at_ns: int,
        sequence_watermark: int,
        camera_epoch: int,
    ) -> None:
        if started_at_ns < 0 or sequence_watermark < 0 or camera_epoch < 0:
            raise ValueError("visual settle start context is invalid")
        self.state = VisualSettleState.FLUSHING
        self._started_at_ns = started_at_ns
        self._sequence_watermark = sequence_watermark
        self._camera_epoch = camera_epoch
        self._new_frames = 0
        self._stable_frames = 0
        self._previous = None

    def observe(
        self, frame: FramePacket, *, camera_epoch: int
    ) -> VisualSettleObservation:
        if self._started_at_ns is None or self.state is VisualSettleState.IDLE:
            raise ValueError("visual settle gate has not started")
        if camera_epoch != self._camera_epoch:
            self.state = VisualSettleState.TIMED_OUT
            return self._result(frame.sequence_id, None, "camera_epoch_changed")
        if frame.captured_at_ns - self._started_at_ns > self.policy.timeout_ms * 1_000_000:
            self.state = VisualSettleState.TIMED_OUT
            return self._result(frame.sequence_id, None, "visual_settle_timeout")
        if frame.sequence_id <= self._sequence_watermark:
            return self._result(frame.sequence_id, None, "pre_ack_frame_ignored")

        self._new_frames += 1
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        sample = cv2.resize(
            gray,
            (self.policy.sample_width, self.policy.sample_height),
            interpolation=cv2.INTER_AREA,
        )
        difference: float | None = None
        if self._previous is not None:
            difference = float(
                np.mean(
                    cv2.absdiff(sample, self._previous),
                    dtype=np.float64,
                )
            )
        self._previous = sample
        if self._new_frames < self.policy.minimum_new_frames:
            self.state = VisualSettleState.FLUSHING
            return self._result(frame.sequence_id, difference, "flushing_post_ack_frames")

        self.state = VisualSettleState.STABILIZING
        if difference is not None and difference <= self.policy.maximum_mean_absdiff:
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        if self._stable_frames >= self.policy.stable_frames:
            self.state = VisualSettleState.SETTLED
            return self._result(frame.sequence_id, difference, "visual_scene_settled")
        return self._result(frame.sequence_id, difference, "visual_scene_stabilizing")

    def _result(
        self, sequence_id: int, difference: float | None, reason: str
    ) -> VisualSettleObservation:
        return VisualSettleObservation(
            self.state,
            sequence_id,
            self._new_frames,
            self._stable_frames,
            difference,
            reason,
        )

    def clear(self) -> None:
        self.state = VisualSettleState.IDLE
        self._started_at_ns = None
        self._previous = None
