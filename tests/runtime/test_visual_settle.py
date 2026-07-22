from __future__ import annotations

import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket
from poker_dealer.runtime import (
    VisualSettleGate,
    VisualSettlePolicy,
    VisualSettleState,
)


def frame(sequence: int, timestamp_ms: int, value: int = 0) -> FramePacket:
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    return FramePacket(
        sequence_id=sequence,
        captured_at_ns=timestamp_ms * 1_000_000,
        source_id="test",
        device_index=0,
        width=64,
        height=48,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=image,
    )


def test_flushes_pre_ack_frames_and_requires_stable_new_frames() -> None:
    gate = VisualSettleGate(
        VisualSettlePolicy(
            minimum_new_frames=2,
            stable_frames=2,
            maximum_mean_absdiff=1.0,
            timeout_ms=1000,
            sample_width=32,
            sample_height=24,
        )
    )
    gate.begin(started_at_ns=100_000_000, sequence_watermark=5, camera_epoch=0)
    old = gate.observe(frame(5, 110), camera_epoch=0)
    assert old.reason == "pre_ack_frame_ignored"
    assert old.new_frames == 0
    assert gate.observe(frame(6, 120), camera_epoch=0).state is VisualSettleState.FLUSHING
    assert gate.observe(frame(7, 130), camera_epoch=0).state is VisualSettleState.STABILIZING
    settled = gate.observe(frame(8, 140), camera_epoch=0)
    assert settled.state is VisualSettleState.SETTLED
    assert settled.stable_frames == 2


def test_motion_resets_stability_and_camera_epoch_fails_closed() -> None:
    gate = VisualSettleGate(
        VisualSettlePolicy(minimum_new_frames=1, stable_frames=2, timeout_ms=1000)
    )
    gate.begin(started_at_ns=100_000_000, sequence_watermark=1, camera_epoch=2)
    gate.observe(frame(2, 110, 0), camera_epoch=2)
    moving = gate.observe(frame(3, 120, 255), camera_epoch=2)
    assert moving.stable_frames == 0
    failed = gate.observe(frame(4, 130, 255), camera_epoch=3)
    assert failed.state is VisualSettleState.TIMED_OUT
    assert failed.reason == "camera_epoch_changed"
