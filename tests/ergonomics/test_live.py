import numpy as np
import pytest

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.features.ergonomics import (
    LiveScheduleConfig,
    PartALiveEngine,
)
from deskmate_advance.perception.ergonomics import (
    FaceObservation,
    ObservationContext,
    ObservationState,
    PoseObservation,
)


def _frame(sequence_id: int, captured_at_ns: int, *, dropped: int = 0) -> FramePacket:
    return FramePacket(
        sequence_id=sequence_id,
        captured_at_ns=captured_at_ns,
        source_id="camera",
        device_index=0,
        width=2,
        height=2,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=dropped,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


class FakePose:
    def __init__(self, state: ObservationState = ObservationState.MISSING) -> None:
        self.state = state
        self.calls: list[int] = []

    def observe(self, frame: FramePacket) -> PoseObservation:
        self.calls.append(frame.captured_at_ns)
        return PoseObservation(
            context=ObservationContext(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=frame.captured_at_ns,
                model_id="pose",
                inference_ms=2.0,
                dropped_before=frame.dropped_before,
            ),
            state=self.state,
            reason="pose_not_detected" if self.state is ObservationState.MISSING else None,
        )


class FakeFace:
    def __init__(self, state: ObservationState = ObservationState.MISSING) -> None:
        self.state = state
        self.calls: list[int] = []

    def observe(self, frame: FramePacket) -> FaceObservation:
        self.calls.append(frame.captured_at_ns)
        return FaceObservation(
            context=ObservationContext(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=frame.captured_at_ns,
                model_id="face",
                inference_ms=3.0,
                dropped_before=frame.dropped_before,
            ),
            state=self.state,
            reason="face_not_detected" if self.state is ObservationState.MISSING else None,
        )


def test_live_engine_runs_models_at_independent_bounded_cadence() -> None:
    pose = FakePose()
    face = FakeFace()
    engine = PartALiveEngine(
        pose=pose,
        face=face,
        schedule=LiveScheduleConfig(pose_hz=10, face_hz=5),
    )

    snapshots = [
        engine.process(_frame(index, timestamp_ms * 1_000_000))
        for index, timestamp_ms in enumerate((0, 33, 66, 99, 132, 200))
    ]

    assert pose.calls == [0, 132_000_000]
    assert face.calls == [0, 200_000_000]
    assert [item.pose_ran for item in snapshots] == [True, False, False, False, True, False]
    assert [item.face_ran for item in snapshots] == [True, False, False, False, False, True]
    assert [item.luminance_ran for item in snapshots] == [True, False, False, False, False, False]
    assert all(item.luminance.mean == 0 for item in snapshots)


def test_live_engine_marks_cached_observation_stale_without_new_inference() -> None:
    engine = PartALiveEngine(
        pose=FakePose(),
        face=FakeFace(),
        schedule=LiveScheduleConfig(
            pose_hz=1,
            face_hz=1,
            stale_after_ms=50,
        ),
    )

    first = engine.process(_frame(0, 0))
    later = engine.process(_frame(1, 60_000_000))

    assert first.pose_stale is False
    assert later.pose_ran is False
    assert later.pose_age_ms == pytest.approx(60)
    assert later.pose_stale is True
    assert later.face_stale is True


def test_live_summary_reports_states_latency_drops_and_never_records() -> None:
    engine = PartALiveEngine(
        pose=FakePose(ObservationState.ERROR),
        face=FakeFace(ObservationState.MISSING),
        schedule=LiveScheduleConfig(pose_hz=10, face_hz=10),
    )
    engine.process(_frame(0, 0, dropped=2))
    engine.process(_frame(1, 100_000_000))

    summary = engine.summary()

    assert summary["records_media"] is False
    assert summary["frames"] == 2
    assert summary["effective_capture_fps"] == pytest.approx(10)
    assert summary["dropped_before_total"] == 2
    assert summary["pose"]["state_counts"]["error"] == 2
    assert summary["face"]["state_counts"]["missing"] == 2
    assert summary["pose"]["latency_ms"]["p95"] == pytest.approx(2.0)
    assert summary["luminance"]["runs"] == 1
    assert summary["latest_evidence"]["pose"]["state"] == "error"
    assert summary["latest_evidence"]["face"]["state"] == "missing"
    assert summary["latest_evidence"]["luminance"]["mean"] == 0


def test_live_engine_rejects_non_increasing_camera_timestamp() -> None:
    engine = PartALiveEngine(pose=FakePose(), face=FakeFace())
    engine.process(_frame(0, 1_000_000))

    with pytest.raises(ValueError, match="monotonically"):
        engine.process(_frame(1, 1_000_000))


def test_live_schedule_rejects_unbounded_metric_configuration() -> None:
    with pytest.raises(ValueError):
        LiveScheduleConfig(metric_reservoir_size=0)
    with pytest.raises(ValueError):
        LiveScheduleConfig(pose_hz=float("nan"))
    with pytest.raises(ValueError):
        LiveScheduleConfig(face_hz=1001)
