from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.perception.ergonomics import (
    FaceLandmarkerAdapter,
    FaceLandmarkerConfig,
    ObservationState,
    PoseLandmarkerAdapter,
    PoseLandmarkerConfig,
)


def _frame(*, sequence_id: int = 0, captured_at_ns: int = 1_000_000) -> FramePacket:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]  # BGR -> RGB should become [3, 2, 1].
    return FramePacket(
        sequence_id=sequence_id,
        captured_at_ns=captured_at_ns,
        source_id="fixture",
        device_index=0,
        width=3,
        height=2,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=image,
    )


def _landmark(index: int, *, pose: bool) -> SimpleNamespace:
    values = {
        "x": index / 1000,
        "y": index / 500,
        "z": -index / 1000,
    }
    if pose:
        values.update(visibility=0.9, presence=0.8)
    return SimpleNamespace(**values)


class FakeTask:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.timestamps: list[int] = []
        self.first_pixel: list[int] | None = None
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        self.first_pixel = image.numpy_view()[0, 0].tolist()
        return self.result

    def close(self) -> None:
        self.closed = True


def test_pose_adapter_converts_framework_result_and_color_order() -> None:
    normalized = [_landmark(index, pose=True) for index in range(33)]
    world = [_landmark(index, pose=True) for index in range(33)]
    task = FakeTask(
        SimpleNamespace(
            pose_landmarks=[normalized],
            pose_world_landmarks=[world],
        )
    )
    adapter = PoseLandmarkerAdapter(
        PoseLandmarkerConfig(asset_path=Path("unused.task")),
        task=task,
    )

    observation = adapter.observe(_frame(captured_at_ns=2_500_000))

    assert observation.state is ObservationState.VALID
    assert len(observation.landmarks) == 33
    assert len(observation.world_landmarks) == 33
    assert observation.landmarks[1].visibility == 0.9
    assert task.timestamps == [2]
    assert task.first_pixel == [3, 2, 1]
    assert observation.context.source_id == "fixture"
    assert observation.context.inference_ms >= 0


def test_pose_adapter_marks_missing_and_rejects_repeated_millisecond() -> None:
    task = FakeTask(
        SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    )
    adapter = PoseLandmarkerAdapter(
        PoseLandmarkerConfig(asset_path=Path("unused.task")),
        task=task,
    )

    missing = adapter.observe(_frame(captured_at_ns=2_000_000))
    repeated = adapter.observe(
        _frame(sequence_id=1, captured_at_ns=2_900_000)
    )

    assert missing.state is ObservationState.MISSING
    assert missing.reason == "pose_not_detected"
    assert repeated.state is ObservationState.ERROR
    assert repeated.reason == "non_monotonic_millisecond_timestamp"
    assert task.timestamps == [2]


def test_face_adapter_converts_landmarks_blendshapes_and_matrix() -> None:
    task = FakeTask(
        SimpleNamespace(
            face_landmarks=[
                [_landmark(index, pose=False) for index in range(478)]
            ],
            face_blendshapes=[
                [SimpleNamespace(category_name="eyeBlinkLeft", score=0.75)]
            ],
            facial_transformation_matrixes=[np.eye(4, dtype=np.float32)],
        )
    )
    adapter = FaceLandmarkerAdapter(
        FaceLandmarkerConfig(asset_path=Path("unused.task")),
        task=task,
    )

    observation = adapter.observe(_frame(captured_at_ns=10_000_000))

    assert observation.state is ObservationState.VALID
    assert len(observation.landmarks) == 478
    assert observation.landmarks[0].visibility is None
    assert observation.blendshapes[0].name == "eyeBlinkLeft"
    assert observation.blendshapes[0].score == 0.75
    assert observation.transformation_matrix[0] == (1.0, 0.0, 0.0, 0.0)


def test_adapter_close_forwards_to_task() -> None:
    task = FakeTask(
        SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    )
    adapter = PoseLandmarkerAdapter(
        PoseLandmarkerConfig(asset_path=Path("unused.task")),
        task=task,
    )

    adapter.close()

    assert task.closed is True
