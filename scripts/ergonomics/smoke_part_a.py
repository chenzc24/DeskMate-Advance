"""Run bounded Part A inference using local assets and synthetic signals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.perception.ergonomics import (
    AudioLevelCalculator,
    FaceLandmarkerAdapter,
    FaceLandmarkerConfig,
    LuminanceCalculator,
    PoseLandmarkerAdapter,
    PoseLandmarkerConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ergonomics/perception.json"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = FramePacket(
        sequence_id=0,
        captured_at_ns=1_000_000,
        source_id="synthetic",
        device_index=0,
        width=640,
        height=480,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=image,
    )

    pose_config = config["pose"]
    face_config = config["face"]
    with PoseLandmarkerAdapter(
        PoseLandmarkerConfig(
            asset_path=project_root / pose_config["primary_asset"],
            num_poses=pose_config["num_poses"],
            min_pose_detection_confidence=pose_config[
                "min_pose_detection_confidence"
            ],
            min_pose_presence_confidence=pose_config[
                "min_pose_presence_confidence"
            ],
            min_tracking_confidence=pose_config["min_tracking_confidence"],
        )
    ) as pose, FaceLandmarkerAdapter(
        FaceLandmarkerConfig(
            asset_path=project_root / face_config["asset"],
            num_faces=face_config["num_faces"],
            output_face_blendshapes=face_config["output_face_blendshapes"],
            output_facial_transformation_matrixes=face_config[
                "output_facial_transformation_matrixes"
            ],
            min_face_detection_confidence=face_config[
                "min_face_detection_confidence"
            ],
            min_face_presence_confidence=face_config[
                "min_face_presence_confidence"
            ],
            min_tracking_confidence=face_config["min_tracking_confidence"],
        )
    ) as face:
        pose_observation = pose.observe(frame)
        face_observation = face.observe(frame)

    luminance = LuminanceCalculator().observe(frame)
    audio = AudioLevelCalculator().observe(
        np.zeros(1600, dtype=np.float32),
        source_id="synthetic_microphone",
        window_started_at_ns=1,
        window_ended_at_ns=100_000_001,
        sample_rate_hz=16_000,
    )
    for name, observation in (
        ("pose", pose_observation),
        ("face", face_observation),
        ("luminance", luminance),
        ("audio_level", audio),
    ):
        payload = asdict(observation)
        payload["component"] = name
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
