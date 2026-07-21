"""Load the offline gesture asset and run one blank-frame inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket
from poker_dealer.perception.actions import (
    GesturePilotConfig,
    MediaPipeGestureAdapter,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/actions_laptop_pilot.json",
    )
    parser.add_argument("--image", type=Path, help="Optional local public test image")
    args = parser.parse_args()
    config = GesturePilotConfig.from_json(args.config)
    if args.image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        source_id = "synthetic_blank"
    else:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read image: {args.image}")
        source_id = "public_test_image"
    image = np.ascontiguousarray(image)
    image.setflags(write=False)
    height, width = image.shape[:2]
    frame = FramePacket(
        sequence_id=0,
        captured_at_ns=1_000_000,
        source_id=source_id,
        device_index=0,
        width=width,
        height=height,
        color_space=ColorSpace.BGR,
        nominal_fps=30.0,
        dropped_before=0,
        image=image,
    )
    with MediaPipeGestureAdapter(config) as adapter:
        evidence = adapter.recognize(frame)
    print(
        json.dumps(
            {
                "model_id": config.model.model_id,
                "model_version": config.model.version,
                "sha256": config.verify_model_asset(),
                "hand_present": evidence.hand_present,
                "hand_in_focus_roi": evidence.hand_in_focus_roi,
                "gesture_label": evidence.gesture_label,
                "gesture_score": evidence.gesture_score,
                "inference_latency_ms": evidence.inference_latency_ms,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
