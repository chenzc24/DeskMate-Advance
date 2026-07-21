"""Run the pinned offline card model on one local image or a blank frame."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from poker_dealer.domain import ColorSpace, FramePacket, VisionSlot
from poker_dealer.perception.cards import (
    CardObservationPromoter,
    CardPilotConfig,
    OpenCvCardRecognitionAdapter,
    card_observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]


def _card_dict(card: object | None) -> dict[str, str] | None:
    if card is None:
        return None
    return {"rank": card.rank.value, "suit": card.suit.value}  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/cards_lgd_pilot.json",
    )
    parser.add_argument("--image", type=Path, help="Local image; never modified or saved")
    parser.add_argument(
        "--slot",
        choices=[slot.value for slot in VisionSlot],
        default=VisionSlot.BOARD_FLOP_1.value,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Simulate repeated identical frames for temporal-promotion testing",
    )
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100:
        parser.error("--repeat must be in [1, 100]")
    if args.interval_ms < 0:
        parser.error("--interval-ms must be non-negative")

    if args.image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        source_id = "synthetic_blank"
    else:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read image: {args.image}")
        source_id = f"local_image:{args.image.name}"
    image = np.ascontiguousarray(image)
    image.setflags(write=False)
    height, width = image.shape[:2]
    frame = FramePacket(
        sequence_id=0,
        captured_at_ns=1_000_000_000,
        source_id=source_id,
        device_index=0,
        width=width,
        height=height,
        color_space=ColorSpace.BGR,
        nominal_fps=0.0,
        dropped_before=0,
        image=image,
    )

    config = CardPilotConfig.from_json(args.config)
    adapter = OpenCvCardRecognitionAdapter(config)
    raw = adapter.analyze(frame)
    promoter = CardObservationPromoter(config)
    statuses: list[str] = []
    observation = None
    for index in range(args.repeat):
        evidence = replace(
            raw,
            sequence_id=index,
            observed_at_ns=frame.captured_at_ns + index * args.interval_ms * 1_000_000,
        )
        observation = promoter.process(VisionSlot(args.slot), evidence)
        statuses.append(observation.status.value)
    assert observation is not None

    model_sha256, classes_sha256 = config.verify_assets()
    print(
        json.dumps(
            {
                "pilot_status": config.pilot_status,
                "model_id": config.model.model_id,
                "model_version": config.model.version,
                "model_sha256": model_sha256,
                "classes_sha256": classes_sha256,
                "input": {
                    "source": source_id,
                    "width": width,
                    "height": height,
                    "slot_id": args.slot,
                },
                "raw_model_evidence": {
                    "card": _card_dict(raw.card),
                    "confidence": raw.confidence,
                    "quality_flags": list(raw.quality_flags),
                    "inference_latency_ms": raw.inference_latency_ms,
                    "detections": [
                        {
                            "card": _card_dict(item.card),
                            "confidence": item.confidence,
                            "bbox_xywh": list(item.bbox_xywh),
                        }
                        for item in raw.detections
                    ],
                },
                "temporal_test": {
                    "simulated_identical_frames": args.repeat,
                    "interval_ms": args.interval_ms,
                    "statuses": statuses,
                    "final_observation": card_observation_to_dict(observation),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
