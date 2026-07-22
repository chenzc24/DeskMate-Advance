"""Create an unvalidated 13-slot geometry from one target-camera still image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from poker_dealer.domain import VisionSlot
from poker_dealer.perception.cards import CardSlotGeometryConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    return parser.parse_args(argv)


def normalized_roi(
    roi: tuple[int, int, int, int], *, width: int, height: int
) -> dict[str, float]:
    x, y, roi_width, roi_height = roi
    if width <= 0 or height <= 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("image and selected ROI dimensions must be positive")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError("selected ROI is outside the image")
    return {
        "x_min": x / width,
        "y_min": y / height,
        "x_max": (x + roi_width) / width,
        "y_max": (y + roi_height) / height,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.calibration_id.strip():
        raise ValueError("calibration ID is required")
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"could not read calibration image: {args.image}")
    height, width = image.shape[:2]
    slots: dict[str, dict[str, float]] = {}
    window = "Poker Dealer 13-slot calibration"
    try:
        for slot in VisionSlot:
            preview = image.copy()
            cv2.putText(
                preview,
                f"Select {slot.value}; Enter accepts, Esc cancels",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            roi = tuple(
                int(value)
                for value in cv2.selectROI(window, preview, showCrosshair=True)
            )
            slots[slot.value] = normalized_roi(roi, width=width, height=height)
    finally:
        cv2.destroyWindow(window)
    payload = {
        "schema_version": "1.0",
        "calibration_id": args.calibration_id,
        "target_geometry_validated": False,
        "slots": slots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    CardSlotGeometryConfig.from_json(args.output)
    print(
        json.dumps(
            {
                "type": "card_slot_calibration",
                "output": str(args.output),
                "calibration_id": args.calibration_id,
                "slots": len(slots),
                "target_geometry_validated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
