"""Run the separate YOLO + 10/20 outer-rim-colour live detector."""

from __future__ import annotations

import live_chip_yolo11 as live
from chip_rim_color_value import RimColourBinaryClassifier, recognize_chip_rim_colour


def main() -> int:
    live.recognize_chip_value = recognize_chip_rim_colour
    live.ChipTemplateMatcher = RimColourBinaryClassifier
    live.VALUE_ENGINE_NAME = "track-best-frame-rim-colour-binary-10-20-v1"
    live.DISPLAY_ENGINE_LABEL = "YOLO + rim colour 10/20"
    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())
