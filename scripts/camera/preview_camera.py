"""Display a live camera preview without recording or saving frames."""

from __future__ import annotations

import argparse

from poker_dealer.io.camera import CameraConfig
from poker_dealer.io.camera.diagnostics import run_camera_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CameraConfig(
        device_index=args.index,
        source_id="table_camera",
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return run_camera_preview(
        config,
        title="Poker Dealer - Table Camera (Q/Esc to close)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
