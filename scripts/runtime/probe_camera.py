"""Probe bounded camera indexes and print machine-readable capture results."""

from __future__ import annotations

import argparse
import json

from deskmate_advance.perception.camera.diagnostics import probe_camera_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find readable OpenCV camera indexes on Windows."
    )
    parser.add_argument("--indexes", default="0,1,2,3", help="comma-separated indexes")
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indexes = [int(value.strip()) for value in args.indexes.split(",") if value.strip()]
    reports = probe_camera_indices(
        indexes,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    for report in reports:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if any(report["read_status"] == "ok" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
