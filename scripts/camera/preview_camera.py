"""Display a live camera preview without recording or saving frames."""

from __future__ import annotations

import argparse

from poker_dealer.io.camera import CameraConfig
from poker_dealer.io.camera.diagnostics import run_camera_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument(
        "--stream-url",
        help="HTTP(S) MJPEG stream; mutually exclusive with --index",
    )
    parser.add_argument("--stream-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--stream-read-timeout-ms", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stream_url is not None:
        if args.index is not None:
            raise SystemExit("--stream-url and --index are mutually exclusive")
        if args.backend not in {None, "auto"}:
            raise SystemExit("network streams use FFmpeg; omit --backend")
        config = CameraConfig(
            stream_url=args.stream_url,
            source_id="robot_mjpeg_stream",
            backend="auto",
            width=None,
            height=None,
            fps=None,
            open_timeout_ms=args.stream_open_timeout_ms,
            read_timeout_ms=args.stream_read_timeout_ms,
        )
    else:
        config = CameraConfig(
            device_index=0 if args.index is None else args.index,
            source_id="table_camera",
            backend="dshow" if args.backend is None else args.backend,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    return run_camera_preview(
        config,
        title="Poker Dealer - Camera Preview (Q/Esc to close)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
