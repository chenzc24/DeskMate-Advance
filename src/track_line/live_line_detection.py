"""Run OpenCV guide-line detection on a laptop camera, video or image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

try:
    from .config import LineDetectorConfig
    from .detector import OpenCVLineDetector
    from .visualization import colorize_mask, render_debug
except ImportError:  # Support: python src/track_line/live_line_detection.py
    from config import LineDetectorConfig
    from detector import OpenCVLineDetector
    from visualization import colorize_mask, render_debug


HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="camera index such as 0, video path, image path or stream URL",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.white_on_green.json",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open OpenCV windows",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N frames; zero means run until input ends",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="print one JSON observation every N frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional annotated image or MP4 output path",
    )
    return parser.parse_args()


def _source_value(source: str) -> int | str:
    stripped = source.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped


def _emit_observation(result) -> None:
    print(
        json.dumps(
            result.observation.as_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _run_image(
    source: Path,
    detector: OpenCVLineDetector,
    *,
    output: Path | None,
    headless: bool,
) -> int:
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot decode image: {source}")
    result = detector.detect(frame, frame_index=0)
    annotated = render_debug(frame, result)
    _emit_observation(result)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), annotated):
            raise OSError(f"cannot write annotated image: {output}")
        print(f"saved {output.resolve()}")
    if not headless:
        cv2.imshow("track-line", annotated)
        cv2.imshow("track-line mask", colorize_mask(result, frame.shape[1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_capture(
    source: int | str,
    detector: OpenCVLineDetector,
    *,
    output: Path | None,
    headless: bool,
    max_frames: int,
    print_every: int,
) -> int:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source: {source}")

    writer: cv2.VideoWriter | None = None
    frame_index = 0
    paused = False
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                result = detector.detect(
                    frame,
                    frame_index=frame_index,
                    timestamp_ns=time.monotonic_ns(),
                )
                annotated = render_debug(frame, result)

                if print_every > 0 and frame_index % print_every == 0:
                    _emit_observation(result)

                if output is not None:
                    if writer is None:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        if not 1 <= fps <= 240:
                            fps = 20.0
                        writer = cv2.VideoWriter(
                            str(output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (annotated.shape[1], annotated.shape[0]),
                        )
                        if not writer.isOpened():
                            raise OSError(f"cannot create video: {output}")
                    writer.write(annotated)

                if not headless:
                    cv2.imshow("track-line", annotated)
                    cv2.imshow(
                        "track-line mask",
                        colorize_mask(result, frame.shape[1]),
                    )

                frame_index += 1
                if max_frames > 0 and frame_index >= max_frames:
                    break

            if headless:
                if paused:
                    paused = False
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError("source opened but produced no frames")
    print(f"processed_frames={frame_index}")
    return 0


def main() -> int:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")

    config = LineDetectorConfig.from_json(args.config)
    detector = OpenCVLineDetector(config)
    source_value = _source_value(args.source)

    if isinstance(source_value, str):
        source_path = Path(source_value)
        if source_path.suffix.lower() in IMAGE_SUFFIXES and source_path.is_file():
            return _run_image(
                source_path,
                detector,
                output=args.output,
                headless=args.headless,
            )

    return _run_capture(
        source_value,
        detector,
        output=args.output,
        headless=args.headless,
        max_frames=args.max_frames,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
