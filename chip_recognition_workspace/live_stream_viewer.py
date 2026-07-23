"""Display raw MJPEG video and save an unmodified frame when X is pressed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2

from poker_dealer.io.camera import CameraConfig, CameraReadStatus, OpenCVCamera


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "chips" / "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stream-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--stream-read-timeout-ms", type=int, default=2000)
    return parser.parse_args()


def save_frame(frame, output_dir: Path, stream_url: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    captured_utc = datetime.now(timezone.utc)
    stem = (
        f"chip_{captured_utc.strftime('%Y%m%dT%H%M%S_%fZ')}"
        f"_seq{frame.sequence_id:08d}"
    )
    image_path = output_dir / f"{stem}.png"
    encoded_ok, encoded = cv2.imencode(".png", frame.image)
    if not encoded_ok:
        raise RuntimeError("OpenCV could not encode the current frame as PNG")
    image_bytes = encoded.tobytes()
    with image_path.open("xb") as stream:
        stream.write(image_bytes)

    record = {
        "schema_version": "1.0",
        "file": image_path.name,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "captured_utc": captured_utc.isoformat(),
        "captured_monotonic_ns": frame.captured_at_ns,
        "sequence_id": frame.sequence_id,
        "source_id": frame.source_id,
        "stream_url": stream_url,
        "width": frame.width,
        "height": frame.height,
        "color_space": str(frame.color_space),
        "nominal_fps": frame.nominal_fps,
        "dropped_before": frame.dropped_before,
        "image_modified": False,
        "inference_performed": False,
    }
    with (output_dir / "capture_manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return image_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config = CameraConfig(
        stream_url=args.stream_url,
        source_id="robot_mjpeg_raw_viewer",
        backend="auto",
        width=None,
        height=None,
        fps=None,
        open_timeout_ms=args.stream_open_timeout_ms,
        read_timeout_ms=args.stream_read_timeout_ms,
    )
    frames = 0
    frames_saved = 0
    try:
        with OpenCVCamera(config) as camera:
            while True:
                read = camera.read()
                if read.status is not CameraReadStatus.OK or read.frame is None:
                    if read.status is CameraReadStatus.DISCONNECTED:
                        break
                    continue
                frames += 1
                cv2.imshow("Raspberry Pi Camera - Raw Stream", read.frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("x"), ord("X")):
                    image_path = save_frame(read.frame, output_dir, args.stream_url)
                    frames_saved += 1
                    print(f"saved raw frame: {image_path}")
    finally:
        cv2.destroyAllWindows()
    print(
        f"raw viewer stopped: frames={frames}, inference=False, "
        f"frames_saved={frames_saved}, output_dir={output_dir}, "
        "robot_connected=False"
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
