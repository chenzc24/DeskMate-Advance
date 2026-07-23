"""Display and count visible poker-chip detections from an HTTP MJPEG stream.

This is an ignored, development-only feasibility runner. It never saves frames,
changes the game ledger, or sends robot commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from poker_dealer.io.camera import CameraConfig, CameraReadStatus, OpenCVCamera

try:
    import openvino as ov
except ImportError as exc:  # pragma: no cover - operator environment diagnostic
    raise SystemExit(
        "OpenVINO is missing. Install it with: "
        ".venv\\Scripts\\python.exe -m pip install openvino==2026.2.1"
    ) from exc


ROOT = Path(__file__).resolve().parent
MODEL_DIR = (
    ROOT
    / "pretrained"
    / "Shiranai17-poker-chips-dice-openvino-int8"
)
MODEL_XML = MODEL_DIR / "best.xml"
MODEL_BIN = MODEL_DIR / "best.bin"
EXPECTED_XML_SHA256 = "41bef7a7203a6396c30983fcad954fdb7c36c4d60da984d9ab5473271c8474c2"
EXPECTED_BIN_SHA256 = "1b58a9cee827735e1dd7a3abef6f89c4b88a9e0f5a59e83678256a560844f137"
CLASS_NAMES = ("Dice", "PokerChip")
INPUT_SIZE = 640


@dataclass(frozen=True, slots=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    source_width: int
    source_height: int


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    confidence: float
    bbox_xywh: tuple[int, int, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model() -> None:
    for path, expected in (
        (MODEL_XML, EXPECTED_XML_SHA256),
        (MODEL_BIN, EXPECTED_BIN_SHA256),
    ):
        if not path.is_file():
            raise SystemExit(f"model asset is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise SystemExit(
                f"model SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
            )


def preprocess(image: np.ndarray) -> tuple[np.ndarray, Letterbox]:
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), Letterbox(
        scale, pad_x, pad_y, width, height
    )


def decode(
    output: np.ndarray,
    letterbox: Letterbox,
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> tuple[Detection, ...]:
    values = np.asarray(output, dtype=np.float32)
    if values.shape == (1, 6, 8400):
        values = values[0].T
    elif values.shape == (1, 8400, 6):
        values = values[0]
    else:
        raise RuntimeError(f"unexpected model output shape: {values.shape}")

    candidates: list[Detection] = []
    for row in values:
        class_id = int(np.argmax(row[4:]))
        confidence = float(row[4 + class_id])
        if confidence < confidence_threshold:
            continue
        center_x, center_y, width, height = (float(value) for value in row[:4])
        x1 = (center_x - width / 2 - letterbox.pad_x) / letterbox.scale
        y1 = (center_y - height / 2 - letterbox.pad_y) / letterbox.scale
        x2 = (center_x + width / 2 - letterbox.pad_x) / letterbox.scale
        y2 = (center_y + height / 2 - letterbox.pad_y) / letterbox.scale
        x1 = max(0.0, min(float(letterbox.source_width - 1), x1))
        y1 = max(0.0, min(float(letterbox.source_height - 1), y1))
        x2 = max(0.0, min(float(letterbox.source_width), x2))
        y2 = max(0.0, min(float(letterbox.source_height), y2))
        box_width = max(1, int(round(x2 - x1)))
        box_height = max(1, int(round(y2 - y1)))
        candidates.append(
            Detection(
                class_id,
                confidence,
                (int(round(x1)), int(round(y1)), box_width, box_height),
            )
        )

    kept: list[Detection] = []
    for class_id in range(len(CLASS_NAMES)):
        class_candidates = [item for item in candidates if item.class_id == class_id]
        if not class_candidates:
            continue
        indices = cv2.dnn.NMSBoxes(
            [item.bbox_xywh for item in class_candidates],
            [item.confidence for item in class_candidates],
            confidence_threshold,
            nms_iou_threshold,
        )
        for index in np.asarray(indices).reshape(-1):
            kept.append(class_candidates[int(index)])
    return tuple(sorted(kept, key=lambda item: item.confidence, reverse=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--stream-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--stream-read-timeout-ms", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence must be in (0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise SystemExit("--nms-iou must be in (0, 1]")
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    verify_model()
    core = ov.Core()
    model = core.read_model(MODEL_XML)
    compiled = core.compile_model(model, "CPU")
    output_layer = compiled.output(0)
    camera_config = CameraConfig(
        stream_url=args.stream_url,
        source_id="robot_mjpeg_chip_pilot",
        backend="auto",
        width=None,
        height=None,
        fps=None,
        open_timeout_ms=args.stream_open_timeout_ms,
        read_timeout_ms=args.stream_read_timeout_ms,
    )

    started_ns = time.monotonic_ns()
    frames = 0
    missing_reads = 0
    frames_with_chips = 0
    maximum_visible_chips = 0
    last_visible_chips = 0
    inference_latencies: list[float] = []
    negotiated: dict[str, int | float | str] = {}

    try:
        with OpenCVCamera(camera_config) as camera:
            negotiated = camera.negotiated_properties()
            while (time.monotonic_ns() - started_ns) / 1_000_000_000 < args.max_seconds:
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                read = camera.read()
                if read.status is not CameraReadStatus.OK or read.frame is None:
                    missing_reads += 1
                    if read.status is CameraReadStatus.DISCONNECTED:
                        break
                    continue
                frames += 1
                image = read.frame.image
                blob, letterbox = preprocess(image)
                inference_started = time.perf_counter_ns()
                output = compiled([blob])[output_layer]
                inference_ms = (time.perf_counter_ns() - inference_started) / 1_000_000
                inference_latencies.append(inference_ms)
                detections = decode(
                    output, letterbox, args.confidence, args.nms_iou
                )
                visible_chips = sum(item.class_id == 1 for item in detections)
                last_visible_chips = visible_chips
                frames_with_chips += int(visible_chips > 0)
                maximum_visible_chips = max(maximum_visible_chips, visible_chips)

                if args.emit_all or visible_chips:
                    print(
                        json.dumps(
                            {
                                "type": "chip_frame_evidence",
                                "sequence_id": read.frame.sequence_id,
                                "visible_chip_count": visible_chips,
                                "dice_count": sum(
                                    item.class_id == 0 for item in detections
                                ),
                                "detections": [
                                    {
                                        "class": CLASS_NAMES[item.class_id],
                                        "confidence": item.confidence,
                                        "bbox_xywh": list(item.bbox_xywh),
                                    }
                                    for item in detections
                                ],
                                "inference_latency_ms": inference_ms,
                            },
                            ensure_ascii=False,
                        )
                    )

                if args.headless:
                    continue
                display = image.copy()
                for item in detections:
                    x, y, width, height = item.bbox_xywh
                    color = (70, 220, 70) if item.class_id == 1 else (0, 180, 255)
                    cv2.rectangle(display, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(
                        display,
                        f"{CLASS_NAMES[item.class_id]} {item.confidence:.2f}",
                        (x, max(20, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                cv2.putText(
                    display,
                    f"Visible chips: {visible_chips}",
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (70, 220, 70),
                    2,
                )
                cv2.putText(
                    display,
                    "Development count only | no denomination/value | Q/Esc quit",
                    (18, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Poker Dealer - Chip Count Pilot", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    summary = {
        "type": "summary",
        "status": "completed" if frames else "no_readable_frames",
        "model_status": "development_feasibility_only",
        "classes": list(CLASS_NAMES),
        "camera": negotiated,
        "frames": frames,
        "missing_reads": missing_reads,
        "elapsed_seconds": elapsed_s,
        "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
        "frames_with_visible_chips": frames_with_chips,
        "maximum_visible_chip_count": maximum_visible_chips,
        "last_visible_chip_count": last_visible_chips,
        "mean_inference_latency_ms": (
            sum(inference_latencies) / len(inference_latencies)
            if inference_latencies
            else None
        ),
        "frames_saved": 0,
        "game_state_mutated": False,
        "robot_connected": False,
        "can_estimate_value": False,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
