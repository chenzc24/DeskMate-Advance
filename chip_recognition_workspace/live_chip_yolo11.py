"""Run the trained YOLO11 poker-chip detector on a live camera stream."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from chip_best_frame import BestFrameCandidate, ChipBestFrameSelector
from chip_live_value import ChipValueObservation, recognize_chip_value
from chip_template_matcher import ChipTemplateMatcher
from chip_value_tracker import ChipValueTracker
from poker_dealer.io.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = (
    ROOT
    / "models"
    / "assets"
    / "chip_recognition"
    / "yolo11n-localization-hard-negative-v3"
    / "best.pt"
)
DEFAULT_TEMPLATE_LIBRARY = (
    ROOT
    / "data"
    / "work"
    / "chips"
    / "2026-07-23-template-matching"
    / "library"
)
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_BACKEND = "msmf"
SINGLE_CLASS_ALIASES = {"pokerchip", "poker_chip", "chip"}
DENOMINATION_NAMES = {0: "chip_1", 1: "chip_5", 2: "chip_10", 3: "chip_20"}
DENOMINATION_VALUES = {"chip_1": 1, "chip_5": 5, "chip_10": 10, "chip_20": 20}
CLASS_COLORS = {
    "PokerChip": (70, 220, 70),
    "chip_1": (230, 230, 230),
    "chip_5": (90, 200, 255),
    "chip_10": (80, 80, 255),
    "chip_20": (255, 120, 80),
}


def _bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _run_value_batch(
    matcher: ChipTemplateMatcher,
    candidates: Sequence[BestFrameCandidate],
    minimum_minor_axis_px: float,
    minimum_aspect_ratio: float,
) -> tuple[tuple[ChipValueObservation, ...], float]:
    started = time.perf_counter_ns()
    results = tuple(
        replace(
            recognize_chip_value(
                matcher,
                candidate.image,
                candidate.local_bbox_xyxy,
                minimum_minor_axis_px=minimum_minor_axis_px,
                minimum_aspect_ratio=minimum_aspect_ratio,
            ),
            bbox_xyxy=candidate.source_bbox_xyxy,
            track_id=candidate.track_id,
            source_frame=candidate.source_frame,
            best_frame_quality=candidate.quality_score,
        )
        for candidate in candidates
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return results, elapsed_ms


def _attach_values(
    detections: list[dict[str, object]],
    cached_results: Sequence[ChipValueObservation],
    source_frame: int | None = None,
) -> None:
    for detection in detections:
        bbox = detection["bbox_xyxy"]
        best_match: ChipValueObservation | None = None
        best_iou = 0.0
        track_id = detection.get("track_id")
        for result in cached_results:
            if isinstance(track_id, int) and result.track_id == track_id:
                best_match = result
                best_iou = 1.0
                break
            overlap = _bbox_iou(bbox, result.bbox_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_match = result
        if best_match is None or best_iou < 0.55:
            detection.update(
                {
                    "denomination": None,
                    "value_score": None,
                    "value_margin": None,
                    "ellipse_quality": None,
                    "ellipse_aspect_ratio": None,
                    "ellipse_minor_axis_px": None,
                    "value_decision_reason": None,
                    "value_rejection_reason": "no_fresh_match",
                    "value_source_frame": None,
                    "best_frame_quality": None,
                    "raw_color_denomination": None,
                    "raw_color_score": None,
                    "raw_color_margin": None,
                    "digit_denomination": None,
                    "digit_score": None,
                    "digit_margin": None,
                }
            )
            continue
        detection.update(
            {
                "denomination": best_match.denomination,
                "value_score": best_match.score,
                "value_margin": best_match.margin,
                "ellipse_quality": best_match.ellipse_quality,
                "ellipse_aspect_ratio": best_match.ellipse_aspect_ratio,
                "ellipse_minor_axis_px": best_match.ellipse_minor_axis_px,
                "value_decision_reason": best_match.decision_reason,
                "value_rejection_reason": best_match.rejection_reason,
                "value_source_frame": (
                    best_match.source_frame
                    if best_match.source_frame is not None
                    else source_frame
                ),
                "best_frame_quality": best_match.best_frame_quality,
                "raw_color_denomination": best_match.raw_color_denomination,
                "raw_color_score": best_match.raw_color_score,
                "raw_color_margin": best_match.raw_color_margin,
                "digit_denomination": best_match.digit_denomination,
                "digit_score": best_match.digit_score,
                "digit_margin": best_match.digit_margin,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--template-library",
        type=Path,
        default=DEFAULT_TEMPLATE_LIBRARY,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    source.add_argument(
        "--stream-url",
        help="Optional HTTP(S) MJPEG stream; omit to use the laptop camera",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf"),
        default=DEFAULT_CAMERA_BACKEND,
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--value-score", type=float, default=0.58)
    parser.add_argument("--value-margin", type=float, default=0.035)
    parser.add_argument(
        "--value-interval",
        type=int,
        default=5,
        help="Deprecated compatibility option; use --best-frame-window",
    )
    parser.add_argument(
        "--best-frame-window",
        type=int,
        default=5,
        help="Choose one best raw frame from this many samples per track",
    )
    parser.add_argument("--value-cache-frames", type=int, default=8)
    parser.add_argument("--value-min-minor-axis", type=float, default=42.0)
    parser.add_argument("--value-min-aspect-ratio", type=float, default=0.38)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 runs until Q/Esc",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--stream-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--stream-read-timeout-ms", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"fine-tuned model is missing: {args.model}")
    if not (args.template_library / "manifest.json").is_file():
        raise SystemExit(
            f"template library is missing: {args.template_library}; "
            "run build_chip_templates.py first"
        )
    if args.camera_index < 0:
        raise SystemExit("--camera-index must be non-negative")
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence must be in (0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise SystemExit("--nms-iou must be in (0, 1]")
    if not 0.0 < args.value_score <= 1.0:
        raise SystemExit("--value-score must be in (0, 1]")
    if not 0.0 <= args.value_margin <= 1.0:
        raise SystemExit("--value-margin must be in [0, 1]")
    if args.value_interval <= 0:
        raise SystemExit("--value-interval must be positive")
    if args.best_frame_window <= 0:
        raise SystemExit("--best-frame-window must be positive")
    if args.value_cache_frames <= 0:
        raise SystemExit("--value-cache-frames must be positive")
    if args.value_min_minor_axis <= 0.0:
        raise SystemExit("--value-min-minor-axis must be positive")
    if not 0.0 < args.value_min_aspect_ratio <= 1.0:
        raise SystemExit("--value-min-aspect-ratio must be in (0, 1]")
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    model = YOLO(str(args.model.resolve()))
    names = model.names
    normalized_names = (
        {int(key): value for key, value in names.items()}
        if isinstance(names, dict)
        else dict(enumerate(names))
    )
    is_single_class_chip = (
        set(normalized_names) == {0}
        and str(normalized_names[0]).strip().lower() in SINGLE_CLASS_ALIASES
    )
    if not is_single_class_chip and normalized_names != DENOMINATION_NAMES:
        raise SystemExit(
            "refusing to run a non-chip or generic COCO weight; expected exactly "
            "one poker-chip class or the four denomination classes "
            "{0: chip_1, 1: chip_5, 2: chip_10, 3: chip_20}"
        )
    model.predict(
        np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
        conf=args.confidence,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    value_matcher = ChipTemplateMatcher(
        args.template_library,
        minimum_score=args.value_score,
        minimum_margin=args.value_margin,
    )
    value_tracker = ChipValueTracker()
    best_frame_selector = ChipBestFrameSelector(
        window_samples=args.best_frame_window,
    )
    value_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="chip-template-value"
    )
    value_future: (
        Future[tuple[tuple[ChipValueObservation, ...], float]] | None
    ) = None
    value_cache: tuple[ChipValueObservation, ...] = ()
    value_cache_received_frame = -args.value_cache_frames
    last_value_latency_ms: float | None = None

    is_network_stream = args.stream_url is not None
    camera_config = CameraConfig(
        device_index=args.camera_index,
        stream_url=args.stream_url,
        source_id=(
            "robot_mjpeg_chip_yolo11"
            if is_network_stream
            else f"laptop_chip_yolo11_camera_{args.camera_index}"
        ),
        backend="auto" if is_network_stream else args.backend,
        width=None if is_network_stream else args.width,
        height=None if is_network_stream else args.height,
        fps=None if is_network_stream else args.fps,
        open_timeout_ms=args.stream_open_timeout_ms,
        read_timeout_ms=args.stream_read_timeout_ms,
    )
    started_ns = time.monotonic_ns()
    frames = 0
    missing_reads = 0
    frames_with_chips = 0
    maximum_visible_chips = 0
    last_visible_chips = 0
    last_denomination_counts = {value: 0 for value in DENOMINATION_VALUES.values()}
    last_observed_visible_total = 0
    last_total_complete = False
    value_errors = 0
    latencies_ms: list[float] = []
    negotiated: dict[str, int | float | str] = {}
    camera_error: str | None = None

    try:
        with OpenCVCamera(camera_config) as camera:
            negotiated = camera.negotiated_properties()
            while True:
                elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
                if args.max_seconds and elapsed_s >= args.max_seconds:
                    break
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
                inference_started = time.perf_counter_ns()
                result = model.predict(
                    image,
                    conf=args.confidence,
                    iou=args.nms_iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                latency_ms = (time.perf_counter_ns() - inference_started) / 1_000_000
                latencies_ms.append(latency_ms)
                detections: list[dict[str, object]] = []
                if result.boxes is not None:
                    for xyxy, confidence, class_id in zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                    ):
                        class_index = int(class_id)
                        class_name = normalized_names.get(class_index)
                        if class_name is None:
                            continue
                        detections.append(
                            {
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox_xyxy": [int(round(value)) for value in xyxy],
                            }
                        )
                visible_chips = len(detections)
                last_visible_chips = visible_chips
                value_tracker.associate(frames, detections)
                best_frame_selector.observe(frames, image, detections)
                if value_future is not None and value_future.done():
                    try:
                        (
                            value_cache,
                            last_value_latency_ms,
                        ) = value_future.result()
                        value_cache_received_frame = frames
                    except Exception as exc:  # noqa: BLE001 - runtime diagnostic boundary
                        value_errors += 1
                        value_cache = ()
                        print(
                            json.dumps(
                                {"type": "chip_value_error", "reason": str(exc)},
                                ensure_ascii=False,
                            )
                        )
                    value_future = None
                if value_future is None:
                    ready_candidates = best_frame_selector.take_ready()
                    if ready_candidates:
                        value_future = value_executor.submit(
                            _run_value_batch,
                            value_matcher,
                            ready_candidates,
                            args.value_min_minor_axis,
                            args.value_min_aspect_ratio,
                        )
                if (
                    frames - value_cache_received_frame
                    <= args.value_cache_frames
                ):
                    _attach_values(detections, value_cache)
                else:
                    _attach_values(detections, ())
                value_tracker.ingest(detections)
                class_counts = {
                    class_name: sum(
                        detection["class"] == class_name for detection in detections
                    )
                    for class_name in normalized_names.values()
                }
                denomination_counts = {
                    value: sum(
                        detection["stable_denomination"] == value
                        for detection in detections
                    )
                    for value in DENOMINATION_VALUES.values()
                }
                recognized_denominations = sum(denomination_counts.values())
                observed_visible_total = sum(
                    value * count for value, count in denomination_counts.items()
                )
                total_complete = recognized_denominations == visible_chips
                last_denomination_counts = denomination_counts
                last_observed_visible_total = observed_visible_total
                last_total_complete = total_complete
                frames_with_chips += int(visible_chips > 0)
                maximum_visible_chips = max(maximum_visible_chips, visible_chips)
                if args.emit_all or visible_chips:
                    print(
                        json.dumps(
                            {
                                "type": "chip_frame_evidence",
                                "sequence_id": read.frame.sequence_id,
                                "visible_chip_count": visible_chips,
                                "class_counts": class_counts,
                                "denomination_counts": denomination_counts,
                                "observed_visible_total": observed_visible_total,
                                "recognized_denominations": recognized_denominations,
                                "total_complete": total_complete,
                                "detections": detections,
                                "inference_latency_ms": latency_ms,
                                "value_batch_latency_ms": last_value_latency_ms,
                            },
                            ensure_ascii=False,
                        )
                    )

                if args.headless:
                    continue
                display = image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    confidence = detection["confidence"]
                    class_name = str(detection["class"])
                    denomination = detection["stable_denomination"]
                    rejection_reason = detection["value_rejection_reason"]
                    value_state = str(detection["value_state"])
                    track_id = int(detection["track_id"])
                    color = CLASS_COLORS.get(
                        f"chip_{denomination}",
                        CLASS_COLORS.get(class_name, (70, 220, 70)),
                    )
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        (
                            f"T{track_id} chip {confidence:.2f} | "
                            f"value {denomination} stable"
                            if denomination is not None
                            else (
                                f"T{track_id} chip {confidence:.2f} | value ? "
                                f"{value_state if value_state else rejection_reason or ''}"
                            ).rstrip()
                        ),
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                cv2.putText(
                    display,
                    f"Visible chips: {visible_chips} | inference: {latency_ms:.1f} ms",
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (70, 220, 70),
                    2,
                )
                count_text = "  ".join(
                    f"{value}:{denomination_counts[value]}"
                    for value in DENOMINATION_VALUES.values()
                )
                total_suffix = "" if total_complete else " (partial)"
                cv2.putText(
                    display,
                    f"Values {count_text} | total: {observed_visible_total}{total_suffix}",
                    (18, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "YOLO + template evidence only | no ledger/robot action | Q/Esc quit",
                    (18, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Poker Dealer - YOLO11 Chip Pilot", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except CameraError as exc:
        camera_error = str(exc)
        print(
            json.dumps(
                {
                    "type": "camera_error",
                    "camera_index": None if is_network_stream else args.camera_index,
                    "stream_url": args.stream_url,
                    "reason": camera_error,
                },
                ensure_ascii=False,
            )
        )
    finally:
        value_executor.shutdown(wait=True, cancel_futures=True)
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    print(
        json.dumps(
            {
                "type": "summary",
                "status": (
                    "camera_error"
                    if camera_error is not None
                    else "completed" if frames else "no_readable_frames"
                ),
                "model_status": (
                    "chip-localization-yolo11n@hard-negative-v3-20260723"
                ),
                "model_path": str(args.model.resolve()),
                "camera_index": None if is_network_stream else args.camera_index,
                "stream_url": args.stream_url,
                "camera": negotiated,
                "frames": frames,
                "missing_reads": missing_reads,
                "elapsed_seconds": elapsed_s,
                "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
                "frames_with_visible_chips": frames_with_chips,
                "maximum_visible_chip_count": maximum_visible_chips,
                "last_visible_chip_count": last_visible_chips,
                "last_denomination_counts": last_denomination_counts,
                "last_observed_visible_total": last_observed_visible_total,
                "last_total_complete": last_total_complete,
                "mean_inference_latency_ms": (
                    sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
                ),
                "frames_saved": 0,
                "game_state_mutated": False,
                "robot_connected": False,
                "can_estimate_visible_value": True,
                "authoritative_ledger_value": False,
                "value_engine": "track-best-frame-raw-colour-digit-template-v2",
                "best_frame_window": args.best_frame_window,
                "value_min_minor_axis_px": args.value_min_minor_axis,
                "value_min_aspect_ratio": args.value_min_aspect_ratio,
                "value_cache_frames": args.value_cache_frames,
                "value_errors": value_errors,
                "last_value_batch_latency_ms": last_value_latency_ms,
                "camera_error": camera_error,
            },
            ensure_ascii=False,
        )
    )
    return 0 if frames and camera_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
