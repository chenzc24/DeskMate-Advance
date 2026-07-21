"""Run a bounded, non-recording fixed-ROI Laptop card-recognition pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from poker_dealer.domain import ObservationStatus, VisionSlot
from poker_dealer.io.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)
from poker_dealer.perception.cards import (
    CardFrameEvidence,
    CardModelError,
    CardObservationPromoter,
    CardPilotConfig,
    OpenCvCardRecognitionAdapter,
    card_observation_to_dict,
    crop_fixed_card_roi,
)


ROOT = Path(__file__).resolve().parents[2]


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _card_label(evidence: CardFrameEvidence | None) -> str:
    if evidence is None or evidence.card is None:
        return "unknown"
    return (
        f"{evidence.card.rank.value} {evidence.card.suit.value} "
        f"{evidence.confidence or 0.0:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/cards_lgd_pilot.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--slot",
        choices=tuple(slot.value for slot in VisionSlot),
        default=VisionSlot.BOARD_FLOP_1.value,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot = CardPilotConfig.from_json(args.config)
    camera_values = pilot.camera
    camera_config = CameraConfig(
        device_index=(
            int(camera_values["device_index"]) if args.index is None else args.index
        ),
        source_id="laptop_card_pilot",
        backend=(
            str(camera_values["backend"])
            if args.backend is None
            else args.backend
        ),
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fps=float(camera_values["fps"]),
    )
    max_seconds = (
        float(pilot.max_seconds_default)
        if args.max_seconds is None
        else args.max_seconds
    )
    if max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    slot = VisionSlot(args.slot)
    promoter = CardObservationPromoter(pilot)
    frames = 0
    missing_reads = 0
    raw_candidates = 0
    confirmed = 0
    status_counts = {status.value: 0 for status in ObservationStatus}
    latencies: list[float] = []
    last_evidence: CardFrameEvidence | None = None
    last_observation = None
    negotiated: dict[str, int | float | str] = {}
    started_ns = time.monotonic_ns()

    try:
        model = OpenCvCardRecognitionAdapter(pilot)
        with OpenCVCamera(camera_config) as camera:
            negotiated = camera.negotiated_properties()
            while (time.monotonic_ns() - started_ns) / 1_000_000_000 < max_seconds:
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                read = camera.read()
                if read.status is not CameraReadStatus.OK or read.frame is None:
                    missing_reads += 1
                    if read.status is CameraReadStatus.DISCONNECTED:
                        break
                    continue
                frames += 1
                cropped, pixel_roi = crop_fixed_card_roi(
                    read.frame, pilot.fixed_roi, slot
                )
                evidence = model.analyze(cropped)
                observation = promoter.process(slot, evidence)
                last_evidence = evidence
                last_observation = observation
                latencies.append(evidence.inference_latency_ms)
                raw_candidates += int(evidence.card is not None)
                confirmed += int(observation.status is ObservationStatus.CONFIRMED)
                status_counts[observation.status.value] += 1
                if args.emit_all or observation.status is ObservationStatus.CONFIRMED:
                    print(
                        json.dumps(
                            {
                                "type": "card_observation",
                                **card_observation_to_dict(observation),
                            },
                            ensure_ascii=False,
                        )
                    )

                if args.headless:
                    continue
                import cv2

                display = read.frame.image.copy()
                roi_color = (
                    (0, 220, 0)
                    if observation.status is ObservationStatus.CONFIRMED
                    else (0, 210, 255)
                )
                cv2.rectangle(
                    display,
                    (pixel_roi.x, pixel_roi.y),
                    (pixel_roi.x + pixel_roi.width, pixel_roi.y + pixel_roi.height),
                    roi_color,
                    3,
                )
                for detection in evidence.detections:
                    x, y, width, height = detection.bbox_xywh
                    x += pixel_roi.x
                    y += pixel_roi.y
                    cv2.rectangle(
                        display,
                        (x, y),
                        (x + width, y + height),
                        (255, 160, 40),
                        2,
                    )
                    cv2.putText(
                        display,
                        (
                            f"{detection.card.rank.value}"
                            f"/{detection.card.suit.value} {detection.confidence:.2f}"
                        ),
                        (x, max(20, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 160, 40),
                        2,
                    )
                flags = ",".join(evidence.quality_flags) or "none"
                cv2.putText(
                    display,
                    f"RAW {_card_label(evidence)}",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"{slot.value}: {observation.status.value} | flags: {flags}",
                    (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    roi_color,
                    2,
                )
                cv2.putText(
                    display,
                    "Place one face-up card inside ROI | Q/Esc quit | no frames saved",
                    (18, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Poker Dealer - Fixed ROI Card Pilot", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except (CameraError, CardModelError) as exc:
        print(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    finally:
        if not args.headless:
            import cv2

            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    summary = {
        "type": "summary",
        "status": "completed" if frames else "no_readable_frames",
        "pilot_status": pilot.pilot_status,
        "model_id": pilot.model.model_id,
        "model_version": pilot.model.version,
        "slot_id": slot.value,
        "roi_status": "laptop_fixture_not_target_geometry",
        "camera": negotiated,
        "elapsed_seconds": elapsed_s,
        "frames": frames,
        "missing_reads": missing_reads,
        "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
        "raw_candidate_frames": raw_candidates,
        "confirmed_observation_frames": confirmed,
        "status_counts": status_counts,
        "inference_latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p95": _p95(latencies),
            "maximum": max(latencies) if latencies else None,
        },
        "last_raw_card": _card_label(last_evidence),
        "last_observation": (
            card_observation_to_dict(last_observation)
            if last_observation is not None
            else None
        ),
        "frames_saved": 0,
        "game_state_mutated": False,
        "robot_connected": False,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
