"""Run a bounded, non-recording Laptop-camera hand-gesture pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from poker_dealer.domain import ActionEvidenceState, Seat
from poker_dealer.io.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GesturePilotConfig,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/actions_laptop_pilot.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--focus-seat", choices=tuple(seat.value for seat in Seat), default=Seat.A.value)
    parser.add_argument("--hand-id", default="laptop-gesture-pilot")
    parser.add_argument("--state-version", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot = GesturePilotConfig.from_json(args.config)
    camera_values = pilot.camera
    camera_config = CameraConfig(
        device_index=(
            int(camera_values["device_index"])
            if args.index is None
            else args.index
        ),
        source_id="laptop_gesture_pilot",
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

    context = ActionObservationContext(
        hand_id=args.hand_id,
        expected_state_version=args.state_version,
        focus_seat=Seat(args.focus_seat),
    )
    temporal = GestureTemporalAdapter(pilot)
    frames = 0
    missing = 0
    hand_frames = 0
    out_of_roi_frames = 0
    candidates = 0
    evidence_counts = {state.value: 0 for state in ActionEvidenceState}
    candidate_counts = {action.value: 0 for action in pilot.gesture_to_action.values()}
    latencies: list[float] = []
    started_ns = time.monotonic_ns()
    last_evidence = None
    negotiated: dict[str, int | float | str] = {
        "device_index": camera_config.device_index,
        "source_id": camera_config.source_id,
        "backend": camera_config.backend,
        "width": camera_config.width or 0,
        "height": camera_config.height or 0,
        "nominal_fps": camera_config.fps or 0.0,
    }

    try:
        with OpenCVCamera(camera_config) as camera, MediaPipeGestureAdapter(
            pilot
        ) as model:
            negotiated = camera.negotiated_properties()
            while (time.monotonic_ns() - started_ns) / 1_000_000_000 < max_seconds:
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                read = camera.read()
                if read.status is not CameraReadStatus.OK or read.frame is None:
                    missing += 1
                    if read.status is CameraReadStatus.DISCONNECTED:
                        break
                    continue
                frames += 1
                evidence = model.recognize(read.frame)
                last_evidence = evidence
                if evidence.inference_latency_ms is not None:
                    latencies.append(evidence.inference_latency_ms)
                hand_frames += int(evidence.hand_present)
                out_of_roi_frames += int(
                    evidence.hand_present and not evidence.hand_in_focus_roi
                )
                observation = temporal.process(evidence, context)
                evidence_counts[observation.evidence_state.value] += 1
                if observation.evidence_state is ActionEvidenceState.CANDIDATE:
                    candidates += 1
                    assert observation.candidate_action is not None
                    candidate_counts[observation.candidate_action.value] += 1
                if args.emit_all or observation.evidence_state is ActionEvidenceState.CANDIDATE:
                    print(
                        json.dumps(
                            {
                                "type": "action_observation",
                                **observation_to_dict(observation),
                            },
                            ensure_ascii=False,
                        )
                    )

                if not args.headless:
                    import cv2

                    display = read.frame.image.copy()
                    height, width = display.shape[:2]
                    roi = pilot.focus_roi
                    cv2.rectangle(
                        display,
                        (int(roi.x_min * width), int(roi.y_min * height)),
                        (int(roi.x_max * width), int(roi.y_max * height)),
                        (0, 255, 0),
                        2,
                    )
                    label = evidence.gesture_label or "no-hand"
                    score = evidence.gesture_score or 0.0
                    cv2.putText(
                        display,
                        f"{label} {score:.2f} -> {observation.evidence_state.value}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Poker Dealer - Gesture Pilot (Q/Esc)", display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            if not args.headless:
                import cv2

                cv2.destroyAllWindows()
    except CameraError as exc:
        print(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
        return 2

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    summary = {
        "type": "summary",
        "status": "completed" if frames else "no_readable_frames",
        "pilot_status": pilot.pilot_status,
        "model_id": pilot.model.model_id,
        "model_version": pilot.model.version,
        "model_sha256": pilot.model.sha256,
        "camera": negotiated,
        "elapsed_seconds": elapsed_s,
        "frames": frames,
        "missing_reads": missing,
        "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
        "hand_frames": hand_frames,
        "out_of_roi_frames": out_of_roi_frames,
        "candidates": candidates,
        "evidence_counts": evidence_counts,
        "candidate_counts": candidate_counts,
        "inference_latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p95": _p95(latencies),
            "maximum": max(latencies) if latencies else None,
        },
        "last_gesture_label": (
            last_evidence.gesture_label if last_evidence is not None else None
        ),
        "frames_saved": 0,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
