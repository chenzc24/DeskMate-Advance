"""Run a four-fixed-seat, multi-hand Laptop gesture attribution UI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from poker_dealer.domain import ActionEvidenceState, SEAT_ORDER, Seat
from poker_dealer.io.camera import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    MultiSeatGesturePilotConfig,
    SeatRoiRouter,
    observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]
SEAT_KEYS = {ord("1"): Seat.A, ord("2"): Seat.B, ord("3"): Seat.C, ord("4"): Seat.D}
SEAT_COLORS = {
    Seat.A: (255, 160, 40),
    Seat.B: (190, 80, 255),
    Seat.C: (40, 200, 255),
    Seat.D: (80, 220, 80),
}


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
        default=ROOT / "configs/perception/actions_multiseat_laptop_pilot.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--focus-seat", choices=tuple(seat.value for seat in Seat))
    parser.add_argument("--hand-id", default="laptop-multiseat-pilot")
    parser.add_argument("--state-version", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot = MultiSeatGesturePilotConfig.from_json(args.config)
    gesture = pilot.gesture
    camera_values = gesture.camera
    camera_config = CameraConfig(
        device_index=(
            int(camera_values["device_index"]) if args.index is None else args.index
        ),
        source_id="laptop_multiseat_action_pilot",
        backend=(
            str(camera_values["backend"]) if args.backend is None else args.backend
        ),
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fps=float(camera_values["fps"]),
    )
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    focus_seat = (
        pilot.initial_focus_seat if args.focus_seat is None else Seat(args.focus_seat)
    )
    state_version = args.state_version
    temporal = GestureTemporalAdapter(gesture)
    router = SeatRoiRouter(pilot.seat_rois)
    frames = 0
    missing = 0
    candidates = 0
    unassigned_hands = 0
    ambiguous_hands = 0
    hand_frames_by_seat = {seat.value: 0 for seat in SEAT_ORDER}
    candidates_by_seat = {seat.value: 0 for seat in SEAT_ORDER}
    latencies: list[float] = []
    last_status = "waiting for focused-seat hand"
    started_ns = time.monotonic_ns()

    try:
        with OpenCVCamera(camera_config) as camera, MediaPipeGestureAdapter(
            gesture
        ) as model:
            negotiated = camera.negotiated_properties()
            while (time.monotonic_ns() - started_ns) / 1_000_000_000 < args.max_seconds:
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                read = camera.read()
                if read.status is not CameraReadStatus.OK or read.frame is None:
                    missing += 1
                    if read.status is CameraReadStatus.DISCONNECTED:
                        break
                    continue
                frames += 1
                hands = model.recognize_all(read.frame)
                routed = router.route(hands)
                unassigned_hands += len(routed.unassigned)
                ambiguous_hands += len(routed.ambiguous)
                for seat, seat_hands in routed.assignments.items():
                    hand_frames_by_seat[seat.value] += int(bool(seat_hands))
                if model.last_inference_latency_ms is not None:
                    latencies.append(model.last_inference_latency_ms)
                focused = router.focus_evidence(
                    routed,
                    focus_seat,
                    observed_at_ns=read.frame.captured_at_ns,
                    inference_latency_ms=(
                        model.last_inference_latency_ms
                    ),
                )
                context = ActionObservationContext(
                    args.hand_id, state_version, focus_seat
                )
                observation = temporal.process(focused, context)
                last_status = observation.evidence_state.value
                if observation.evidence_state is ActionEvidenceState.CANDIDATE:
                    candidates += 1
                    candidates_by_seat[focus_seat.value] += 1
                    assert observation.candidate_action is not None
                    last_status = f"candidate: {observation.candidate_action.value}"
                if args.emit_all or observation.evidence_state is ActionEvidenceState.CANDIDATE:
                    print(
                        json.dumps(
                            {
                                "type": "multiseat_action_observation",
                                "layout_status": pilot.layout_status,
                                **observation_to_dict(observation),
                            },
                            ensure_ascii=False,
                        )
                    )

                if args.headless:
                    continue
                import cv2

                display = read.frame.image.copy()
                height, width = display.shape[:2]
                for seat in SEAT_ORDER:
                    roi = pilot.seat_rois[seat]
                    color = (0, 255, 0) if seat is focus_seat else SEAT_COLORS[seat]
                    thickness = 4 if seat is focus_seat else 2
                    top_left = (int(roi.x_min * width), int(roi.y_min * height))
                    bottom_right = (int(roi.x_max * width), int(roi.y_max * height))
                    cv2.rectangle(display, top_left, bottom_right, color, thickness)
                    seat_hands = routed.assignments[seat]
                    raw = "no hand"
                    if len(seat_hands) == 1:
                        item = seat_hands[0]
                        raw = f"{item.gesture_label or 'None'} {item.gesture_score or 0.0:.2f}"
                    elif len(seat_hands) > 1:
                        raw = f"{len(seat_hands)} hands - reject"
                    cv2.putText(
                        display,
                        f"{seat.value} [{SEAT_ORDER.index(seat) + 1}] {raw}",
                        (top_left[0] + 8, top_left[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        color,
                        2,
                    )
                for item in hands:
                    if item.centroid_x is None or item.centroid_y is None:
                        continue
                    point = (int(item.centroid_x * width), int(item.centroid_y * height))
                    cv2.circle(display, point, 8, (255, 255, 255), 2)
                cv2.putText(
                    display,
                    f"FOCUS {focus_seat.value} | {last_status} | keys 1-4 switch, Q/Esc quit",
                    (18, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Poker Dealer - Four Seat Action Pilot", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in SEAT_KEYS and SEAT_KEYS[key] is not focus_seat:
                    focus_seat = SEAT_KEYS[key]
                    state_version += 1
                    temporal = GestureTemporalAdapter(gesture)
                    last_status = "focus switched; temporal state reset"
            if not args.headless:
                import cv2

                cv2.destroyAllWindows()
    except CameraError as exc:
        print(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
        return 2

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    print(
        json.dumps(
            {
                "type": "summary",
                "status": "completed" if frames else "no_readable_frames",
                "pilot_status": pilot.pilot_status,
                "layout_status": pilot.layout_status,
                "camera": negotiated if frames else {},
                "elapsed_seconds": elapsed_s,
                "frames": frames,
                "missing_reads": missing,
                "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
                "final_focus_seat": focus_seat.value,
                "final_state_version": state_version,
                "hand_frames_by_seat": hand_frames_by_seat,
                "candidates": candidates,
                "candidates_by_seat": candidates_by_seat,
                "unassigned_hands": unassigned_hands,
                "ambiguous_hands": ambiguous_hands,
                "inference_latency_ms": {
                    "mean": sum(latencies) / len(latencies) if latencies else None,
                    "p95": _p95(latencies),
                    "maximum": max(latencies) if latencies else None,
                },
                "frames_saved": 0,
                "biometric_identity_used": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
