"""Run the consent-gated, session-only Laptop face identity pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from poker_dealer.domain import SEAT_ORDER, Seat
from poker_dealer.io.camera import CameraConfig, CameraError, CameraReadStatus, OpenCVCamera
from poker_dealer.perception.identity import (
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
    identity_observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]
SEAT_KEYS = {ord("1"): Seat.A, ord("2"): Seat.B, ord("3"): Seat.C, ord("4"): Seat.D}
PLAYER_BY_SEAT = {
    Seat.A: "player_a",
    Seat.B: "player_b",
    Seat.C: "player_c",
    Seat.D: "player_d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/face_identity_session.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--focus-seat", choices=tuple(seat.value for seat in Seat), default=Seat.A.value)
    parser.add_argument("--session-id", default="laptop-face-pilot")
    parser.add_argument("--state-version", type=int, default=0)
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def _camera_config(config: FaceIdentityConfig, args: argparse.Namespace) -> CameraConfig:
    values = config.camera
    return CameraConfig(
        device_index=int(values["device_index"]) if args.index is None else args.index,
        source_id="laptop_session_face_identity_pilot",
        backend=str(values["backend"]) if args.backend is None else args.backend,
        width=int(values["width"]),
        height=int(values["height"]),
        fps=float(values["fps"]),
    )


def main() -> int:
    args = parse_args()
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    config = FaceIdentityConfig.from_json(args.config)
    camera_config = _camera_config(config, args)
    focus_seat = Seat(args.focus_seat)
    state_version = args.state_version
    temporal = FaceIdentityTemporalAdapter(config)
    frames = 0
    missing = 0
    enrollment_samples = []
    enrollment_active = False
    last_enrollment_sample_ns: int | None = None
    status_text = "gallery empty - choose seat 1-4, then press E"
    last_emitted: tuple[str, str | None, str | None] | None = None
    latencies: list[float] = []
    negotiated: dict[str, int | float | str] = {}
    started_ns = time.monotonic_ns()

    try:
        with SessionFaceGallery(config, args.session_id) as gallery:
            with OpenCVCamera(camera_config) as camera:
                model = OpenCvFaceIdentityAdapter(config)
                negotiated = camera.negotiated_properties()
                print(
                    json.dumps(
                        {
                            "type": "ready",
                            "policy": config.policy_version,
                            "consent_confirmed": args.consent_confirmed,
                            "focus_seat": focus_seat.value,
                            "camera": negotiated,
                            "frames_saved": 0,
                            "embeddings_persisted": False,
                        },
                        ensure_ascii=False,
                    )
                )
                while (time.monotonic_ns() - started_ns) / 1_000_000_000 < args.max_seconds:
                    if args.max_frames is not None and frames >= args.max_frames:
                        break
                    read = camera.read()
                    if read.status is not CameraReadStatus.OK or read.frame is None:
                        missing += 1
                        if read.status is CameraReadStatus.DISCONNECTED:
                            status_text = "camera disconnected"
                            break
                        continue
                    frames += 1
                    evidence = model.analyze(read.frame)
                    latencies.append(evidence.inference_latency_ms)

                    if enrollment_active:
                        can_sample = (
                            evidence.detected_face_count == 1
                            and len(evidence.features) == 1
                            and (
                                last_enrollment_sample_ns is None
                                or evidence.observed_at_ns - last_enrollment_sample_ns >= 150_000_000
                            )
                        )
                        if can_sample:
                            enrollment_samples.append(evidence.features[0])
                            last_enrollment_sample_ns = evidence.observed_at_ns
                        status_text = (
                            f"ENROLL {PLAYER_BY_SEAT[focus_seat]}: "
                            f"{len(enrollment_samples)}/{config.minimum_samples}"
                        )
                        if len(enrollment_samples) >= config.minimum_samples:
                            try:
                                gallery.enroll(
                                    PLAYER_BY_SEAT[focus_seat],
                                    focus_seat,
                                    enrollment_samples,
                                    consent_granted=args.consent_confirmed,
                                )
                                status_text = f"enrolled {PLAYER_BY_SEAT[focus_seat]} for {focus_seat.value}"
                            except (PermissionError, ValueError) as exc:
                                status_text = f"enrollment rejected: {exc}"
                            enrollment_active = False
                            enrollment_samples = []
                            last_enrollment_sample_ns = None
                            temporal = FaceIdentityTemporalAdapter(config)

                    match = gallery.match_frame(evidence)
                    context = FaceIdentityContext(args.session_id, state_version, focus_seat)
                    observation = temporal.process(match, evidence.observed_at_ns, context)
                    emit_key = (
                        observation.identity_state.value,
                        observation.player_id,
                        observation.registered_seat.value if observation.registered_seat else None,
                    )
                    if args.emit_all or emit_key != last_emitted:
                        print(
                            json.dumps(
                                {"type": "face_identity_observation", **identity_observation_to_dict(observation)},
                                ensure_ascii=False,
                            )
                        )
                        last_emitted = emit_key
                    if not enrollment_active:
                        if observation.identity_state is FaceIdentityState.MATCHED:
                            status_text = f"VERIFIED {observation.player_id} at {focus_seat.value}"
                        elif observation.identity_state is FaceIdentityState.SEAT_MISMATCH:
                            status_text = (
                                f"MISMATCH {observation.player_id}: registered "
                                f"{observation.registered_seat.value if observation.registered_seat else '?'}"
                            )
                        elif observation.identity_state is FaceIdentityState.IDENTITY_START:
                            status_text = "identity match pending temporal confirmation"
                        else:
                            status_text = observation.identity_state.value

                    if args.headless:
                        continue
                    display = read.frame.image.copy()
                    for feature in evidence.features:
                        x, y, width, height = feature.bbox_xywh
                        color = (0, 220, 0) if observation.identity_state is FaceIdentityState.MATCHED else (0, 180, 255)
                        cv2.rectangle(display, (x, y), (x + width, y + height), color, 2)
                        cv2.putText(
                            display,
                            f"face {feature.detection_score:.2f}",
                            (x, max(24, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                        )
                    height, _width = display.shape[:2]
                    cv2.putText(
                        display,
                        f"FOCUS {focus_seat.value} -> {PLAYER_BY_SEAT[focus_seat]} | gallery {gallery.size}/4",
                        (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        status_text,
                        (18, height - 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        "1-4 focus | E enroll | X clear session gallery | Q/Esc quit",
                        (18, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Poker Dealer - Session Face Identity Pilot", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key in SEAT_KEYS and SEAT_KEYS[key] is not focus_seat:
                        focus_seat = SEAT_KEYS[key]
                        state_version += 1
                        temporal = FaceIdentityTemporalAdapter(config)
                        enrollment_active = False
                        enrollment_samples = []
                        last_enrollment_sample_ns = None
                        status_text = f"focus switched to {focus_seat.value}"
                    elif key == ord("e"):
                        if not args.consent_confirmed:
                            status_text = "enrollment blocked: restart with --consent-confirmed"
                        elif any(item["seat"] == focus_seat.value for item in gallery.metadata()):
                            status_text = f"{focus_seat.value} is already enrolled; X clears all"
                        else:
                            enrollment_active = True
                            enrollment_samples = []
                            last_enrollment_sample_ns = None
                            status_text = f"ENROLL {PLAYER_BY_SEAT[focus_seat]}: face camera alone"
                    elif key == ord("x"):
                        gallery.clear()
                        temporal = FaceIdentityTemporalAdapter(config)
                        enrollment_active = False
                        enrollment_samples = []
                        last_enrollment_sample_ns = None
                        status_text = "session gallery cleared from memory"
    except (CameraError, OSError, ValueError) as exc:
        print(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    print(
        json.dumps(
            {
                "type": "summary",
                "status": "completed" if frames else "no_readable_frames",
                "frames": frames,
                "missing_reads": missing,
                "elapsed_seconds": elapsed_s,
                "effective_fps": frames / elapsed_s if elapsed_s else 0.0,
                "camera": negotiated,
                "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "gallery_cleared_on_exit": True,
                "frames_saved": 0,
                "embeddings_persisted": False,
                "robot_motion": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
