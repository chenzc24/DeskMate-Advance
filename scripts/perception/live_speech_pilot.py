"""Run a bounded, non-recording English poker-command microphone pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import time

from poker_dealer.domain import ActionEvidenceState, Seat
from poker_dealer.perception.actions import (
    ActionObservationContext,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    VoskSpeechRecognizer,
    observation_to_dict,
)

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - clear CLI error below
    sd = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/perception/actions_speech_pilot.json",
    )
    parser.add_argument("--device", type=_device)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--focus-seat", choices=tuple(seat.value for seat in Seat), default=Seat.A.value
    )
    parser.add_argument("--hand-id", default="laptop-speech-pilot")
    parser.add_argument("--state-version", type=int, default=0)
    parser.add_argument("--emit-partials", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sd is None:
        print(json.dumps({"type": "error", "error": "sounddevice is unavailable"}))
        return 2
    if args.list_devices:
        devices = []
        for index, device in enumerate(sd.query_devices()):
            if int(device["max_input_channels"]) > 0:
                devices.append(
                    {
                        "index": index,
                        "name": str(device["name"]),
                        "max_input_channels": int(device["max_input_channels"]),
                        "default_samplerate": float(device["default_samplerate"]),
                    }
                )
        print(json.dumps({"type": "input_devices", "devices": devices}, ensure_ascii=False))
        return 0

    config = SpeechPilotConfig.from_json(args.config)
    max_seconds = (
        float(config.max_seconds_default)
        if args.max_seconds is None
        else args.max_seconds
    )
    if max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    context = ActionObservationContext(
        hand_id=args.hand_id,
        expected_state_version=args.state_version,
        focus_seat=Seat(args.focus_seat),
    )
    audio_queue: queue.Queue[bytes] = queue.Queue(
        maxsize=int(config.audio["queue_max_blocks"])
    )
    dropped_blocks = 0

    def callback(indata: bytes, _frames: int, _time_info: object, status: object) -> None:
        nonlocal dropped_blocks
        if status:
            dropped_blocks += 1
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            dropped_blocks += 1

    sample_rate = int(config.audio["sample_rate_hz"])
    blocksize = int(config.audio["blocksize_frames"])
    started_ns = time.monotonic_ns()
    blocks = 0
    utterances = 0
    candidates = 0
    command_counts = {action.value: 0 for action in config.command_to_action.values()}
    last_partial = ""
    recognizer = VoskSpeechRecognizer(config)
    adapter = SpeechObservationAdapter(config)
    selected = sd.query_devices(args.device, "input")
    device_summary = {
        "requested": args.device,
        "name": str(selected["name"]),
        "sample_rate_hz": sample_rate,
        "channels": 1,
    }
    print(
        json.dumps(
            {
                "type": "ready",
                "commands": list(config.command_to_action),
                "controls": sorted(config.control_commands),
                "focus_seat": context.focus_seat.value,
                "device": device_summary,
                "audio_saved": False,
            },
            ensure_ascii=False,
        )
    )

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=blocksize,
            device=args.device,
            dtype=str(config.audio["dtype"]),
            channels=1,
            callback=callback,
        ):
            while (time.monotonic_ns() - started_ns) / 1_000_000_000 < max_seconds:
                try:
                    pcm = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                blocks += 1
                observed_at_ns = time.monotonic_ns()
                evidence = recognizer.accept_audio(pcm, observed_at_ns)
                if args.emit_partials:
                    partial = recognizer.partial_text()
                    if partial != last_partial:
                        print(
                            json.dumps(
                                {"type": "partial", "text": "".join(partial.split())},
                                ensure_ascii=False,
                            )
                        )
                        last_partial = partial
                if evidence is None:
                    continue
                utterances += 1
                observation = adapter.process(evidence, context)
                if observation.evidence_state is ActionEvidenceState.CANDIDATE:
                    candidates += 1
                    assert observation.candidate_action is not None
                    command_counts[observation.candidate_action.value] += 1
                print(
                    json.dumps(
                        {
                            "type": "speech_observation",
                            "transcript": evidence.canonical_transcript,
                            **observation_to_dict(observation),
                        },
                        ensure_ascii=False,
                    )
                )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(
            json.dumps(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 2

    final = recognizer.flush(time.monotonic_ns())
    if final is not None:
        utterances += 1
        observation = adapter.process(final, context)
        if observation.evidence_state is ActionEvidenceState.CANDIDATE:
            candidates += 1
            assert observation.candidate_action is not None
            command_counts[observation.candidate_action.value] += 1
        print(
            json.dumps(
                {
                    "type": "speech_observation",
                    "transcript": final.canonical_transcript,
                    **observation_to_dict(observation),
                },
                ensure_ascii=False,
            )
        )

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    print(
        json.dumps(
            {
                "type": "summary",
                "status": "completed",
                "pilot_status": config.pilot_status,
                "model_id": config.model.model_id,
                "model_version": config.model.version,
                "model_tree_sha256": config.model.tree_sha256,
                "device": device_summary,
                "elapsed_seconds": elapsed_s,
                "audio_blocks": blocks,
                "dropped_blocks": dropped_blocks,
                "utterances": utterances,
                "candidates": candidates,
                "candidate_counts": command_counts,
                "audio_saved_bytes": 0,
                "seat_attribution": config.seat_attribution,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
