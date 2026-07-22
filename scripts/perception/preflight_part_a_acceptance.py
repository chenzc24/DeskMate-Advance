"""Run read-only Stage 2A asset, environment and device preflight checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import run_part_a_preflight


ROOT = Path(__file__).resolve().parents[2]


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--speech-device", type=_device, default=1)
    parser.add_argument("--skip-devices", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_part_a_preflight(
        ROOT,
        include_devices=not args.skip_devices,
        camera_index=args.camera_index,
        camera_backend=args.camera_backend,
        speech_device=args.speech_device,
        minimum_free_gib=args.minimum_free_gib,
    )
    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    print(rendered)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        except OSError as exc:
            print(json.dumps({"result": "ERROR", "error": str(exc)}))
            return 2
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
