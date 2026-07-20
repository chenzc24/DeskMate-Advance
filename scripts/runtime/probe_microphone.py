"""List microphones, validate the built-in input, and optionally smoke-read it."""

from __future__ import annotations

import argparse
import json

from deskmate_advance.perception.audio import MicrophoneConfig, MicrophoneError
from deskmate_advance.perception.audio.diagnostics import (
    check_input_config,
    list_input_devices,
    smoke_read,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--block-ms", type=int, default=100)
    parser.add_argument(
        "--read-once",
        action="store_true",
        help="capture one bounded block and print aggregate levels; saves nothing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MicrophoneConfig(
        device_index=args.device,
        sample_rate_hz=args.sample_rate,
        channel_count=args.channels,
        block_duration_ms=args.block_ms,
    )
    try:
        report: dict[str, object] = {
            "input_devices": list_input_devices(),
            "selected_config": check_input_config(config),
        }
        if args.read_once:
            report["smoke_read"] = smoke_read(config)
    except MicrophoneError as error:
        report = {"error": str(error)}
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["selected_config"]["supported"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
