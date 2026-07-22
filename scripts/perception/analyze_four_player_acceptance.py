"""Analyze one Stage 2A four-player live-acceptance JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import (
    analyze_acceptance_case,
    load_acceptance_protocol,
    load_jsonl_events,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_jsonl", type=Path)
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            ROOT
            / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        protocol = load_acceptance_protocol(args.protocol)
        events = load_jsonl_events(args.log_jsonl)
        report = analyze_acceptance_case(protocol, args.case_id, events)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {"result": "ERROR", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        except OSError as exc:
            print(
                json.dumps(
                    {"result": "ERROR", "error": f"cannot create report: {exc}"},
                    ensure_ascii=True,
                )
            )
            return 2
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
