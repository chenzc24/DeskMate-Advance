"""Run deterministic no-device Part A rejection and recovery replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import run_action_safety_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-action-events", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_action_safety_replay(args.no_action_events)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"result": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(rendered)
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
