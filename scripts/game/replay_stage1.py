"""Run the executable Stage 0 walkthrough matrix against the Stage 1 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.game import run_walkthroughs


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--walkthroughs",
        type=Path,
        default=ROOT / "configs" / "game" / "stage0_walkthroughs.json",
    )
    parser.add_argument("--scenario", help="Optional full scenario ID")
    args = parser.parse_args()

    results = run_walkthroughs(args.walkthroughs)
    if args.scenario:
        results = tuple(item for item in results if item.scenario_id == args.scenario)
        if not results:
            parser.error(f"unknown scenario: {args.scenario}")
    payload = [
        {
            "scenario_id": result.scenario_id,
            "passed": result.passed,
            "mismatches": list(result.mismatches),
        }
        for result in results
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
