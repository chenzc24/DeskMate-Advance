"""Run the seeded Stage 1 randomized legal-hand Gate."""

from __future__ import annotations

import argparse
import json

from poker_dealer.game import run_random_hands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    summary = run_random_hands(args.hands, args.seed)
    print(
        json.dumps(
            {
                "hands": summary.hands,
                "actions": summary.actions,
                "showdowns": summary.showdowns,
                "folds": summary.folds,
                "seed": summary.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
