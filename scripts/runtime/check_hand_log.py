"""Verify a runtime JSONL chain and independently recompute hand invariants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from poker_dealer.runtime import RuntimeEventLog, check_runtime_hand_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="check integrity without requiring the hand to be settled",
    )
    args = parser.parse_args(argv)
    result = check_runtime_hand_log(
        RuntimeEventLog.from_path(args.log),
        require_settled=not args.allow_incomplete,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
