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
    args = parser.parse_args(argv)
    result = check_runtime_hand_log(RuntimeEventLog.from_path(args.log))
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
