"""Verify a session chain, all referenced hand logs and continuity invariants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from poker_dealer.runtime import SessionEventLog, check_session_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--skip-hand-files",
        action="store_true",
        help="check only the session chain and declared continuity",
    )
    args = parser.parse_args(argv)
    result = check_session_log(
        SessionEventLog.from_path(args.log),
        verify_hand_logs=not args.skip_hand_files,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
