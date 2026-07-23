"""Independently verify one Runtime diagnostics bundle."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from poker_dealer.runtime import check_diagnostic_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    result = check_diagnostic_bundle(args.bundle)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
