"""Aggregate every preserved FPA attempt for one pseudonymous session group."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from poker_dealer.evaluation import (
    aggregate_acceptance_session,
    load_acceptance_protocol,
    load_acceptance_session_record,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-group", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--session-record", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_root = args.evidence_root or (
        ROOT / "runs/stage2a_four_player_acceptance" / args.session_group
    )
    session_record_path = args.session_record or evidence_root / "session_record.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or evidence_root / f"batch-report-{stamp}.json"
    try:
        protocol = load_acceptance_protocol(args.protocol)
        record = load_acceptance_session_record(
            session_record_path, require_all_consent=True
        )
        if record["session_group"] != args.session_group:
            raise ValueError("session record does not match --session-group")
        report = aggregate_acceptance_session(
            protocol,
            record,
            tuple(evidence_root.glob("**/FPA-*.jsonl")),
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"result": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(rendered)
    return 0 if report["result"] in {"COMPLETE_PASS", "COMPLETE_WITH_RETRIES"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
