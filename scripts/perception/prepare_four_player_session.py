"""Create one ignored pseudonymous record for tomorrow's four-player session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import build_acceptance_session_record


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-group", required=True)
    parser.add_argument("--operator-code", required=True)
    for seat in "abcd":
        parser.add_argument(f"--seat-{seat}-code", required=True)
    parser.add_argument(
        "--all-consent-confirmed",
        action="store_true",
        help="Operator attests that all four participants explicitly consented.",
    )
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--camera-distance-cm", type=float)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    consented = {f"seat_{seat}" for seat in "abcd"} if args.all_consent_confirmed else set()
    try:
        record = build_acceptance_session_record(
            session_group=args.session_group,
            operator_code=args.operator_code,
            participant_codes={
                f"seat_{seat}": getattr(args, f"seat_{seat}_code") for seat in "abcd"
            },
            consented_seats=consented,
            lighting=args.lighting,
            camera_distance_cm=args.camera_distance_cm,
            notes=args.notes,
        )
        output = args.output or (
            ROOT
            / "runs/stage2a_four_player_acceptance"
            / args.session_group
            / "session_record.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({"result": "CREATED", "path": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
