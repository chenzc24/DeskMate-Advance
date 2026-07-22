"""Write the operator's pseudonymous observation beside one FPA JSONL attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import build_case_observation_record, load_jsonl_events


def _yes_no(value: str) -> bool:
    if value == "yes":
        return True
    if value == "no":
        return False
    raise argparse.ArgumentTypeError("use yes or no")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_jsonl", type=Path)
    parser.add_argument("--operator-code", required=True)
    parser.add_argument("--observed-result", choices=("observed_pass", "observed_fail"), required=True)
    parser.add_argument(
        "--handedness-used",
        choices=("left", "right", "both", "not_applicable"),
        required=True,
    )
    parser.add_argument("--camera-distance-cm", type=float, required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--speech-used", type=_yes_no, required=True)
    parser.add_argument("--gesture-used", type=_yes_no, required=True)
    parser.add_argument(
        "--failure-category",
        choices=(
            "none",
            "identity",
            "gesture",
            "speech",
            "fusion",
            "state",
            "ui",
            "environment",
            "crash",
        ),
        required=True,
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        events = load_jsonl_events(args.log_jsonl)
        session_ids = {str(event.get("session_id")) for event in events}
        groups = {str(event.get("acceptance_session_group")) for event in events}
        cases = {str(event.get("acceptance_case")) for event in events}
        if len(session_ids) != 1 or len(groups) != 1 or len(cases) != 1:
            raise ValueError("JSONL does not contain one session/group/case")
        record = build_case_observation_record(
            session_group=next(iter(groups)),
            session_id=next(iter(session_ids)),
            case_id=next(iter(cases)),
            operator_code=args.operator_code,
            observed_result=args.observed_result,
            handedness_used=args.handedness_used,
            camera_distance_cm=args.camera_distance_cm,
            lighting=args.lighting,
            speech_used=args.speech_used,
            gesture_used=args.gesture_used,
            failure_category=args.failure_category,
            notes=args.notes,
        )
        output = args.output or args.log_jsonl.parent / "operator_observation.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"result": "ERROR", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"result": "CREATED", "path": str(output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
