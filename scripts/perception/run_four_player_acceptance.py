"""Launch and analyze one planned Stage 2A four-player acceptance case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import subprocess
import sys

from poker_dealer.evaluation import (
    load_acceptance_protocol,
    load_acceptance_session_record,
)
from poker_dealer.io.camera import CameraConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"
)


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--session-group", required=True)
    parser.add_argument("--session-record", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--index", type=int)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"))
    parser.add_argument(
        "--stream-url",
        help="HTTP(S) MJPEG stream; mutually exclusive with --index",
    )
    parser.add_argument("--speech-device", type=_device, default=1)
    parser.add_argument("--disable-speech", action="store_true")
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _case_ids(protocol: dict[str, object]) -> set[str]:
    return {str(case["case_id"]) for case in protocol["cases"]}  # type: ignore[index]


def build_commands(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], Path, Path, Path, Path]:
    protocol = load_acceptance_protocol(args.protocol)
    if args.case_id not in _case_ids(protocol):
        raise ValueError(f"unknown acceptance case: {args.case_id}")
    if not args.consent_confirmed and not args.dry_run:
        raise ValueError("live enrollment requires --consent-confirmed")
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if args.stream_url is not None:
        if args.index is not None:
            raise ValueError("--stream-url and --index are mutually exclusive")
        if args.backend not in {None, "auto"}:
            raise ValueError("network streams use FFmpeg; omit --backend")
        CameraConfig(
            stream_url=args.stream_url,
            backend="auto",
            width=None,
            height=None,
            fps=None,
        )
    record_path = args.session_record or (
        ROOT
        / "runs/stage2a_four_player_acceptance"
        / args.session_group
        / "session_record.json"
    )
    if not args.dry_run:
        record = load_acceptance_session_record(record_path, require_all_consent=True)
        if record["session_group"] != args.session_group:
            raise ValueError("session record group does not match --session-group")
    elif not args.session_group.replace("-", "").replace("_", "").isalnum():
        raise ValueError("dry-run session group contains unsafe characters")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = (
        f"{args.session_group}-{args.case_id.lower()}-{stamp}-{secrets.token_hex(2)}"
    )
    evidence_dir = (
        ROOT
        / "runs/stage2a_four_player_acceptance"
        / args.session_group
        / session_id
    )
    log_path = evidence_dir / f"{args.case_id}.jsonl"
    report_path = evidence_dir / f"{args.case_id}.report.json"
    operator_observation_path = evidence_dir / "operator_observation.json"
    hand_id = f"{session_id}-hand"
    runtime = protocol["runtime"]

    camera_arguments = (
        ["--stream-url", args.stream_url]
        if args.stream_url is not None
        else [
            "--index",
            str(0 if args.index is None else args.index),
            "--backend",
            "dshow" if args.backend is None else args.backend,
        ]
    )
    live_command = [
        sys.executable,
        str(ROOT / "scripts/perception/live_sequential_part_a.py"),
        *camera_arguments,
        "--player-mode",
        str(runtime["player_mode"]),  # type: ignore[index]
        "--button",
        str(runtime["button"]),  # type: ignore[index]
        "--identity-grace-ms",
        str(runtime["identity_grace_ms"]),  # type: ignore[index]
        "--session-id",
        session_id,
        "--hand-id",
        hand_id,
        "--acceptance-case",
        args.case_id,
        "--acceptance-session-group",
        args.session_group,
        "--log-jsonl",
        str(log_path),
        "--max-seconds",
        str(args.max_seconds),
    ]
    if args.consent_confirmed:
        live_command.append("--consent-confirmed")
    if args.disable_speech:
        live_command.append("--disable-speech")
    else:
        live_command.extend(("--speech-device", str(args.speech_device)))

    analyze_command = [
        sys.executable,
        str(ROOT / "scripts/perception/analyze_four_player_acceptance.py"),
        str(log_path),
        "--case",
        args.case_id,
        "--protocol",
        str(args.protocol),
        "--output",
        str(report_path),
    ]
    return (
        live_command,
        analyze_command,
        log_path,
        report_path,
        record_path,
        operator_observation_path,
    )


def main() -> int:
    args = parse_args()
    try:
        protocol = load_acceptance_protocol(args.protocol)
        selected_case = next(
            case for case in protocol["cases"] if case["case_id"] == args.case_id
        )
        (
            live_command,
            analyze_command,
            log_path,
            report_path,
            record_path,
            operator_observation_path,
        ) = build_commands(args)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "title": selected_case["title"],
                "operator_steps": selected_case["operator_steps"],
                "session_group": args.session_group,
                "session_record": str(record_path),
                "live_command": live_command,
                "log_jsonl": str(log_path),
                "report_json": str(report_path),
                "operator_observation_json": str(operator_observation_path),
                "operator_observation_command": [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts/perception/record_acceptance_case_observation.py"
                    ),
                    str(log_path),
                    "--operator-code",
                    "<operator-code>",
                    "--observed-result",
                    "observed_pass|observed_fail",
                    "--handedness-used",
                    "left|right|both|not_applicable",
                    "--camera-distance-cm",
                    "<cm>",
                    "--lighting",
                    "<condition>",
                    "--speech-used",
                    "yes|no",
                    "--gesture-used",
                    "yes|no",
                    "--failure-category",
                    "none|identity|gesture|speech|fusion|state|ui|environment|crash",
                ],
                "frames_saved": 0,
                "audio_saved": False,
                "embeddings_persisted": False,
                "physical_robot_connected": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    live_result = subprocess.run(live_command, cwd=ROOT, check=False)
    if not log_path.exists():
        print("ERROR: live runtime did not create the JSONL evidence log", file=sys.stderr)
        return live_result.returncode or 2
    analyze_result = subprocess.run(analyze_command, cwd=ROOT, check=False)
    if live_result.returncode != 0:
        return live_result.returncode
    return analyze_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
