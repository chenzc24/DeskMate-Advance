"""Aggregate all preserved attempts for one four-player acceptance session."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from .acceptance_session import (
    validate_acceptance_session_record,
    validate_case_observation_record,
)
from .four_player_acceptance import analyze_acceptance_case, load_jsonl_events


def aggregate_acceptance_session(
    protocol: Mapping[str, Any],
    session_record: Mapping[str, Any],
    log_paths: Sequence[Path],
) -> dict[str, Any]:
    validate_acceptance_session_record(session_record, require_all_consent=True)
    expected_cases = [str(case["case_id"]) for case in protocol["cases"]]
    session_group = str(session_record["session_group"])
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_session_ids: set[str] = set()
    ingestion_failures: list[str] = []

    for path in sorted(log_paths):
        try:
            events = load_jsonl_events(path)
            case_tags = {event.get("acceptance_case") for event in events}
            group_tags = {event.get("acceptance_session_group") for event in events}
            session_tags = {event.get("session_id") for event in events}
            if len(case_tags) != 1 or next(iter(case_tags)) not in expected_cases:
                raise ValueError(f"invalid case tags {case_tags}")
            if group_tags != {session_group}:
                raise ValueError(f"session group tags {group_tags} do not match {session_group}")
            if None in session_tags or len(session_tags) != 1:
                raise ValueError("attempt does not contain exactly one session_id")
            session_id = str(next(iter(session_tags)))
            if session_id in seen_session_ids:
                raise ValueError(f"duplicate session_id {session_id}")
            seen_session_ids.add(session_id)
            case_id = str(next(iter(case_tags)))
            report = analyze_acceptance_case(protocol, case_id, events)
            manual_path = path.parent / "operator_observation.json"
            manual_failures: list[str] = []
            manual_record: Mapping[str, Any] | None = None
            try:
                manual_record = json.loads(manual_path.read_text(encoding="utf-8"))
                validate_case_observation_record(
                    manual_record,
                    session_group=session_group,
                    session_id=session_id,
                    case_id=case_id,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                manual_failures.append(
                    f"operator observation missing or invalid: {type(exc).__name__}: {exc}"
                )
            if (
                manual_record is not None
                and manual_record.get("observed_result") != "observed_pass"
            ):
                manual_failures.append("operator marked the attempt observed_fail")
            combined_failures = list(report["failures"]) + manual_failures
            attempts[case_id].append(
                {
                    "session_id": session_id,
                    "log_path": str(path),
                    "result": "PASS" if not combined_failures else "FAIL",
                    "machine_result": report["result"],
                    "operator_result": (
                        manual_record.get("observed_result")
                        if manual_record is not None
                        else "missing"
                    ),
                    "operator_observation_path": str(manual_path),
                    "failures": combined_failures,
                    "warnings": report["warnings"],
                    "metrics": report["metrics"],
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            ingestion_failures.append(f"{path}: {type(exc).__name__}: {exc}")

    case_results: list[dict[str, Any]] = []
    total_event_counts: Counter[str] = Counter()
    total_identity_states: Counter[str] = Counter()
    total_closure_reasons: Counter[str] = Counter()
    any_failed_attempt = False
    for case_id in expected_cases:
        case_attempts = attempts.get(case_id, [])
        passed = [item for item in case_attempts if item["result"] == "PASS"]
        failed = [item for item in case_attempts if item["result"] != "PASS"]
        any_failed_attempt = any_failed_attempt or bool(failed)
        for item in case_attempts:
            metrics = item["metrics"]
            total_event_counts.update(metrics["event_counts"])
            total_identity_states.update(metrics["identity_states"])
            total_closure_reasons.update(metrics["action_window_closure_reasons"])
        case_results.append(
            {
                "case_id": case_id,
                "status": (
                    "PASS"
                    if passed
                    else ("FAIL" if case_attempts else "MISSING")
                ),
                "attempt_count": len(case_attempts),
                "pass_count": len(passed),
                "fail_count": len(failed),
                "attempts": case_attempts,
            }
        )

    missing_cases = [item["case_id"] for item in case_results if item["status"] == "MISSING"]
    failed_cases = [item["case_id"] for item in case_results if item["status"] == "FAIL"]
    if ingestion_failures or failed_cases:
        result = "FAIL"
    elif missing_cases:
        result = "INCOMPLETE"
    elif any_failed_attempt:
        result = "COMPLETE_WITH_RETRIES"
    else:
        result = "COMPLETE_PASS"

    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "session_group": session_group,
        "result": result,
        "case_results": case_results,
        "missing_cases": missing_cases,
        "failed_cases": failed_cases,
        "ingestion_failures": ingestion_failures,
        "aggregate_metrics": {
            "attempts": sum(len(items) for items in attempts.values()),
            "event_counts": dict(sorted(total_event_counts.items())),
            "identity_states": dict(sorted(total_identity_states.items())),
            "action_window_closure_reasons": dict(
                sorted(total_closure_reasons.items())
            ),
        },
        "privacy": session_record["privacy"],
        "physical_robot_connected": False,
    }
