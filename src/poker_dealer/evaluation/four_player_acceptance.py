"""Evaluate one four-player live-acceptance JSONL session."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonObject = dict[str, Any]


def load_acceptance_protocol(path: Path) -> JsonObject:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "1.0":
        raise ValueError("unsupported acceptance protocol schema")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("acceptance protocol must contain cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)) or any(not item for item in case_ids):
        raise ValueError("acceptance case IDs must be present and unique")
    for case in cases:
        assertions = case.get("assertions")
        if not isinstance(assertions, dict):
            raise ValueError(f"{case['case_id']} must contain assertions")
        ordered = assertions.get("ordered_events", [])
        gaps = assertions.get("minimum_ordered_gaps_ms", [])
        if gaps and len(gaps) != len(ordered) - 1:
            raise ValueError(
                f"{case['case_id']} minimum_ordered_gaps_ms does not match ordered events"
            )
        reference = assertions.get("exact_transitions_ref")
        if reference is not None and not isinstance(protocol.get(reference), list):
            raise ValueError(f"{case['case_id']} has unknown transition reference {reference}")
    return protocol


def load_jsonl_events(path: Path) -> list[JsonObject]:
    events: list[JsonObject] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError(f"invalid event object at line {line_number}")
        events.append(event)
    if not events:
        raise ValueError("acceptance log contains no events")
    return events


def _case(protocol: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for item in protocol["cases"]:
        if item.get("case_id") == case_id:
            return item
    raise ValueError(f"unknown acceptance case: {case_id}")


def _matches(event: Mapping[str, Any], specification: Mapping[str, Any]) -> bool:
    if event.get("type") != specification.get("type"):
        return False
    fields = specification.get("fields", {})
    return all(event.get(key) == value for key, value in fields.items())


def _describe(specification: Mapping[str, Any]) -> str:
    fields = specification.get("fields", {})
    suffix = f" {fields}" if fields else ""
    return f"{specification.get('type')}{suffix}"


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _metrics(events: list[JsonObject]) -> JsonObject:
    event_counts = Counter(str(event["type"]) for event in events)
    identity_states = Counter(
        str(event.get("state"))
        for event in events
        if event["type"] == "identity_observation"
    )
    closure_reasons = Counter(
        str(event.get("reason"))
        for event in events
        if event["type"] == "action_window_closed"
    )
    transitions = [
        {
            key: event.get(key)
            for key in (
                "acting_seat",
                "action",
                "before_version",
                "after_version",
                "next_seat",
                "game_phase",
                "runtime_phase",
            )
        }
        for event in events
        if event["type"] == "state_transition"
    ]

    gate_opened_at: dict[tuple[Any, Any], int] = {}
    confirmation_latencies_ms: list[float] = []
    for event in events:
        timestamp = event.get("logged_at_monotonic_ns")
        if not isinstance(timestamp, int):
            continue
        if event["type"] == "identity_gate_opened":
            gate_opened_at[(event.get("focus_seat"), event.get("state_version"))] = timestamp
        elif event["type"] == "state_transition":
            key = (event.get("acting_seat"), event.get("before_version"))
            started = gate_opened_at.get(key)
            if started is not None and timestamp >= started:
                confirmation_latencies_ms.append((timestamp - started) / 1_000_000)

    accepted_decisions = sum(
        1
        for event in events
        if event["type"] == "multimodal_action_decision"
        and event.get("accepted") is True
    )
    rejected_decisions = sum(
        1
        for event in events
        if event["type"] == "multimodal_action_decision"
        and event.get("accepted") is False
    )
    return {
        "event_counts": dict(sorted(event_counts.items())),
        "enrolled_seats": sorted(
            {
                str(event.get("seat"))
                for event in events
                if event["type"] == "enrollment_completed" and event.get("seat")
            }
        ),
        "identity_states": dict(sorted(identity_states.items())),
        "action_window_closure_reasons": dict(sorted(closure_reasons.items())),
        "accepted_decisions": accepted_decisions,
        "rejected_decisions": rejected_decisions,
        "transitions": transitions,
        "confirmation_latency_ms": {
            "samples": len(confirmation_latencies_ms),
            "p50": _percentile(confirmation_latencies_ms, 0.50),
            "p95": _percentile(confirmation_latencies_ms, 0.95),
        },
    }


def analyze_acceptance_case(
    protocol: Mapping[str, Any], case_id: str, events: list[JsonObject]
) -> JsonObject:
    case = _case(protocol, case_id)
    assertions = case.get("assertions", {})
    failures: list[str] = []
    warnings: list[str] = []

    ready = [event for event in events if event["type"] == "ready"]
    summaries = [event for event in events if event["type"] == "summary"]
    if len(ready) != 1:
        failures.append(f"expected one ready event, found {len(ready)}")
    if len(summaries) != 1:
        failures.append(f"expected one summary event, found {len(summaries)}")
    if any(event["type"] == "error" for event in events):
        failures.append("runtime emitted an error event")

    expected_mode = protocol["runtime"]["player_mode"]
    if ready and ready[0].get("player_mode") != expected_mode:
        failures.append(
            f"ready player_mode is {ready[0].get('player_mode')!r}, expected {expected_mode!r}"
        )

    session_ids = {event.get("session_id") for event in events}
    hand_ids = {event.get("hand_id") for event in events}
    case_ids = {event.get("acceptance_case") for event in events}
    session_groups = {event.get("acceptance_session_group") for event in events}
    if None in session_ids or len(session_ids) != 1:
        failures.append("events do not share one non-null session_id")
    if None in hand_ids or len(hand_ids) != 1:
        failures.append("events do not share one non-null hand_id")
    if case_ids != {case_id}:
        failures.append(f"acceptance_case tags are {sorted(map(str, case_ids))}, expected {case_id}")
    if (
        None in session_groups
        or "UNASSIGNED" in session_groups
        or len(session_groups) != 1
    ):
        failures.append("events do not share one assigned acceptance_session_group")
    if any(not isinstance(event.get("logged_at_monotonic_ns"), int) for event in events):
        failures.append("one or more events lack logged_at_monotonic_ns")

    for event in ready + summaries:
        if event.get("frames_saved") != 0:
            failures.append(f"{event['type']} reports persisted frames")
        if event.get("audio_saved") is not False:
            failures.append(f"{event['type']} reports persisted audio or omits the flag")
        if event.get("embeddings_persisted") is not False:
            failures.append(f"{event['type']} reports persisted embeddings or omits the flag")
    if summaries and summaries[0].get("physical_robot_connected") is not False:
        failures.append("summary does not prove physical_robot_connected=false")

    for specification in assertions.get("required_events", []):
        count = sum(_matches(event, specification) for event in events)
        minimum = int(specification.get("min_count", 1))
        if count < minimum:
            failures.append(
                f"required {_describe(specification)} at least {minimum} time(s), found {count}"
            )

    for specification in assertions.get("forbidden_events", []):
        count = sum(_matches(event, specification) for event in events)
        if count:
            failures.append(f"forbidden {_describe(specification)} occurred {count} time(s)")

    cursor = 0
    ordered_indices: list[int] = []
    for specification in assertions.get("ordered_events", []):
        for index in range(cursor, len(events)):
            if _matches(events[index], specification):
                ordered_indices.append(index)
                cursor = index + 1
                break
        else:
            failures.append(f"ordered event missing after index {cursor}: {_describe(specification)}")
            break

    minimum_gaps = assertions.get("minimum_ordered_gaps_ms", [])
    if minimum_gaps and len(ordered_indices) == len(assertions.get("ordered_events", [])):
        if len(minimum_gaps) != len(ordered_indices) - 1:
            failures.append("minimum_ordered_gaps_ms length does not match ordered events")
        else:
            for index, minimum_ms in enumerate(minimum_gaps):
                before = events[ordered_indices[index]].get("logged_at_monotonic_ns")
                after = events[ordered_indices[index + 1]].get("logged_at_monotonic_ns")
                if not isinstance(before, int) or not isinstance(after, int):
                    failures.append("ordered event gap cannot be computed without timestamps")
                    break
                actual_ms = (after - before) / 1_000_000
                if actual_ms < float(minimum_ms):
                    failures.append(
                        f"ordered event gap {index + 1} is {actual_ms:.3f} ms, "
                        f"expected at least {minimum_ms} ms"
                    )

    if "minimum_summary_elapsed_seconds" in assertions and summaries:
        minimum_elapsed = float(assertions["minimum_summary_elapsed_seconds"])
        actual_elapsed = summaries[0].get("elapsed_seconds")
        if not isinstance(actual_elapsed, (int, float)) or actual_elapsed < minimum_elapsed:
            failures.append(
                f"summary elapsed_seconds is {actual_elapsed!r}, expected at least {minimum_elapsed}"
            )

    transitions = [event for event in events if event["type"] == "state_transition"]
    if "transition_count" in assertions:
        expected_count = int(assertions["transition_count"])
        if len(transitions) != expected_count:
            failures.append(
                f"expected {expected_count} state transition(s), found {len(transitions)}"
            )

    reference_name = assertions.get("exact_transitions_ref")
    if reference_name:
        expected_transitions = protocol.get(reference_name)
        if not isinstance(expected_transitions, list):
            failures.append(f"unknown transition reference: {reference_name}")
        elif len(transitions) != len(expected_transitions):
            failures.append(
                f"expected {len(expected_transitions)} exact transitions, found {len(transitions)}"
            )
        else:
            for index, (event, expected) in enumerate(
                zip(transitions, expected_transitions, strict=True), start=1
            ):
                mismatches = {
                    key: {"expected": value, "actual": event.get(key)}
                    for key, value in expected.items()
                    if event.get(key) != value
                }
                if mismatches:
                    failures.append(f"transition {index} mismatch: {mismatches}")

    for previous, current in zip(transitions, transitions[1:]):
        if current.get("before_version") != previous.get("after_version"):
            failures.append("state transition versions are not contiguous")
            break

    if summaries and summaries[0].get("dropped_audio_blocks", 0):
        warnings.append(
            f"audio queue dropped {summaries[0]['dropped_audio_blocks']} block(s)"
        )

    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "case_id": case_id,
        "case_title": case["title"],
        "result": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "metrics": _metrics(events),
    }
