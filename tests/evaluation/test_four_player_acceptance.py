from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_dealer.evaluation import (
    analyze_acceptance_case,
    load_acceptance_protocol,
    load_jsonl_events,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"
)


def _event(case_id: str, event_type: str, timestamp: int, **payload: object) -> dict[str, object]:
    return {
        "type": event_type,
        "session_id": "acceptance-session",
        "hand_id": "acceptance-hand",
        "acceptance_case": case_id,
        "acceptance_session_group": "group-01",
        "logged_at_monotonic_ns": timestamp,
        **payload,
    }


def _ready(case_id: str, timestamp: int = 1_000_000_000) -> dict[str, object]:
    return _event(
        case_id,
        "ready",
        timestamp,
        player_mode="four_player_core",
        frames_saved=0,
        audio_saved=False,
        embeddings_persisted=False,
    )


def _summary(case_id: str, timestamp: int) -> dict[str, object]:
    return _event(
        case_id,
        "summary",
        timestamp,
        frames_saved=0,
        audio_saved=False,
        embeddings_persisted=False,
        physical_robot_connected=False,
        dropped_audio_blocks=0,
    )


def test_protocol_has_nine_unique_planned_cases() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    assert protocol["status"] == "planned_not_executed"
    assert [case["case_id"] for case in protocol["cases"]] == [
        f"FPA-{index:02d}" for index in range(9)
    ]


def test_happy_path_report_passes_and_reports_latency() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    events = [_ready("FPA-01")]
    timestamp = 1_100_000_000
    for seat in ("seat_a", "seat_b", "seat_c", "seat_d"):
        events.append(
            _event(
                "FPA-01",
                "enrollment_completed",
                timestamp,
                seat=seat,
                player_id=f"player_{seat[-1]}",
            )
        )
        timestamp += 10_000_000
    events.append(
        _event(
            "FPA-01",
            "hand_started",
            timestamp,
            mode="four_player_core",
            button="seat_a",
            first_acting_seat="seat_d",
        )
    )
    timestamp += 10_000_000
    expected = protocol["happy_path_transitions"]
    for transition in expected:
        events.append(
            _event(
                "FPA-01",
                "identity_gate_opened",
                timestamp,
                focus_seat=transition["acting_seat"],
                state_version=transition["before_version"],
            )
        )
        timestamp += 250_000_000
        events.append(_event("FPA-01", "state_transition", timestamp, **transition))
        timestamp += 10_000_000
    events.append(_summary("FPA-01", timestamp))

    report = analyze_acceptance_case(protocol, "FPA-01", events)

    assert report["result"] == "PASS"
    assert report["failures"] == []
    assert report["metrics"]["confirmation_latency_ms"] == {
        "samples": 4,
        "p50": 250.0,
        "p95": 250.0,
    }


def test_happy_path_wrong_seat_order_fails() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    events = [_ready("FPA-01")]
    for index, seat in enumerate(("seat_a", "seat_b", "seat_c", "seat_d"), start=1):
        events.append(_event("FPA-01", "enrollment_completed", index + 1, seat=seat))
    events.append(
        _event(
            "FPA-01",
            "hand_started",
            10,
            mode="four_player_core",
            button="seat_a",
            first_acting_seat="seat_d",
        )
    )
    wrong = [dict(item) for item in protocol["happy_path_transitions"]]
    wrong[1]["acting_seat"] = "seat_c"
    events.extend(
        _event("FPA-01", "state_transition", 20 + index, **transition)
        for index, transition in enumerate(wrong)
    )
    events.append(_summary("FPA-01", 100))

    report = analyze_acceptance_case(protocol, "FPA-01", events)

    assert report["result"] == "FAIL"
    assert any("transition 2 mismatch" in failure for failure in report["failures"])


def test_multiple_face_safety_case_passes_without_transition() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    events = [
        _ready("FPA-03"),
        _event(
            "FPA-03",
            "identity_gate_opened",
            2_000_000_000,
            focus_seat="seat_d",
            state_version=0,
        ),
        _event(
            "FPA-03",
            "action_window_closed",
            2_500_000_000,
            reason="different_or_multiple_player_detected",
            focus_seat="seat_d",
            state_version=0,
        ),
        _summary("FPA-03", 3_000_000_000),
    ]

    report = analyze_acceptance_case(protocol, "FPA-03", events)

    assert report["result"] == "PASS"
    assert report["metrics"]["action_window_closure_reasons"] == {
        "different_or_multiple_player_detected": 1
    }


def test_face_loss_before_configured_grace_fails() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    events = [
        _ready("FPA-04"),
        _event(
            "FPA-04",
            "identity_gate_opened",
            2_000_000_000,
            focus_seat="seat_d",
            state_version=0,
        ),
        _event(
            "FPA-04",
            "action_window_closed",
            2_500_000_000,
            reason="face_missing_or_unknown_beyond_grace",
            focus_seat="seat_d",
            state_version=0,
        ),
        _summary("FPA-04", 3_000_000_000),
    ]

    report = analyze_acceptance_case(protocol, "FPA-04", events)

    assert report["result"] == "FAIL"
    assert any("expected at least 1000 ms" in failure for failure in report["failures"])


def test_context_or_persistence_omission_fails() -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)
    events = [_ready("FPA-07"), _summary("FPA-07", 2)]
    del events[1]["session_id"]
    del events[1]["audio_saved"]

    report = analyze_acceptance_case(protocol, "FPA-07", events)

    assert report["result"] == "FAIL"
    assert any("session_id" in failure for failure in report["failures"])
    assert any("audio" in failure for failure in report["failures"])


def test_jsonl_loader_reports_bad_line(tmp_path: Path) -> None:
    log_path = tmp_path / "bad.jsonl"
    log_path.write_text(json.dumps({"type": "ready"}) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl_events(log_path)


def _passing_events(case_id: str, protocol: dict[str, object]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [_ready(case_id)]
    timestamp = 2_000_000_000
    if case_id == "FPA-00":
        for seat in ("seat_a", "seat_b", "seat_c"):
            events.append(_event(case_id, "enrollment_completed", timestamp, seat=seat))
            timestamp += 10_000_000
        events.append(
            _event(case_id, "hand_start_blocked", timestamp, mode="four_player_core")
        )
    elif case_id == "FPA-01":
        for seat in ("seat_a", "seat_b", "seat_c", "seat_d"):
            events.append(_event(case_id, "enrollment_completed", timestamp, seat=seat))
            timestamp += 10_000_000
        events.append(
            _event(
                case_id,
                "hand_started",
                timestamp,
                mode="four_player_core",
                button="seat_a",
                first_acting_seat="seat_d",
            )
        )
        for transition in protocol["happy_path_transitions"]:
            timestamp += 10_000_000
            events.append(_event(case_id, "state_transition", timestamp, **transition))
    elif case_id == "FPA-02":
        events.append(
            _event(
                case_id,
                "identity_observation",
                timestamp,
                state="seat_mismatch",
                focus_seat="seat_d",
                state_version=0,
            )
        )
    elif case_id in {"FPA-03", "FPA-04"}:
        events.append(
            _event(
                case_id,
                "identity_gate_opened",
                timestamp,
                focus_seat="seat_d",
                state_version=0,
            )
        )
        timestamp += 1_100_000_000 if case_id == "FPA-04" else 100_000_000
        events.append(
            _event(
                case_id,
                "action_window_closed",
                timestamp,
                reason=(
                    "face_missing_or_unknown_beyond_grace"
                    if case_id == "FPA-04"
                    else "different_or_multiple_player_detected"
                ),
                focus_seat="seat_d",
                state_version=0,
            )
        )
    elif case_id == "FPA-05":
        events.append(
            _event(
                case_id,
                "speech_observation",
                timestamp,
                candidate_action="call",
                focus_seat="seat_d",
                state_version=0,
            )
        )
        timestamp += 600_000_000
        events.append(
            _event(
                case_id,
                "speech_ui_confirmation",
                timestamp,
                candidate_action="call",
                focus_seat="seat_d",
                state_version=0,
            )
        )
        timestamp += 1_000_000
        events.append(
            _event(
                case_id,
                "state_transition",
                timestamp,
                acting_seat="seat_d",
                action="call",
                before_version=0,
                after_version=1,
                next_seat="seat_a",
            )
        )
    elif case_id == "FPA-06":
        events.append(
            _event(
                case_id,
                "multimodal_action_decision",
                timestamp,
                accepted=False,
                evidence_state="ambiguous",
                focus_seat="seat_d",
                expected_state_version=0,
            )
        )
    elif case_id == "FPA-07":
        events.append(
            _event(
                case_id,
                "identity_gate_opened",
                timestamp,
                focus_seat="seat_d",
                state_version=0,
            )
        )
    elif case_id == "FPA-08":
        for seat in ("seat_d", "seat_a", "seat_b", "seat_c"):
            events.append(
                _event(
                    case_id,
                    "identity_observation",
                    timestamp,
                    state="seat_mismatch",
                    focus_seat=seat,
                )
            )
            timestamp += 10_000_000
        for transition in protocol["happy_path_transitions"]:
            events.append(_event(case_id, "state_transition", timestamp, **transition))
            timestamp += 10_000_000
    summary = _summary(case_id, timestamp + 2_000_000_000)
    if case_id == "FPA-07":
        summary["elapsed_seconds"] = 121.0
    events.append(summary)
    return events


@pytest.mark.parametrize("case_id", [f"FPA-{index:02d}" for index in range(9)])
def test_each_protocol_case_has_a_satisfiable_evidence_vector(case_id: str) -> None:
    protocol = load_acceptance_protocol(PROTOCOL_PATH)

    report = analyze_acceptance_case(protocol, case_id, _passing_events(case_id, protocol))

    assert report["result"] == "PASS", report["failures"]
