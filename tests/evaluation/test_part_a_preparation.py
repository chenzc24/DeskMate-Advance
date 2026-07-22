from __future__ import annotations

import json
from pathlib import Path

import pytest
import jsonschema

from poker_dealer.evaluation import (
    aggregate_acceptance_session,
    assign_participant_splits,
    build_case_observation_record,
    build_acceptance_session_record,
    load_acceptance_protocol,
    run_action_safety_replay,
    run_part_a_preflight,
    validate_acceptance_session_record,
    validate_action_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"


def _session_record(all_consent: bool = True) -> dict[str, object]:
    return build_acceptance_session_record(
        session_group="group-01",
        operator_code="operator-01",
        participant_codes={f"seat_{seat}": f"P0{index}" for index, seat in enumerate("abcd", 1)},
        consented_seats=(
            {f"seat_{seat}" for seat in "abcd"} if all_consent else {"seat_a"}
        ),
        lighting="office-even",
        camera_distance_cm=85.0,
    )


def _event(event_type: str, timestamp: int, **payload: object) -> dict[str, object]:
    return {
        "type": event_type,
        "session_id": "group-01-fpa-00-attempt-1",
        "hand_id": "hand-1",
        "acceptance_case": "FPA-00",
        "acceptance_session_group": "group-01",
        "logged_at_monotonic_ns": timestamp,
        **payload,
    }


def test_preflight_without_devices_checks_assets_and_environment() -> None:
    report = run_part_a_preflight(ROOT, include_devices=False)

    assert report["result"] == "PASS"
    statuses = {item["check_id"]: item["status"] for item in report["checks"]}
    assert statuses["gesture_asset_sha256"] == "PASS"
    assert statuses["speech_asset_tree_sha256"] == "PASS"
    assert statuses["face_assets_sha256"] == "PASS"
    assert statuses["hand_landmarker_asset_sha256"] == "PASS"
    assert statuses["camera_read"] == "SKIP"
    assert report["physical_robot_connected"] is False


def test_session_record_is_pseudonymous_and_requires_four_consents() -> None:
    record = _session_record()
    validate_acceptance_session_record(record, require_all_consent=True)
    assert record["privacy"]["contains_real_names"] is False

    partial = _session_record(all_consent=False)
    with pytest.raises(ValueError, match="consent"):
        validate_acceptance_session_record(partial, require_all_consent=True)


def test_batch_report_preserves_incomplete_session(tmp_path: Path) -> None:
    events = [
        _event(
            "ready",
            1,
            player_mode="four_player_core",
            frames_saved=0,
            audio_saved=False,
            embeddings_persisted=False,
        )
    ]
    for index, seat in enumerate(("seat_a", "seat_b", "seat_c"), start=2):
        events.append(_event("enrollment_completed", index, seat=seat))
    events.extend(
        (
            _event("hand_start_blocked", 5, mode="four_player_core"),
            _event(
                "summary",
                6,
                frames_saved=0,
                audio_saved=False,
                embeddings_persisted=False,
                physical_robot_connected=False,
                dropped_audio_blocks=0,
            ),
        )
    )
    log_path = tmp_path / "FPA-00.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    operator_record = build_case_observation_record(
        session_group="group-01",
        session_id="group-01-fpa-00-attempt-1",
        case_id="FPA-00",
        operator_code="operator-01",
        observed_result="observed_pass",
        handedness_used="not_applicable",
        camera_distance_cm=85.0,
        lighting="office-even",
        speech_used=False,
        gesture_used=False,
        failure_category="none",
        notes="incomplete roster was blocked",
    )
    (tmp_path / "operator_observation.json").write_text(
        json.dumps(operator_record), encoding="utf-8"
    )

    report = aggregate_acceptance_session(
        load_acceptance_protocol(PROTOCOL), _session_record(), [log_path]
    )

    assert report["result"] == "INCOMPLETE"
    assert report["case_results"][0]["status"] == "PASS"
    assert report["missing_cases"] == [f"FPA-{index:02d}" for index in range(1, 9)]


def _source_manifest() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for participant_index in range(1, 7):
        participant = f"P{participant_index:02d}"
        for session_index in range(1, 3):
            source_id = f"{participant}-S{session_index}"
            records.append(
                {
                    "source_id": source_id,
                    "participant_code": participant,
                    "session_id": f"{participant}-session-{session_index}",
                    "seat": f"seat_{'abcd'[(participant_index - 1) % 4]}",
                    "capture_path": f"data/raw/action/{source_id}.mp4",
                    "sha256": f"{len(records) + 1:064x}",
                    "bytes": 100 + len(records),
                    "camera_id": "laptop-camera-0",
                    "lighting": "office",
                    "label": "call" if session_index == 1 else "no_action",
                    "duration_ms": 3000,
                    "contains_identity_media": True,
                    "git_tracked": False,
                }
            )
    return {
        "schema_version": "1.0",
        "dataset_id": "action-source-v1",
        "grammar_version": "grammar-v1",
        "status": "source",
        "records": records,
    }


def test_action_manifest_split_is_deterministic_and_participant_safe() -> None:
    manifest = _source_manifest()
    schema = json.loads(
        (
            ROOT
            / "configs/evaluation/stage2a_action_source_manifest.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(manifest, schema)
    first = assign_participant_splits(manifest, seed="fixed-seed")
    second = assign_participant_splits(manifest, seed="fixed-seed")

    assert first == second
    assert validate_action_manifest(first) == []
    participant_splits: dict[str, set[str]] = {}
    for record in first["records"]:
        participant_splits.setdefault(record["participant_code"], set()).add(
            record["split"]
        )
    assert all(len(splits) == 1 for splits in participant_splits.values())


def test_action_safety_replay_rejects_every_negative_and_recovers() -> None:
    report = run_action_safety_replay(250)

    assert report["result"] == "PASS"
    assert report["accepted_before_recovery"] == 0
    assert report["rejection_reasons"]["no_action"] == 251
    assert report["rejection_reasons"]["duplicate_observation"] == 1
    assert report["state_and_ledger_unchanged_before_recovery"] is True
    assert report["recovery_next_seat"] == "seat_a"
