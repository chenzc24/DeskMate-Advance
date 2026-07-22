"""Pseudonymous local session records for four-player acceptance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping


SEATS = ("seat_a", "seat_b", "seat_c", "seat_d")
SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,47}$")


def _safe(value: str, label: str) -> str:
    if not SAFE_CODE.fullmatch(value):
        raise ValueError(
            f"{label} must be 2-48 characters containing only letters, digits, _ or -"
        )
    return value


def build_acceptance_session_record(
    *,
    session_group: str,
    operator_code: str,
    participant_codes: Mapping[str, str],
    consented_seats: set[str],
    lighting: str,
    camera_distance_cm: float | None,
    notes: str = "",
) -> dict[str, Any]:
    _safe(session_group, "session_group")
    _safe(operator_code, "operator_code")
    if set(participant_codes) != set(SEATS):
        raise ValueError("participant codes must cover seat_a through seat_d")
    resolved_codes = {seat: _safe(participant_codes[seat], seat) for seat in SEATS}
    if len(set(resolved_codes.values())) != 4:
        raise ValueError("four participant codes must be distinct")
    if not consented_seats <= set(SEATS):
        raise ValueError("consented_seats contains an unknown seat")
    if camera_distance_cm is not None and camera_distance_cm <= 0:
        raise ValueError("camera distance must be positive")
    if not lighting.strip():
        raise ValueError("lighting description is required")
    return {
        "schema_version": "1.0",
        "record_type": "stage2a_four_player_acceptance_session",
        "protocol_id": "stage2a-four-player-live-acceptance-v1",
        "session_group": session_group,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator_code": operator_code,
        "participants": {
            seat: {
                "participant_code": resolved_codes[seat],
                "consent_confirmed": seat in consented_seats,
            }
            for seat in SEATS
        },
        "environment": {
            "lighting": lighting,
            "camera_distance_cm": camera_distance_cm,
            "camera_index": 0,
            "camera_backend": "dshow",
            "speech_device": 1,
        },
        "privacy": {
            "contains_real_names": False,
            "save_frames": False,
            "save_audio": False,
            "persist_embeddings": False,
            "local_ignored_path_required": True,
        },
        "notes": notes,
    }


def validate_acceptance_session_record(
    record: Mapping[str, Any], *, require_all_consent: bool
) -> None:
    if record.get("schema_version") != "1.0":
        raise ValueError("unsupported session record schema")
    if record.get("record_type") != "stage2a_four_player_acceptance_session":
        raise ValueError("unexpected session record type")
    _safe(str(record.get("session_group", "")), "session_group")
    _safe(str(record.get("operator_code", "")), "operator_code")
    participants = record.get("participants")
    if not isinstance(participants, dict) or set(participants) != set(SEATS):
        raise ValueError("session record must contain exactly four seats")
    codes: list[str] = []
    for seat in SEATS:
        item = participants[seat]
        if not isinstance(item, dict):
            raise ValueError(f"invalid participant record for {seat}")
        codes.append(_safe(str(item.get("participant_code", "")), seat))
        if require_all_consent and item.get("consent_confirmed") is not True:
            raise ValueError(f"explicit consent is not confirmed for {seat}")
    if len(set(codes)) != 4:
        raise ValueError("participant codes must be distinct")
    privacy = record.get("privacy")
    required_privacy = {
        "contains_real_names": False,
        "save_frames": False,
        "save_audio": False,
        "persist_embeddings": False,
        "local_ignored_path_required": True,
    }
    if not isinstance(privacy, dict) or any(
        privacy.get(key) is not value for key, value in required_privacy.items()
    ):
        raise ValueError("session record privacy policy is invalid")


def load_acceptance_session_record(
    path: Path, *, require_all_consent: bool
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_acceptance_session_record(record, require_all_consent=require_all_consent)
    return record


def build_case_observation_record(
    *,
    session_group: str,
    session_id: str,
    case_id: str,
    operator_code: str,
    observed_result: str,
    handedness_used: str,
    camera_distance_cm: float,
    lighting: str,
    speech_used: bool,
    gesture_used: bool,
    failure_category: str,
    notes: str,
) -> dict[str, Any]:
    for value, label in (
        (session_group, "session_group"),
        (session_id, "session_id"),
        (operator_code, "operator_code"),
    ):
        _safe(value, label)
    if not re.fullmatch(r"FPA-0[0-8]", case_id):
        raise ValueError("case_id must be FPA-00 through FPA-08")
    if observed_result not in {"observed_pass", "observed_fail"}:
        raise ValueError("observed_result is invalid")
    if handedness_used not in {"left", "right", "both", "not_applicable"}:
        raise ValueError("handedness_used is invalid")
    if camera_distance_cm <= 0 or not lighting.strip():
        raise ValueError("positive camera distance and lighting are required")
    categories = {
        "none",
        "identity",
        "gesture",
        "speech",
        "fusion",
        "state",
        "ui",
        "environment",
        "crash",
    }
    if failure_category not in categories:
        raise ValueError("failure_category is invalid")
    if observed_result == "observed_pass" and failure_category != "none":
        raise ValueError("a manual pass must use failure_category=none")
    return {
        "schema_version": "1.0",
        "record_type": "stage2a_acceptance_case_observation",
        "session_group": session_group,
        "session_id": session_id,
        "case_id": case_id,
        "operator_code": operator_code,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator_completed": True,
        "observed_result": observed_result,
        "conditions": {
            "handedness_used": handedness_used,
            "camera_distance_cm": camera_distance_cm,
            "lighting": lighting,
            "speech_used": speech_used,
            "gesture_used": gesture_used,
        },
        "failure_category": failure_category,
        "notes": notes,
        "privacy": {
            "contains_real_names": False,
            "contains_raw_media": False,
            "local_ignored_path_required": True,
        },
    }


def validate_case_observation_record(
    record: Mapping[str, Any],
    *,
    session_group: str,
    session_id: str,
    case_id: str,
) -> None:
    if record.get("schema_version") != "1.0" or record.get("record_type") != (
        "stage2a_acceptance_case_observation"
    ):
        raise ValueError("invalid case observation record")
    if (
        record.get("session_group") != session_group
        or record.get("session_id") != session_id
        or record.get("case_id") != case_id
    ):
        raise ValueError("case observation context does not match JSONL evidence")
    if record.get("operator_completed") is not True:
        raise ValueError("case observation is not completed")
    privacy = record.get("privacy")
    if not isinstance(privacy, dict) or privacy != {
        "contains_real_names": False,
        "contains_raw_media": False,
        "local_ignored_path_required": True,
    }:
        raise ValueError("case observation privacy fields are invalid")
