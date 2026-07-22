"""Immutable Stage 2A action-source manifests and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


LABELS = {
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "no_action",
    "cancelled",
    "ambiguous",
    "occluded",
}
SEATS = {"seat_a", "seat_b", "seat_c", "seat_d"}
SPLITS = {"train", "validation", "test"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_action_manifest(
    manifest: Mapping[str, Any], *, root: Path | None = None, verify_files: bool = False
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("unsupported schema_version")
    if manifest.get("status") not in {"source", "resolved"}:
        errors.append("status must be source or resolved")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        return errors + ["records must be a non-empty list"]

    source_ids: set[str] = set()
    hashes: dict[str, str] = {}
    participant_splits: dict[str, str] = {}
    session_owners: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    resolved = manifest.get("status") == "resolved"
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        source_id = str(record.get("source_id", ""))
        participant = str(record.get("participant_code", ""))
        session = str(record.get("session_id", ""))
        if not source_id or source_id in source_ids:
            errors.append(f"{prefix} source_id is empty or duplicated")
        source_ids.add(source_id)
        if len(participant) < 2 or len(session) < 2:
            errors.append(f"{prefix} participant/session code is invalid")
        if record.get("seat") not in SEATS:
            errors.append(f"{prefix} seat is invalid")
        if record.get("label") not in LABELS:
            errors.append(f"{prefix} label is invalid")
        if record.get("contains_identity_media") is not True:
            errors.append(f"{prefix} must mark identity-bearing media")
        if record.get("git_tracked") is not False:
            errors.append(f"{prefix} raw media must be git_tracked=false")
        digest = str(record.get("sha256", ""))
        if not HEX64.fullmatch(digest):
            errors.append(f"{prefix} sha256 is invalid")
        elif digest in hashes and hashes[digest] != source_id:
            errors.append(f"{prefix} duplicates bytes from {hashes[digest]}")
        else:
            hashes[digest] = source_id
        if not isinstance(record.get("bytes"), int) or record.get("bytes", 0) <= 0:
            errors.append(f"{prefix} bytes must be positive")
        if not isinstance(record.get("duration_ms"), int) or record.get("duration_ms", 0) <= 0:
            errors.append(f"{prefix} duration_ms must be positive")
        capture_path = PurePosixPath(str(record.get("capture_path", "")))
        if capture_path.is_absolute() or ".." in capture_path.parts:
            errors.append(f"{prefix} capture_path must be a safe relative path")
        if tuple(capture_path.parts[:2]) != ("data", "raw"):
            errors.append(f"{prefix} capture_path must stay under data/raw")

        split = record.get("split")
        if resolved and split not in SPLITS:
            errors.append(f"{prefix} resolved record requires a split")
        if split in SPLITS:
            prior = participant_splits.setdefault(participant, str(split))
            if prior != split:
                errors.append(f"participant {participant} crosses splits")
            session_key = f"{participant}/{session}"
            prior_session = session_splits.setdefault(session_key, str(split))
            if prior_session != split:
                errors.append(f"session {session_key} crosses splits")
        prior_owner = session_owners.setdefault(session, participant)
        if prior_owner != participant:
            errors.append(f"session_id {session} is reused by multiple participants")

        if verify_files:
            if root is None:
                errors.append("root is required when verify_files=true")
            else:
                full_path = root.resolve() / Path(*capture_path.parts)
                if not full_path.is_file():
                    errors.append(f"{prefix} capture file is missing")
                else:
                    if full_path.stat().st_size != record.get("bytes"):
                        errors.append(f"{prefix} byte count mismatch")
                    if _file_sha256(full_path) != digest:
                        errors.append(f"{prefix} file hash mismatch")
    return errors


def assign_participant_splits(
    manifest: Mapping[str, Any],
    *,
    seed: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    if not seed:
        raise ValueError("split seed is required")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions are invalid")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below one")
    errors = validate_action_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    records = [dict(record) for record in manifest["records"]]
    participants = sorted({str(record["participant_code"]) for record in records})
    assignment: dict[str, str] = {}
    for participant in participants:
        bucket = int(
            hashlib.sha256(f"{seed}\0{participant}".encode("utf-8")).hexdigest()[:8],
            16,
        ) / 0xFFFFFFFF
        assignment[participant] = (
            "train"
            if bucket < train_fraction
            else (
                "validation"
                if bucket < train_fraction + validation_fraction
                else "test"
            )
        )
    for record in records:
        record["split"] = assignment[str(record["participant_code"])]
    source_hash = canonical_sha256(manifest["records"])
    resolved = {
        "schema_version": "1.0",
        "dataset_id": manifest["dataset_id"],
        "grammar_version": manifest["grammar_version"],
        "status": "resolved",
        "split_seed": seed,
        "source_records_sha256": source_hash,
        "records": records,
    }
    errors = validate_action_manifest(resolved)
    if errors:
        raise ValueError("resolved manifest is invalid: " + "; ".join(errors))
    return resolved
