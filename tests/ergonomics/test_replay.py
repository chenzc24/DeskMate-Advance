from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

import pytest

from deskmate_advance.temporal.ergonomics.replay import (
    ReplayFile,
    ReplayLimits,
    ReplayValidationError,
    sha256_file,
    validate_local_provenance,
)
from deskmate_advance.temporal.ergonomics.rules import (
    ErgonomicsEventConfig,
    ErgonomicsRuleEngine,
)


FIXTURE = Path("tests/fixtures/ergonomics/scalar-replay-valid-v1.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FEATURE_FILES = (
    "src/deskmate_advance/features/ergonomics/audio_live.py",
    "src/deskmate_advance/features/ergonomics/face.py",
    "src/deskmate_advance/features/ergonomics/live.py",
    "src/deskmate_advance/features/ergonomics/pose.py",
    "src/deskmate_advance/perception/ergonomics/landmarkers.py",
    "src/deskmate_advance/perception/ergonomics/observations.py",
    "src/deskmate_advance/perception/ergonomics/signals.py",
)


def _rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _validated(path: Path) -> ReplayFile:
    replay = ReplayFile(path, expected_sha256=sha256_file(path))
    replay.validate()
    return replay


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _local_provenance_root(tmp_path: Path) -> tuple[Path, Any]:
    """Create a tiny fully bound checkout without copying real model weights."""

    root = tmp_path / "checkout"
    event_config = json.loads(
        (PROJECT_ROOT / "configs/ergonomics/events.json").read_text(encoding="utf-8")
    )
    perception_config = json.loads(
        (PROJECT_ROOT / "configs/ergonomics/perception.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (PROJECT_ROOT / "models/manifest.yaml").read_text(encoding="utf-8")
    )
    base_header = ReplayFile(
        FIXTURE,
        expected_sha256=sha256_file(FIXTURE),
    ).header

    pose_bytes = b"test pose model bytes"
    face_bytes = b"test face model bytes"
    pose_sha256 = hashlib.sha256(pose_bytes).hexdigest()
    face_sha256 = hashlib.sha256(face_bytes).hexdigest()
    pose_relative = "models/assets/test-pose.task"
    face_relative = "models/assets/test-face.task"
    pose_path = root / pose_relative
    face_path = root / face_relative
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_bytes(pose_bytes)
    face_path.write_bytes(face_bytes)

    perception_config["pose"]["primary_asset"] = pose_relative
    perception_config["pose"]["primary_asset_sha256"] = pose_sha256
    perception_config["face"]["asset"] = face_relative
    perception_config["face"]["asset_sha256"] = face_sha256
    for model in manifest["models"]:
        if (
            model["model_id"] == "pose_landmarker"
            and model["version"] == base_header.provenance.pose_model.model_version
        ):
            model["local_path"] = pose_relative
            model["bytes"] = len(pose_bytes)
            model["sha256"] = pose_sha256
        if (
            model["model_id"] == "face_landmarker"
            and model["version"] == base_header.provenance.face_model.model_version
        ):
            model["local_path"] = face_relative
            model["bytes"] = len(face_bytes)
            model["sha256"] = face_sha256

    event_path = _write_json(root / "configs/ergonomics/events.json", event_config)
    perception_path = _write_json(
        root / "configs/ergonomics/perception.json", perception_config
    )
    manifest_path = _write_json(root / "models/manifest.yaml", manifest)
    for relative in _FEATURE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, target)

    provenance = replace(
        base_header.provenance,
        event_config_sha256=sha256_file(event_path),
        perception_config_sha256=sha256_file(perception_path),
        model_manifest_sha256=sha256_file(manifest_path),
        pose_model=replace(
            base_header.provenance.pose_model,
            asset_sha256=pose_sha256,
        ),
        face_model=replace(
            base_header.provenance.face_model,
            asset_sha256=face_sha256,
        ),
    )
    return root, replace(base_header, provenance=provenance)


def _with_provenance_hash(header: Any, field: str, value: str) -> Any:
    return replace(
        header,
        provenance=replace(header.provenance, **{field: value}),
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()


def test_valid_fixture_is_bounded_scalar_only_and_preserves_cached_age() -> None:
    replay = _validated(FIXTURE)
    samples = list(replay.iter_samples())

    assert replay.header.record_count == 3
    assert len(samples) == 3
    assert not hasattr(samples[0].snapshot.frame, "image")
    assert samples[0].snapshot.pose_ran is True
    assert samples[1].snapshot.pose_ran is False
    assert samples[1].snapshot.pose_features is samples[0].snapshot.pose_features
    assert samples[1].snapshot.pose_age_ms == pytest.approx(100.0)
    assert samples[1].snapshot.pose_stale is False
    assert samples[2].snapshot.frame.dropped_before == 1
    assert samples[2].snapshot.pose_features.temporal_gap is True
    assert samples[0].audio_level is not None
    assert samples[1].audio_level is None

    forbidden = {"image", "landmarks", "normalized_landmarks", "samples"}
    assert not (forbidden & _all_keys(_rows()))


def test_fixture_provenance_binds_configs_models_assets_and_feature_bundle() -> None:
    replay = ReplayFile(FIXTURE, expected_sha256=sha256_file(FIXTURE))

    validate_local_provenance(replay.header, project_root=PROJECT_ROOT)


def test_tiny_local_provenance_fixture_is_fully_bound(tmp_path: Path) -> None:
    root, header = _local_provenance_root(tmp_path)

    validate_local_provenance(header, project_root=root)


@pytest.mark.parametrize(
    ("relative", "message"),
    (
        ("configs/ergonomics/events.json", "event config SHA-256 mismatch"),
        (
            "configs/ergonomics/perception.json",
            "perception config SHA-256 mismatch",
        ),
        ("models/manifest.yaml", "model manifest SHA-256 mismatch"),
    ),
)
def test_each_bound_provenance_document_rejects_changed_bytes(
    tmp_path: Path,
    relative: str,
    message: str,
) -> None:
    root, header = _local_provenance_root(tmp_path)
    path = root / relative
    path.write_bytes(path.read_bytes() + b" \n")

    with pytest.raises(ReplayValidationError, match=message):
        validate_local_provenance(header, project_root=root)


def test_bound_documents_use_strict_json_after_hash_verification(
    tmp_path: Path,
) -> None:
    root, header = _local_provenance_root(tmp_path)
    event_path = root / "configs/ergonomics/events.json"
    duplicate = event_path.read_text(encoding="utf-8").replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0",\n  "schema_version": "1.0",',
        1,
    )
    event_path.write_text(duplicate, encoding="utf-8")
    duplicate_header = _with_provenance_hash(
        header,
        "event_config_sha256",
        sha256_file(event_path),
    )
    with pytest.raises(ReplayValidationError, match="duplicate JSON key"):
        validate_local_provenance(duplicate_header, project_root=root)

    root, header = _local_provenance_root(tmp_path / "non-finite")
    perception_path = root / "configs/ergonomics/perception.json"
    non_finite = perception_path.read_text(encoding="utf-8").replace(
        '"pose_hz": 10.0',
        '"pose_hz": NaN',
        1,
    )
    perception_path.write_text(non_finite, encoding="utf-8")
    non_finite_header = _with_provenance_hash(
        header,
        "perception_config_sha256",
        sha256_file(perception_path),
    )
    with pytest.raises(ReplayValidationError, match="non-finite JSON number"):
        validate_local_provenance(non_finite_header, project_root=root)

    root, header = _local_provenance_root(tmp_path / "deep")
    manifest_path = root / "models/manifest.yaml"
    manifest_path.write_bytes(b"[" * 5000 + b"0" + b"]" * 5000)
    deep_header = _with_provenance_hash(
        header,
        "model_manifest_sha256",
        sha256_file(manifest_path),
    )
    with pytest.raises(ReplayValidationError, match="nesting is too deep"):
        validate_local_provenance(deep_header, project_root=root)


def test_model_config_and_manifest_identities_must_be_complete(
    tmp_path: Path,
) -> None:
    root, header = _local_provenance_root(tmp_path)
    perception_path = root / "configs/ergonomics/perception.json"
    perception = json.loads(perception_path.read_text(encoding="utf-8"))
    del perception["pose"]["primary_model_id"]
    _write_json(perception_path, perception)
    incomplete_header = _with_provenance_hash(
        header,
        "perception_config_sha256",
        sha256_file(perception_path),
    )
    with pytest.raises(ReplayValidationError, match="identity is incomplete"):
        validate_local_provenance(incomplete_header, project_root=root)

    root, header = _local_provenance_root(tmp_path / "manifest-id")
    manifest_path = root / "models/manifest.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["model_id"] == "pose_landmarker" and (
            model["version"] == header.provenance.pose_model.model_version
        ):
            model["model_id"] = "wrong_pose_family"
    _write_json(manifest_path, manifest)
    wrong_id_header = _with_provenance_hash(
        header,
        "model_manifest_sha256",
        sha256_file(manifest_path),
    )
    with pytest.raises(ReplayValidationError, match="does not resolve uniquely"):
        validate_local_provenance(wrong_id_header, project_root=root)


def test_perception_and_manifest_asset_paths_must_match_after_normalization(
    tmp_path: Path,
) -> None:
    root, header = _local_provenance_root(tmp_path)
    perception_path = root / "configs/ergonomics/perception.json"
    perception = json.loads(perception_path.read_text(encoding="utf-8"))
    perception["pose"]["primary_asset"] = "models/assets/other-pose.task"
    _write_json(perception_path, perception)
    changed_header = _with_provenance_hash(
        header,
        "perception_config_sha256",
        sha256_file(perception_path),
    )

    with pytest.raises(ReplayValidationError, match="asset path does not match"):
        validate_local_provenance(
            changed_header,
            project_root=root,
            verify_assets=False,
        )


def test_feature_bundle_and_selected_asset_bytes_are_verified(tmp_path: Path) -> None:
    root, header = _local_provenance_root(tmp_path)
    feature_path = root / _FEATURE_FILES[0]
    feature_path.write_bytes(feature_path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ReplayValidationError, match="feature bundle SHA-256 mismatch"):
        validate_local_provenance(header, project_root=root)

    root, header = _local_provenance_root(tmp_path / "asset")
    (root / "models/assets/test-pose.task").write_bytes(b"changed model bytes")
    with pytest.raises(ReplayValidationError, match="asset is missing or has changed"):
        validate_local_provenance(header, project_root=root)


def test_scalar_snapshot_duck_types_the_current_rule_engine() -> None:
    replay = _validated(FIXTURE)
    config = ErgonomicsEventConfig.load(Path("configs/ergonomics/events.json"))
    engine = ErgonomicsRuleEngine(
        config,
        profile=replay.header.calibration_profile,
    )

    outputs = [
        engine.update(sample.snapshot, audio_level=sample.audio_level)
        for sample in replay.iter_samples()
    ]

    assert len(outputs) == 3
    assert outputs[0].source_id == replay.header.camera.source_id
    assert len(outputs[0].evaluations) == len(ErgonomicsRuleEngine.EVENT_NAMES)
    assert all(
        item.semantic_state.value == "unknown" for item in outputs[-1].evaluations
    )


def test_artifact_hash_is_checked_before_header_or_samples_are_used() -> None:
    with pytest.raises(ReplayValidationError, match="SHA-256 mismatch"):
        ReplayFile(FIXTURE, expected_sha256="0" * 64)


def test_duplicate_keys_and_non_finite_json_are_rejected(tmp_path: Path) -> None:
    original = FIXTURE.read_text(encoding="utf-8")
    duplicate = original.replace(
        '"sequence_id":0,',
        '"sequence_id":0,"sequence_id":0,',
        1,
    )
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ReplayValidationError, match="duplicate JSON key"):
        _validated(duplicate_path)

    non_finite = original.replace('"mean":100.0', '"mean":NaN', 1)
    non_finite_path = tmp_path / "nan.jsonl"
    non_finite_path.write_text(non_finite, encoding="utf-8")
    with pytest.raises(ReplayValidationError, match="non-finite JSON number"):
        _validated(non_finite_path)


Mutation = Callable[[list[dict[str, Any]]], None]


def _unknown_field(rows: list[dict[str, Any]]) -> None:
    rows[1]["unexpected"] = True


def _wrong_source(rows: list[dict[str, Any]]) -> None:
    rows[1]["source_id"] = "fixture-camera-b"


def _wrong_device(rows: list[dict[str, Any]]) -> None:
    rows[1]["device_index"] = 2


def _skipped_sequence(rows: list[dict[str, Any]]) -> None:
    rows[2]["sequence_id"] = 3


def _repeated_timestamp(rows: list[dict[str, Any]]) -> None:
    rows[2]["captured_at_ns"] = rows[1]["captured_at_ns"]


def _valid_state_conflict(rows: list[dict[str, Any]]) -> None:
    rows[1]["pose"]["update"]["valid"] = False


def _ran_update_conflict(rows: list[dict[str, Any]]) -> None:
    rows[1]["pose"]["ran"] = False


def _stale_age_conflict(rows: list[dict[str, Any]]) -> None:
    rows[1]["pose"]["stale"] = True


def _blink_mean_conflict(rows: list[dict[str, Any]]) -> None:
    rows[1]["face"]["update"]["eye_blink_mean"] = 0.2


def _audio_duration_conflict(rows: list[dict[str, Any]]) -> None:
    rows[1]["audio"]["window_started_at_ns"] += 1


def _timestamp_overflow(rows: list[dict[str, Any]]) -> None:
    rows[1]["captured_at_ns"] = 1 << 63


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_unknown_field, "fields do not match schema"),
        (_wrong_source, "camera source mismatch"),
        (_wrong_device, "camera device mismatch"),
        (_skipped_sequence, "sequence_id must be"),
        (_repeated_timestamp, "timestamps must increase strictly"),
        (_valid_state_conflict, "valid does not match state"),
        (_ran_update_conflict, "ran must match update presence"),
        (_stale_age_conflict, "stale flag does not match age"),
        (_blink_mean_conflict, "eye_blink_mean"),
        (_audio_duration_conflict, "audio window duration"),
        (_timestamp_overflow, "signed-64-bit"),
    ),
)
def test_strict_snapshot_contract_rejects_inconsistent_records(
    tmp_path: Path,
    mutate: Mutation,
    message: str,
) -> None:
    rows = deepcopy(_rows())
    mutate(rows)
    path = _write_rows(tmp_path / "invalid.jsonl", rows)

    with pytest.raises(ReplayValidationError, match=message):
        _validated(path)


def test_header_profile_hash_and_privacy_contracts_are_strict(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["provenance"]["event_config_sha256"] = "ABC"
    bad_hash = _write_rows(tmp_path / "bad-hash.jsonl", rows)
    with pytest.raises(ReplayValidationError, match="lowercase SHA-256"):
        ReplayFile(bad_hash, expected_sha256=sha256_file(bad_hash))

    rows = _rows()
    rows[0]["privacy"]["contains_landmarks"] = True
    bad_privacy = _write_rows(tmp_path / "bad-privacy.jsonl", rows)
    with pytest.raises(ReplayValidationError, match="must be false"):
        ReplayFile(bad_privacy, expected_sha256=sha256_file(bad_privacy))

    rows = _rows()
    rows[0]["calibration_profile"]["source_id"] = "fixture-camera-b"
    bad_profile = _write_rows(tmp_path / "bad-profile.jsonl", rows)
    with pytest.raises(ReplayValidationError, match="profile source mismatch"):
        ReplayFile(bad_profile, expected_sha256=sha256_file(bad_profile))


def test_line_and_record_limits_are_enforced_before_replay(tmp_path: Path) -> None:
    with pytest.raises(ReplayValidationError, match="exceeds"):
        ReplayFile(
            FIXTURE,
            expected_sha256=sha256_file(FIXTURE),
            limits=ReplayLimits(max_line_bytes=100, max_records=10),
        )
    with pytest.raises(ReplayValidationError, match="record_count exceeds"):
        ReplayFile(
            FIXTURE,
            expected_sha256=sha256_file(FIXTURE),
            limits=ReplayLimits(max_line_bytes=32 * 1024, max_records=2),
        )

    rows = _rows()
    rows[0]["record_count"] = 4
    mismatched = _write_rows(tmp_path / "count.jsonl", rows)
    replay = ReplayFile(mismatched, expected_sha256=sha256_file(mismatched))
    with pytest.raises(ReplayValidationError, match="record count does not match"):
        replay.validate()


def test_repeated_audio_window_cannot_change_level_evidence(tmp_path: Path) -> None:
    rows = deepcopy(_rows())
    repeated = deepcopy(rows[1]["audio"])
    repeated["rms"] = 0.1
    repeated["dbfs"] = -20.0
    rows[2]["audio"] = repeated
    path = _write_rows(tmp_path / "changed-audio-window.jsonl", rows)

    with pytest.raises(ReplayValidationError, match="audio window changed content"):
        _validated(path)


def test_nonzero_audio_rms_and_dbfs_must_be_mathematically_consistent(
    tmp_path: Path,
) -> None:
    rows = deepcopy(_rows())
    rows[1]["audio"]["dbfs"] = -39.0
    path = _write_rows(tmp_path / "inconsistent-audio-level.jsonl", rows)

    with pytest.raises(ReplayValidationError, match="rms and dbfs are inconsistent"):
        _validated(path)


@pytest.mark.parametrize(
    ("rms", "dbfs", "message"),
    (
        (2.0, 20.0 * math.log10(2.0), "at most 1.0"),
        (1.0, 0.1, "at most 0.0"),
    ),
)
def test_normalized_audio_levels_cannot_exceed_full_scale(
    tmp_path: Path,
    rms: float,
    dbfs: float,
    message: str,
) -> None:
    rows = deepcopy(_rows())
    rows[1]["audio"]["rms"] = rms
    rows[1]["audio"]["dbfs"] = dbfs
    path = _write_rows(tmp_path / "above-full-scale.jsonl", rows)

    with pytest.raises(ReplayValidationError, match=message):
        _validated(path)


def test_validate_cli_reports_compact_deterministic_summary() -> None:
    command = [
        sys.executable,
        "scripts/ergonomics/replay_part_a.py",
        "validate",
        str(FIXTURE),
        "--sha256",
        sha256_file(FIXTURE),
    ]
    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["mode"] == "validate"
    assert payload["records"] == 3
    assert payload["privacy"]["declared_contains_images"] is False
    assert payload["privacy"]["structural_scalar_schema_verified"] is True
    assert payload["privacy"]["direct_identifier_absence"] == "declared_not_verified"
