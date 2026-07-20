import hashlib
import json
from pathlib import Path

import pytest

from deskmate_advance.temporal.ergonomics.annotations import (
    ANNOTATION_SCHEMA_NAME,
    ANNOTATION_SCHEMA_VERSION,
    AnnotationJsonFile,
    AnnotationLimits,
    AnnotationValidationError,
)
from deskmate_advance.temporal.ergonomics.evaluation import AnnotationLabel


REPLAY_SHA256 = "a" * 64
SOURCE_ID = "fixture-camera-a"


def _payload() -> dict[str, object]:
    return {
        "schema": ANNOTATION_SCHEMA_NAME,
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_set_id": "fixture-annotations-v1",
        "replay_sha256": REPLAY_SHA256,
        "source_id": SOURCE_ID,
        "clock": {
            "kind": "monotonic",
            "origin": "session_relative",
            "unit": "ns",
        },
        "privacy": {"contains_direct_identifiers": False},
        "annotations": [
            {
                "event_id": "static-positive-1",
                "event_name": "static_too_long",
                "label": "positive",
                "onset_ns": 1_000_000_000,
                "offset_ns": 4_000_000_000,
                "eligible_at_ns": 2_000_000_000,
            },
            {
                "event_id": "static-negative-1",
                "event_name": "static_too_long",
                "label": "negative",
                "onset_ns": 5_000_000_000,
                "offset_ns": 6_000_000_000,
                "eligible_at_ns": None,
            },
        ],
    }


def _write(path: Path, payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_annotation_artifact_loads_and_binds_exact_replay_and_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.json"
    artifact_sha256 = _write(path, _payload())

    result = AnnotationJsonFile(
        path,
        expected_sha256=artifact_sha256,
    ).load(replay_sha256=REPLAY_SHA256, source_id=SOURCE_ID)

    assert result.annotation_set_id == "fixture-annotations-v1"
    assert result.artifact_sha256 == artifact_sha256
    assert len(result.annotations) == 2
    assert result.annotations[0].label is AnnotationLabel.POSITIVE
    assert result.annotations[1].label is AnnotationLabel.NEGATIVE


def test_annotation_artifact_rejects_wrong_bytes_or_replay_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.json"
    artifact_sha256 = _write(path, _payload())

    with pytest.raises(AnnotationValidationError, match="SHA-256"):
        AnnotationJsonFile(path, expected_sha256="b" * 64).load(
            replay_sha256=REPLAY_SHA256,
            source_id=SOURCE_ID,
        )

    with pytest.raises(AnnotationValidationError, match="different replay"):
        AnnotationJsonFile(path, expected_sha256=artifact_sha256).load(
            replay_sha256="c" * 64,
            source_id=SOURCE_ID,
        )

    with pytest.raises(AnnotationValidationError, match="source"):
        AnnotationJsonFile(path, expected_sha256=artifact_sha256).load(
            replay_sha256=REPLAY_SHA256,
            source_id="different-camera",
        )


def test_annotation_artifact_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    raw = (
        b'{"schema":"deskmate.ergonomics.annotations",'
        b'"schema":"deskmate.ergonomics.annotations"}'
    )
    path.write_bytes(raw)

    with pytest.raises(AnnotationValidationError, match="duplicate"):
        AnnotationJsonFile(
            path,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ).load(replay_sha256=REPLAY_SHA256, source_id=SOURCE_ID)


def test_annotation_artifact_rejects_overlapping_lane_intervals(
    tmp_path: Path,
) -> None:
    payload = _payload()
    rows = payload["annotations"]
    assert isinstance(rows, list)
    second = rows[1]
    assert isinstance(second, dict)
    second["onset_ns"] = 3_000_000_000
    path = tmp_path / "overlap.json"
    artifact_sha256 = _write(path, payload)

    with pytest.raises(AnnotationValidationError, match="overlapping"):
        AnnotationJsonFile(path, expected_sha256=artifact_sha256).load(
            replay_sha256=REPLAY_SHA256,
            source_id=SOURCE_ID,
        )


def test_annotation_artifact_enforces_configured_byte_and_record_limits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.json"
    artifact_sha256 = _write(path, _payload())

    with pytest.raises(AnnotationValidationError, match="byte limit"):
        AnnotationJsonFile(
            path,
            expected_sha256=artifact_sha256,
            limits=AnnotationLimits(max_file_bytes=16, max_annotations=10),
        ).load(replay_sha256=REPLAY_SHA256, source_id=SOURCE_ID)

    with pytest.raises(AnnotationValidationError, match="count"):
        AnnotationJsonFile(
            path,
            expected_sha256=artifact_sha256,
            limits=AnnotationLimits(max_file_bytes=4096, max_annotations=1),
        ).load(replay_sha256=REPLAY_SHA256, source_id=SOURCE_ID)
