import hashlib
import json
from pathlib import Path

import pytest

from deskmate_advance.features.ergonomics.benchmark import (
    ComponentMetrics,
    Reservoir,
    iter_timestamp_sidecar,
    load_recording_manifest,
)
from deskmate_advance.perception.ergonomics import ObservationState


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_row(
    *,
    media_hash: str,
    timestamp_hash: str,
    sample_id: str = "sample-a",
    split: str = "selection",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "local_path": "data/raw/ergonomics/sample.mp4",
        "sha256": media_hash,
        "timestamp_sidecar_path": "data/raw/ergonomics/sample.timestamps.jsonl",
        "timestamp_sidecar_sha256": timestamp_hash,
        "participant_id": "participant-a",
        "session_id": "session-a",
        "device_id": "camera-a",
        "scenario": "neutral_seated",
        "scenario_tags": ["exploratory_camera", "neutral"],
        "split": split,
        "consent_status": "confirmed",
        "license_status": "project_recording_approved",
    }


def test_manifest_verifies_owned_files_and_hashes(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "ergonomics"
    raw.mkdir(parents=True)
    media = b"video"
    timestamps = b'{"frame_index":0,"captured_at_ns":1000000}\n'
    (raw / "sample.mp4").write_bytes(media)
    (raw / "sample.timestamps.jsonl").write_bytes(timestamps)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            _manifest_row(
                media_hash=_sha(media),
                timestamp_hash=_sha(timestamps),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_recording_manifest(manifest, project_root=tmp_path)

    assert len(records) == 1
    assert records[0].sample_id == "sample-a"
    assert records[0].media_path == (raw / "sample.mp4").resolve()


def test_manifest_rejects_unconfirmed_or_cross_split_duplicates(
    tmp_path: Path,
) -> None:
    media_hash = "a" * 64
    timestamp_hash = "b" * 64
    first = _manifest_row(media_hash=media_hash, timestamp_hash=timestamp_hash)
    second = _manifest_row(
        media_hash=media_hash,
        timestamp_hash=timestamp_hash,
        sample_id="sample-b",
        split="test",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple splits"):
        load_recording_manifest(
            manifest,
            project_root=tmp_path,
            verify_files=False,
        )


def test_timestamp_sidecar_requires_contiguous_indices_and_distinct_ms(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        '{"frame_index":0,"captured_at_ns":1000000,"dropped_before":0}\n'
        '{"frame_index":1,"captured_at_ns":34000000,"dropped_before":2}\n',
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"frame_index":0,"captured_at_ns":1000000}\n'
        '{"frame_index":1,"captured_at_ns":1900000}\n',
        encoding="utf-8",
    )

    entries = list(iter_timestamp_sidecar(valid))

    assert [item.frame_index for item in entries] == [0, 1]
    assert entries[1].dropped_before == 2
    with pytest.raises(ValueError, match="millisecond resolution"):
        list(iter_timestamp_sidecar(invalid))


def test_reservoir_and_component_metrics_handle_empty_and_states() -> None:
    reservoir = Reservoir(capacity=2, seed=1)
    assert reservoir.summary()["p95"] is None
    for value in (1.0, 2.0, 3.0, 4.0):
        reservoir.add(value)
    summary = reservoir.summary()
    assert summary["seen"] == 4
    assert summary["sampled"] == 2
    assert summary["p50"] is not None

    metrics = ComponentMetrics(("feature",))
    metrics.begin_recording()
    metrics.add(
        state=ObservationState.MISSING,
        timestamp_ns=1_000_000,
        inference_ms=2.0,
        dropped_before=0,
        available={"feature": False},
    )
    metrics.add(
        state=ObservationState.VALID,
        timestamp_ns=51_000_000,
        inference_ms=3.0,
        dropped_before=1,
        available={"feature": True},
    )
    result = metrics.summary()
    assert result["state_counts"]["missing"] == 1
    assert result["state_rates"]["valid"] == pytest.approx(0.5)
    assert result["feature_available_rates"]["feature"] == pytest.approx(0.5)
    assert result["max_contiguous_invalid_ms"] == pytest.approx(50.0)
    assert result["dropped_before_total"] == 1
