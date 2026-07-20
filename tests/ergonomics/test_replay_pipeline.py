from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Collection
from uuid import uuid4

import pytest

from deskmate_advance.temporal.ergonomics.replay import sha256_file
from deskmate_advance.temporal.ergonomics.rules import ErgonomicsRuleEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_FIXTURE = Path("tests/fixtures/ergonomics/scalar-replay-valid-v1.jsonl")
ABNORMAL_UNTIL_MS = 35_000
RECOVERY_BLINKS_UNTIL_MS = 39_000
SAMPLE_PERIOD_MS = 250


def _header() -> dict[str, Any]:
    return json.loads(BASE_FIXTURE.read_text(encoding="utf-8").splitlines()[0])


def _sample(
    sequence_id: int,
    captured_at_ns: int,
    *,
    active_events: Collection[str],
    blink_score: float,
) -> dict[str, Any]:
    camera_source = "fixture-camera-a"
    pose = {
        "ran": True,
        "stale": False,
        "update": {
            "captured_at_ns": captured_at_ns,
            "state": "valid",
            "valid": True,
            "reason": None,
            "temporal_gap": False,
            "shoulder_tilt_deg": (
                20.0 if "bad_posture" in active_events else 0.0
            ),
            "torso_lean_from_vertical_deg": (
                20.0 if "bad_posture" in active_events else 0.0
            ),
            "upper_body_motion_per_second": (
                0.01 if "static_too_long" in active_events else 0.2
            ),
        },
    }
    face = {
        "ran": True,
        "stale": False,
        "update": {
            "captured_at_ns": captured_at_ns,
            "state": "valid",
            "valid": True,
            "reason": None,
            "geometry_state": "valid",
            "rotation_state": "valid",
            "blink_state": "valid",
            "face_bbox_area_ratio": (
                0.2 if "screen_too_close" in active_events else 0.1
            ),
            "raw_rotation_xyz_deg": (
                [30.0, 0.0, 0.0]
                if "head_off_center" in active_events
                else [0.0, 0.0, 0.0]
            ),
            "eye_blink_left": blink_score,
            "eye_blink_right": blink_score,
            "eye_blink_mean": blink_score,
        },
    }
    luminance = {
        "ran": True,
        "stale": False,
        "update": {
            "captured_at_ns": captured_at_ns,
            "state": "valid",
            "valid": True,
            "reason": None,
            "mean": (
                20.0 if "environment_too_dark" in active_events else 100.0
            ),
            "p90": (
                250.0 if "environment_too_bright" in active_events else 150.0
            ),
        },
    }
    return {
        "record_type": "snapshot",
        "schema_version": "1.0",
        "source_id": camera_source,
        "device_index": 0,
        "sequence_id": sequence_id,
        "captured_at_ns": captured_at_ns,
        "dropped_before": 0,
        "pose": pose,
        "face": face,
        "luminance": luminance,
        "audio": {
            "source_id": "fixture-microphone-a",
            "device_index": 1,
            "state": "valid",
            "valid": True,
            "stale": False,
            "reason": None,
            "window_started_at_ns": captured_at_ns - 250_000_000,
            "window_ended_at_ns": captured_at_ns,
            "sample_rate_hz": 16_000,
            "sample_count": 4_000,
            "rms": 0.3 if "noise_too_high" in active_events else 0.01,
            "dbfs": (
                20.0 * math.log10(0.3)
                if "noise_too_high" in active_events
                else -40.0
            ),
        },
    }


def _blink_score(
    relative_ms: int,
    sequence_id: int,
    *,
    low_blink_rate_is_target: bool,
) -> float:
    if low_blink_rate_is_target:
        if relative_ms < ABNORMAL_UNTIL_MS:
            return 0.1
        if relative_ms < RECOVERY_BLINKS_UNTIL_MS:
            return 0.6 if sequence_id % 2 == 0 else 0.1
        return 0.1
    # A 250 ms closed pulse every four seconds yields valid normal-rate blinks.
    # This prevents unrelated single-lane tests from accidentally activating the
    # low-blink lane once its 30-second evidence window becomes available.
    return 0.6 if relative_ms % 4_000 == SAMPLE_PERIOD_MS else 0.1


def _write_contract_replay(
    path: Path,
    *,
    target_events: Collection[str] = ErgonomicsRuleEngine.EVENT_NAMES,
    temporal_gap_at_ms: int | None = None,
) -> None:
    target_event_set = frozenset(target_events)
    assert target_event_set <= frozenset(ErgonomicsRuleEngine.EVENT_NAMES)
    rows: list[dict[str, Any]] = []
    for sequence_id, relative_ms in enumerate(
        range(0, 42_001, SAMPLE_PERIOD_MS)
    ):
        active_events = (
            target_event_set if relative_ms < ABNORMAL_UNTIL_MS else frozenset()
        )
        row = _sample(
            sequence_id,
            1_000_000_000 + relative_ms * 1_000_000,
            active_events=active_events,
            blink_score=_blink_score(
                relative_ms,
                sequence_id,
                low_blink_rate_is_target="low_blink_rate" in target_event_set,
            ),
        )
        if relative_ms == temporal_gap_at_ms:
            row["dropped_before"] = 1
            row["pose"]["update"]["temporal_gap"] = True
        rows.append(row)
    header = deepcopy(_header())
    lane_id = (
        "all-lanes"
        if target_event_set == frozenset(ErgonomicsRuleEngine.EVENT_NAMES)
        else "-".join(sorted(target_event_set))
    )
    gap_suffix = "-gap" if temporal_gap_at_ms is not None else ""
    header["replay_id"] = f"synthetic-a3-{lane_id}{gap_suffix}-v1"
    header["record_count"] = len(rows)
    path.write_text(
        "".join(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for item in (header, *rows)
        ),
        encoding="utf-8",
    )


def _run(replay: Path, candidates: Path, summary: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/ergonomics/replay_part_a.py",
            "run",
            str(replay),
            "--sha256",
            sha256_file(replay),
            "--data-status",
            "synthetic_contract_test",
            "--candidates",
            str(candidates),
            "--summary",
            str(summary),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _artifact_dir() -> Path:
    path = PROJECT_ROOT / "artifacts" / "pytest-a3" / uuid4().hex
    path.mkdir(parents=True)
    return path


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("target_event", ErgonomicsRuleEngine.EVENT_NAMES)
def test_each_event_lane_activates_without_cross_wiring(
    tmp_path: Path,
    target_event: str,
) -> None:
    replay = tmp_path / f"{target_event}.jsonl"
    _write_contract_replay(replay, target_events=(target_event,))
    output_dir = _artifact_dir()
    candidates = output_dir / "candidates.jsonl"
    summary = output_dir / "summary.json"

    result = _run(replay, candidates, summary)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    rows = _candidate_rows(candidates)
    for event_name in ErgonomicsRuleEngine.EVENT_NAMES:
        lifecycle = [
            row["transition"]
            for row in rows
            if row["event_name"] == event_name
            and row["transition"] in {"start", "clear"}
        ]
        expected = ["start", "clear"] if event_name == target_event else []
        assert lifecycle == expected
        assert payload["candidate_events"][event_name].get("start", 0) == (
            1 if event_name == target_event else 0
        )
        assert payload["candidate_events"][event_name].get("clear", 0) == (
            1 if event_name == target_event else 0
        )
    shutil.rmtree(output_dir)


def test_scalar_replay_drives_all_lanes_candidates_and_continuous_metrics(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "all-lanes.jsonl"
    _write_contract_replay(replay)
    output_dir = _artifact_dir()
    first_candidates = output_dir / "candidates-first.jsonl"
    first_summary = output_dir / "summary-first.json"
    first = _run(replay, first_candidates, first_summary)

    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["continuous_evaluation"]["contract_only"] is True
    assert payload["continuous_evaluation"]["formal_effect_metric_eligible"] is False
    assert payload["continuous_evaluation"]["parallel_max_active"] == 8
    for event_name in ErgonomicsRuleEngine.EVENT_NAMES:
        assert payload["candidate_events"][event_name]["start"] == 1
        assert payload["candidate_events"][event_name]["clear"] == 1

    candidate_rows = _candidate_rows(first_candidates)
    assert candidate_rows
    assert all(item["confidence"]["value"] is None for item in candidate_rows)
    assert not any(
        forbidden in item
        for item in candidate_rows
        for forbidden in ("suggested_action", "motor_speed", "servo_angle")
    )
    for event_name in ErgonomicsRuleEngine.EVENT_NAMES:
        lane = [item for item in candidate_rows if item["event_name"] == event_name]
        start = next(item for item in lane if item["transition"] == "start")
        clear = next(item for item in lane if item["transition"] == "clear")
        assert start["episode_id"] == clear["episode_id"]
        assert clear["duration_ms"] > 0

    second_candidates = output_dir / "candidates-second.jsonl"
    second_summary = output_dir / "summary-second.json"
    second = _run(replay, second_candidates, second_summary)
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    assert first_candidates.read_bytes() == second_candidates.read_bytes()
    assert first_summary.read_bytes() == second_summary.read_bytes()
    shutil.rmtree(output_dir)


def test_temporal_gap_retains_episode_and_excludes_unknown_intervals_from_duration(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "static-with-gap.jsonl"
    gap_relative_ms = 33_000
    _write_contract_replay(
        replay,
        target_events=("static_too_long",),
        temporal_gap_at_ms=gap_relative_ms,
    )
    output_dir = _artifact_dir()
    candidates = output_dir / "candidates.jsonl"
    summary = output_dir / "summary.json"

    result = _run(replay, candidates, summary)

    assert result.returncode == 0, result.stderr
    rows = [
        row
        for row in _candidate_rows(candidates)
        if row["event_name"] == "static_too_long"
    ]
    start = next(row for row in rows if row["transition"] == "start")
    gap_observed_at_ns = 1_000_000_000 + gap_relative_ms * 1_000_000
    unknown = next(
        row
        for row in rows
        if row["transition"] == "unknown"
        and row["observed_at_ns"] == gap_observed_at_ns
    )
    recovery = next(
        row
        for row in rows
        if row["transition"] == "update"
        and row["observed_at_ns"]
        == gap_observed_at_ns + SAMPLE_PERIOD_MS * 1_000_000
    )
    clear = next(row for row in rows if row["transition"] == "clear")

    assert unknown["reason_code"] == (
        "evidence_discontinuity:camera_frames_dropped:1"
    )
    assert not any(
        row["transition"] == "clear"
        and row["observed_at_ns"] == gap_observed_at_ns
        for row in rows
    )
    assert {
        start["episode_id"],
        unknown["episode_id"],
        recovery["episode_id"],
        clear["episode_id"],
    } == {start["episode_id"]}
    assert unknown["duration_ms"] == recovery["duration_ms"]

    wall_duration_ms = (
        clear["observed_at_ns"] - start["observed_at_ns"]
    ) / 1_000_000
    excluded_unknown_ms = 2 * SAMPLE_PERIOD_MS
    assert wall_duration_ms == 26_000
    assert clear["duration_ms"] == wall_duration_ms - excluded_unknown_ms
    shutil.rmtree(output_dir)


def test_run_refuses_colliding_or_existing_outputs(tmp_path: Path) -> None:
    replay = tmp_path / "all-lanes.jsonl"
    _write_contract_replay(replay)
    output_dir = _artifact_dir()
    output = output_dir / "same.json"
    command = [
        sys.executable,
        "scripts/ergonomics/replay_part_a.py",
        "run",
        str(replay),
        "--sha256",
        sha256_file(replay),
        "--candidates",
        str(output),
        "--summary",
        str(output),
    ]
    collision = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert collision.returncode == 2
    assert "distinct" in collision.stderr
    assert not output.exists()

    output.write_text("keep", encoding="utf-8")
    existing = _run(replay, output_dir / "new-candidates.jsonl", output)
    assert existing.returncode == 2
    assert "already exists" in existing.stderr
    assert output.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / "new-candidates.jsonl").exists()
    shutil.rmtree(output_dir)


def test_run_outputs_are_confined_and_cannot_alias_replay(tmp_path: Path) -> None:
    replay = tmp_path / "all-lanes.jsonl"
    _write_contract_replay(replay)
    external = _run(
        replay,
        tmp_path / "external-candidates.jsonl",
        tmp_path / "external-summary.json",
    )
    assert external.returncode == 2
    assert "must stay under" in external.stderr
    assert not (tmp_path / "external-candidates.jsonl").exists()

    output_dir = _artifact_dir()
    protected_replay = output_dir / "input.jsonl"
    shutil.copyfile(replay, protected_replay)
    original_sha256 = sha256_file(protected_replay)
    alias_command = [
        sys.executable,
        "scripts/ergonomics/replay_part_a.py",
        "run",
        str(protected_replay),
        "--sha256",
        original_sha256,
        "--candidates",
        str(protected_replay),
        "--summary",
        str(output_dir / "summary.json"),
        "--overwrite",
    ]
    alias = subprocess.run(
        alias_command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert alias.returncode == 2
    assert "protected input" in alias.stderr
    assert sha256_file(protected_replay) == original_sha256
    assert not (output_dir / "summary.json").exists()
    shutil.rmtree(output_dir)
