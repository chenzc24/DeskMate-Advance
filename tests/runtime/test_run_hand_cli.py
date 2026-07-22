from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runtime/run_hand.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_hand_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_laptop_config_check_succeeds_without_opening_devices(capsys) -> None:
    module = _load_script()
    assert module.main(["--profile", "laptop", "--check-config"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["type"] == "runtime_preflight"
    assert output["ready"] is True
    assert output["full_live_hand_integrated"] is False


def test_real_hardware_config_check_fails_closed(capsys) -> None:
    module = _load_script()
    assert module.main(["--profile", "robot_hardware", "--check-config"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is False
    assert output["physical_motion"] is True
    assert "safety" in output["reason"].lower()


def test_replay_mode_completes_hand_without_opening_devices(tmp_path, capsys) -> None:
    module = _load_script()
    path = tmp_path / "replay.jsonl"
    assert module.main(
        [
            "--profile",
            "laptop",
            "--mode",
            "replay",
            "--log-jsonl",
            str(path),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["type"] == "hand_replay"
    assert output["completed"] is True
    assert output["log_check_passed"] is True
    assert output["physical_motion"] is False
    assert path.exists()


def test_exact_replay_rejects_context_override_before_output(
    tmp_path, capsys
) -> None:
    module = _load_script()
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "must-not-exist.jsonl"
    assert module.main(
        [
            "--profile",
            "laptop",
            "--mode",
            "replay",
            "--session-id",
            "source-session",
            "--hand-id",
            "source-hand",
            "--log-jsonl",
            str(source),
        ]
    ) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--profile",
            "robot_camera",
            "--mode",
            "replay",
            "--replay-log",
            str(source),
            "--hand-id",
            "different-hand",
            "--log-jsonl",
            str(destination),
        ]
    ) == 1
    error = json.loads(capsys.readouterr().err)
    assert "hand_id must match" in error["reason"]
    assert not destination.exists()


def test_live_mode_refuses_unvalidated_hole_orientation_before_device_open(
    capsys,
) -> None:
    module = _load_script()
    assert module.main(
        [
            "--profile",
            "robot_camera",
            "--mode",
            "live",
            "--button",
            "seat_a",
            "--consent-confirmed",
        ]
    ) == 1
    error = json.loads(capsys.readouterr().err)
    assert "face-down occupancy/orientation model is not admitted" in error["reason"]


def test_live_asset_preflight_hashes_models_without_opening_devices(capsys) -> None:
    module = _load_script()
    assert module.main(
        ["--profile", "robot_camera", "--mode", "live-preflight"]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["assets_valid"] is True
    assert output["target_geometry_validated"] is False
    assert output["development_live_available"] is True
    assert output["full_live_hand_integrated"] is False


def test_hardware_profile_cannot_be_used_for_replay_fallback(tmp_path, capsys) -> None:
    module = _load_script()
    assert module.main(
        [
            "--profile",
            "robot_hardware",
            "--mode",
            "replay",
            "--log-jsonl",
            str(tmp_path / "must-not-exist.jsonl"),
        ]
    ) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is False
    assert output["physical_motion"] is True
    assert not (tmp_path / "must-not-exist.jsonl").exists()
