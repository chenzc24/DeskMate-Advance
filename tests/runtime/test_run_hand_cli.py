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


def test_speech_device_override_is_applied_before_resource_lock_selection() -> None:
    module = _load_script()
    args = module.parse_args(
        ["--profile", "laptop", "--speech-device", "test-microphone"]
    )
    profile = module._load_profile(args)
    app = module.LiveHandApplication(module.ROOT, profile)
    assert profile.speech_device == "test-microphone"
    assert "microphone:test-microphone" in app.resource_ids


def test_audiorelay_profile_and_announcer_arguments_are_resolved() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "laptop_audiorelay",
            "--announcer",
            "windows",
            "--announcement-tail-guard-ms",
            "500",
            "--announcement-voice",
            "Microsoft Zira Desktop",
        ]
    )
    profile = module._load_profile(args)
    assert profile.speech_device == "Virtual Mic (AudioRelay Wave)"
    assert profile.speech_capture_sample_rate_hz == 44_100
    assert args.announcer == "windows"
    assert args.announcement_tail_guard_ms == 500
    assert args.announcement_voice == "Microsoft Zira Desktop"


def test_robot_camera_audiorelay_profile_combines_mjpeg_and_native_audio() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "robot_camera_audiorelay",
            "--mode",
            "registration-smoke",
            "--button",
            "seat_a",
            "--consent-confirmed",
            "--announcer",
            "windows",
        ]
    )
    profile = module._load_profile(args)
    assert profile.camera.stream_url == "http://100.80.46.54:5000/video_feed"
    assert profile.speech_device == "Virtual Mic (AudioRelay Wave)"
    assert profile.speech_capture_sample_rate_hz == 44_100
    assert args.mode == "registration-smoke"


def test_registration_mobile_web_console_arguments_are_explicit() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "robot_camera_audiorelay",
            "--mode",
            "registration-smoke",
            "--button",
            "seat_a",
            "--consent-confirmed",
            "--web-console",
            "--headless",
            "--web-host",
            "127.0.0.1",
            "--web-port",
            "9876",
        ]
    )

    assert args.web_console is True
    assert args.headless is True
    assert args.web_host == "127.0.0.1"
    assert args.web_port == 9876
    assert args.web_url == "http://127.0.0.1:9876/"


def test_two_human_development_scenario_flag_is_explicit() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "robot_camera_audiorelay",
            "--mode",
            "live",
            "--button",
            "seat_a",
            "--consent-confirmed",
            "--development-two-human-ad",
        ]
    )

    assert args.development_two_human_ad is True


def test_network_config_supplies_default_web_and_camera_endpoints(
    tmp_path: Path,
) -> None:
    network_config = tmp_path / "network.json"
    network_config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "mobile_web_console": {
                    "bind_host": "0.0.0.0",
                    "advertised_host": "192.168.8.20",
                    "port": 9000,
                },
                "camera_streams": {
                    "robot_camera": {
                        "url": "http://192.168.8.30:5000/video_feed"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "robot_camera",
            "--network-config",
            str(network_config),
        ]
    )
    profile = module._load_profile(args)

    assert args.web_host == "0.0.0.0"
    assert args.web_port == 9000
    assert args.web_url == "http://192.168.8.20:9000/"
    assert profile.camera.stream_url == "http://192.168.8.30:5000/video_feed"


def test_registration_smoke_requires_consent_before_opening_devices(capsys) -> None:
    module = _load_script()
    assert module.main(
        [
            "--profile",
            "robot_camera_audiorelay",
            "--mode",
            "registration-smoke",
            "--button",
            "seat_a",
            "--announcer",
            "windows",
        ]
    ) == 1
    error = json.loads(capsys.readouterr().err)
    assert "--consent-confirmed is required" in error["reason"]


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


def test_replay_mode_qualifies_twenty_hand_session(tmp_path, capsys) -> None:
    module = _load_script()
    session_log = tmp_path / "session.jsonl"
    assert module.main(
        [
            "--profile",
            "laptop",
            "--mode",
            "replay",
            "--max-hands",
            "20",
            "--session-id",
            "qualification-session",
            "--session-log-jsonl",
            str(session_log),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["type"] == "session_replay"
    assert output["hands_completed"] == 20
    assert output["session_log_check_passed"] is True
    assert output["final_button"] == "seat_a"


def test_replay_step_limit_returns_checked_incomplete_result(tmp_path, capsys) -> None:
    module = _load_script()
    hand_log = tmp_path / "incomplete.jsonl"
    assert module.main(
        [
            "--profile",
            "laptop",
            "--mode",
            "replay",
            "--max-steps",
            "1",
            "--log-jsonl",
            str(hand_log),
        ]
    ) == 4
    output = json.loads(capsys.readouterr().out)
    assert output["completed"] is False
    assert output["log_check_passed"] is False
    assert output["session_log_check_passed"] is False


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


def test_live_mode_no_longer_has_operator_face_down_override() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--profile",
            "robot_camera",
            "--mode",
            "live",
            "--button",
            "seat_a",
            "--consent-confirmed",
        ]
    )
    assert not hasattr(args, "development_operator_face_down")


def test_live_asset_preflight_hashes_models_without_opening_devices(capsys) -> None:
    module = _load_script()
    assert module.main(
        ["--profile", "robot_camera", "--mode", "live-preflight"]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["assets_valid"] is True
    assert output["target_geometry_validated"] is False
    assert output["development_live_available"] is True
    assert output["full_live_hand_integrated"] is True
    assert output["card_binding_mode"] == "state_directed_full_frame"
    assert output["logical_card_slot_count"] == 13
    assert output["card_pixel_roi_count"] == 0
    assert output["announcement_language"] == "en-US"
    assert output["announcement_count"] >= 40
    assert output["speech_capture_sample_rate_hz"] == 16_000
    assert output["speech_resampling_enabled"] is False


def test_audiorelay_preflight_reports_native_capture_resampling(capsys) -> None:
    module = _load_script()
    assert module.main(
        ["--profile", "laptop_audiorelay", "--mode", "live-preflight"]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["speech_capture_sample_rate_hz"] == 44_100
    assert output["speech_model_sample_rate_hz"] == 16_000
    assert output["speech_resampling_enabled"] is True


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
