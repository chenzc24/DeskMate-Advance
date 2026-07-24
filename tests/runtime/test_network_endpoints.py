from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from poker_dealer.runtime import NetworkEndpoints, RuntimeProfile


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/runtime/network_endpoints.json"


def test_declared_network_endpoints_validate_and_render_phone_url() -> None:
    schema = json.loads(
        (ROOT / "configs/contracts/network_endpoints.schema.json").read_text(
            encoding="utf-8"
        )
    )
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(raw)

    endpoints = NetworkEndpoints.from_json(CONFIG)

    assert endpoints.mobile_web_console.bind_host == "0.0.0.0"
    assert endpoints.mobile_web_console.browser_url == (
        "http://10.241.149.250:8765/"
    )
    assert endpoints.camera_stream_url("robot_camera") == (
        "http://100.80.46.54:5000/video_feed"
    )


def test_robot_profile_resolves_named_camera_stream_from_shared_config(
    tmp_path: Path,
) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["camera_streams"]["robot_camera"]["url"] = (
        "http://robot.local:5001/video_feed"
    )
    network_path = tmp_path / "network.json"
    network_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = RuntimeProfile.from_json(
        ROOT / "configs/runtime/robot_camera.json",
        network_endpoints_path=network_path,
    )

    assert profile.camera.stream_endpoint == "robot_camera"
    assert profile.camera.stream_url == "http://robot.local:5001/video_feed"


def test_network_config_rejects_credentials_and_unknown_endpoint() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["camera_streams"]["robot_camera"]["url"] = (
        "http://user:secret@robot.local/video_feed"
    )
    with pytest.raises(ValueError, match="credentials"):
        NetworkEndpoints.from_mapping(raw)

    endpoints = NetworkEndpoints.from_json(CONFIG)
    with pytest.raises(ValueError, match="unknown camera stream endpoint"):
        endpoints.camera_stream_url("missing")
