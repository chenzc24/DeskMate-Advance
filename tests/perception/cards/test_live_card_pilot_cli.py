from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/perception/live_card_pilot.py"
SPEC = importlib.util.spec_from_file_location("live_card_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "stream_url": None,
        "index": None,
        "backend": None,
        "stream_open_timeout_ms": 5000,
        "stream_read_timeout_ms": 2000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pilot() -> SimpleNamespace:
    return SimpleNamespace(
        camera={
            "device_index": 1,
            "backend": "dshow",
            "width": 1280,
            "height": 720,
            "fps": 30.0,
        }
    )


def test_robot_stream_uses_bounded_ffmpeg_camera_boundary() -> None:
    config = MODULE._camera_config(
        _args(
            stream_url="http://100.80.46.54:5000/video_feed",
            stream_open_timeout_ms=4000,
            stream_read_timeout_ms=1500,
        ),
        _pilot(),
    )

    assert config.stream_url == "http://100.80.46.54:5000/video_feed"
    assert config.source_id == "robot_mjpeg_card_pilot"
    assert config.backend == "auto"
    assert config.width is None and config.height is None and config.fps is None
    assert config.open_timeout_ms == 4000
    assert config.read_timeout_ms == 1500


def test_robot_stream_rejects_competing_local_camera_options() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE._camera_config(
            _args(
                stream_url="http://robot.local:5000/video_feed",
                index=0,
            ),
            _pilot(),
        )
    with pytest.raises(ValueError, match="use FFmpeg"):
        MODULE._camera_config(
            _args(
                stream_url="http://robot.local:5000/video_feed",
                backend="dshow",
            ),
            _pilot(),
        )


def test_local_camera_defaults_remain_backward_compatible() -> None:
    config = MODULE._camera_config(_args(), _pilot())

    assert config.stream_url is None
    assert config.device_index == 1
    assert config.source_id == "laptop_card_pilot"
    assert config.backend == "dshow"
    assert (config.width, config.height, config.fps) == (1280, 720, 30.0)
