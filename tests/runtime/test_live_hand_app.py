from __future__ import annotations

from pathlib import Path
import json

import pytest

from poker_dealer.robotics.dealer import DealerUnavailableError
from poker_dealer.runtime import (
    LiveHandApplication,
    RuntimeProfile,
    RuntimeResourceLocks,
)


ROOT = Path(__file__).resolve().parents[2]


def test_laptop_and_robot_camera_profiles_share_composition_root() -> None:
    laptop = LiveHandApplication(
        ROOT, RuntimeProfile.from_json(ROOT / "configs/runtime/laptop.json")
    )
    robot_camera = LiveHandApplication(
        ROOT, RuntimeProfile.from_json(ROOT / "configs/runtime/robot_camera.json")
    )
    assert laptop.preflight().ready is True
    assert robot_camera.preflight().ready is True
    assert laptop.preflight().dealer_adapter == "simulated"
    assert robot_camera.preflight().dealer_adapter == "simulated"
    assert laptop.resource_ids[0].startswith("camera:local:")
    assert robot_camera.resource_ids[0].startswith("camera:mjpeg:")
    assert laptop.event_log_path(session_id="s1", hand_id="h1") != (
        robot_camera.event_log_path(session_id="s1", hand_id="h1")
    )


def test_composition_root_loads_core_rules_instead_of_engine_defaults(
    tmp_path: Path,
) -> None:
    value = json.loads((ROOT / "configs/game/core_v1.json").read_text("utf-8"))
    value["session_defaults"]["starting_stack_units"] = 96
    value["betting"]["small_bet_units_default"] = 3
    value["betting"]["big_bet_units_default"] = 6
    config_path = tmp_path / "configs/game/core_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(value), encoding="utf-8")
    app = LiveHandApplication(
        tmp_path, RuntimeProfile.from_json(ROOT / "configs/runtime/laptop.json")
    )
    assert app.game_config.starting_stack_units == 96
    assert app.game_config.rules.small_bet_units == 3
    assert app.game_config.rules.big_bet_units == 6


def test_hardware_profile_is_declared_but_cannot_open() -> None:
    app = LiveHandApplication(
        ROOT, RuntimeProfile.from_json(ROOT / "configs/runtime/robot_hardware.json")
    )
    preflight = app.preflight()
    assert preflight.ready is False
    assert preflight.physical_motion is True
    assert preflight.full_live_hand_integrated is False
    with pytest.raises(DealerUnavailableError, match="safety"):
        app.open(open_camera=False)


def test_two_camera_profiles_can_lock_independently_when_one_does_not_use_mic(
    tmp_path: Path,
) -> None:
    laptop_profile = RuntimeProfile.from_json(
        ROOT / "configs/runtime/laptop.json"
    ).with_speech_override(enabled=False)
    robot_profile = RuntimeProfile.from_json(
        ROOT / "configs/runtime/robot_camera.json"
    ).with_speech_override(enabled=False)
    laptop = LiveHandApplication(ROOT, laptop_profile)
    robot = LiveHandApplication(ROOT, robot_profile)
    assert set(laptop.resource_ids).isdisjoint(robot.resource_ids)
    first = RuntimeResourceLocks(tmp_path, laptop.resource_ids).acquire()
    second = RuntimeResourceLocks(tmp_path, robot.resource_ids).acquire()
    second.release()
    first.release()
