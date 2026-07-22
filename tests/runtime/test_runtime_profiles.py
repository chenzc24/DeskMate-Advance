from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from poker_dealer.runtime import (
    DealerAdapterKind,
    RuntimeCameraKind,
    RuntimeProfile,
    RuntimeProfileId,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "profile_id", "camera_kind", "dealer_kind", "ready"),
    [
        ("laptop", RuntimeProfileId.LAPTOP, RuntimeCameraKind.LOCAL, DealerAdapterKind.SIMULATED, True),
        ("robot_camera", RuntimeProfileId.ROBOT_CAMERA, RuntimeCameraKind.MJPEG, DealerAdapterKind.SIMULATED, True),
        ("robot_hardware", RuntimeProfileId.ROBOT_HARDWARE, RuntimeCameraKind.MJPEG, DealerAdapterKind.REAL, False),
    ],
)
def test_declared_profiles_validate_and_freeze_dependency_combinations(
    name: str,
    profile_id: RuntimeProfileId,
    camera_kind: RuntimeCameraKind,
    dealer_kind: DealerAdapterKind,
    ready: bool,
) -> None:
    schema = json.loads(
        (ROOT / "configs/contracts/runtime_profile.schema.json").read_text(encoding="utf-8")
    )
    path = ROOT / "configs/runtime" / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(raw)
    profile = RuntimeProfile.from_json(path)
    assert profile.profile_id is profile_id
    assert profile.camera.kind is camera_kind
    assert profile.dealer.adapter is dealer_kind
    assert profile.dealer.enabled is ready
    assert profile.resolved_log_root(ROOT).is_relative_to(ROOT / "runs")


def test_runtime_parser_does_not_coerce_string_booleans() -> None:
    raw = json.loads((ROOT / "configs/runtime/laptop.json").read_text(encoding="utf-8"))
    raw["dealer"]["enabled"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        RuntimeProfile.from_mapping(raw)


def test_profile_rejects_cross_profile_camera_override() -> None:
    profile = RuntimeProfile.from_json(ROOT / "configs/runtime/robot_camera.json")
    with pytest.raises(ValueError, match="only valid for laptop"):
        profile.with_camera_override(device_index=1)
