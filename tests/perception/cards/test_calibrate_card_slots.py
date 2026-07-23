from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/perception/calibrate_card_slots.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("calibrate_card_slots", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pixel_roi_is_normalized_without_claiming_geometry_validation() -> None:
    module = _load_script()
    assert module.normalized_roi((100, 50, 200, 100), width=1000, height=500) == {
        "x_min": 0.1,
        "y_min": 0.1,
        "x_max": 0.3,
        "y_max": 0.3,
    }
