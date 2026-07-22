from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from poker_dealer.training import (
    ActionTcnConfig,
    TrainingDependencyError,
    build_compact_tcn,
    make_sequence_windows,
    normalize_hand_landmarks,
    summarize_view_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/training/action_tcn_v1.json"


def test_config_and_landmark_normalization_contract() -> None:
    config = ActionTcnConfig.from_json(CONFIG_PATH)
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.1, 0.9, 21)
    landmarks[:, 1] = np.linspace(0.2, 0.8, 21)

    features = normalize_hand_landmarks(landmarks)

    assert config.status == "prepared_not_trained"
    assert features.shape == (63,)
    assert np.allclose(features[:3], 0.0)
    assert np.max(np.linalg.norm(features.reshape(21, 3)[:, :2], axis=1)) == pytest.approx(1.0)


def test_windows_pad_short_sequence_and_preserve_mask() -> None:
    features = np.ones((5, 63), dtype=np.float32)
    valid = np.asarray([True, True, False, True, True])

    windows, masks, starts = make_sequence_windows(
        features, valid, sequence_length=8, stride=2
    )

    assert windows.shape == (1, 8, 63)
    assert starts.tolist() == [0]
    assert masks[0].tolist() == [True, True, False, True, True, False, False, False]
    assert np.count_nonzero(windows[0, 5:]) == 0


def test_view_manifest_summary_rejects_participant_leakage() -> None:
    config = ActionTcnConfig.from_json(CONFIG_PATH)
    manifest = {
        "schema_version": "1.0",
        "status": "derived",
        "records": [
            {
                "participant_code": "P01",
                "session_id": "S01",
                "split": "train",
                "label": "call",
                "frames": 20,
            },
            {
                "participant_code": "P01",
                "session_id": "S02",
                "split": "test",
                "label": "call",
                "frames": 20,
            },
        ],
    }

    with pytest.raises(ValueError, match="crosses splits"):
        summarize_view_manifest(manifest, config)


def test_compact_tcn_has_explicit_optional_dependency() -> None:
    config = ActionTcnConfig.from_json(CONFIG_PATH)
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(TrainingDependencyError):
            build_compact_tcn(config)
    else:
        model = build_compact_tcn(config)
        assert model is not None
