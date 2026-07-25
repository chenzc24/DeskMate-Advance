from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.models.freeze_model_assets import build_snapshot, verify_snapshot


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_snapshot_covers_tracked_assets_and_detects_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitattributes").write_text(
        "models/assets/demo/best.pt filter=lfs\n",
        encoding="utf-8",
    )
    asset_dir = tmp_path / "models/assets/demo"
    asset_dir.mkdir(parents=True)
    (asset_dir / "best.pt").write_bytes(b"weights")
    (asset_dir / "classes.json").write_text('{"0":"demo"}\n', encoding="utf-8")
    (tmp_path / "models/manifest.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "demo",
                        "version": "v1",
                        "state": "development",
                        "asset": {"path": "models/assets/demo"},
                    },
                    {
                        "model_id": "future",
                        "version": "untrained",
                        "state": "development",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")

    snapshot = build_snapshot(
        tmp_path,
        freeze_id="test-freeze",
        frozen_on="2026-07-25",
        source_commit="test-source",
    )
    assert snapshot["summary"] == {
        "registered_models_with_assets": 1,
        "registered_models_without_assets": 1,
        "tracked_files": 2,
        "git_lfs_files": 1,
        "git_files": 1,
        "total_bytes": 21,
    }
    assert snapshot["registered_models_without_assets"] == ["future@untrained"]

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    assert verify_snapshot(tmp_path, snapshot_path) == []

    (asset_dir / "best.pt").write_bytes(b"changed")
    assert verify_snapshot(tmp_path, snapshot_path)
