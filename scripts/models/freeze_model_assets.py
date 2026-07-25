"""Create or verify a deterministic snapshot of every tracked model asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "models/frozen/all-model-assets-20260725.json"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_asset_paths(root: Path) -> list[str]:
    output = _git(root, "ls-files", "-z", "--", "models/assets")
    return sorted(path for path in output.split("\0") if path)


def _index_oid(root: Path, relative_path: str) -> str:
    output = _git(root, "ls-files", "-s", "--", relative_path).strip()
    if not output:
        raise ValueError(f"asset is not tracked: {relative_path}")
    return output.split(maxsplit=3)[1]


def _storage(root: Path, relative_path: str) -> str:
    output = _git(root, "check-attr", "filter", "--", relative_path).strip()
    return "git-lfs" if output.endswith(": lfs") else "git"


def _registered_models(root: Path, tracked_paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = json.loads((root / "models/manifest.yaml").read_text(encoding="utf-8"))
    registered: list[dict[str, Any]] = []
    without_assets: list[str] = []
    tracked = set(tracked_paths)

    for model in manifest["models"]:
        identity = f"{model['model_id']}@{model['version']}"
        asset = model.get("asset")
        if not asset or not asset.get("path"):
            without_assets.append(identity)
            continue
        asset_path = asset["path"].replace("\\", "/").rstrip("/")
        covered = asset_path in tracked or any(
            path.startswith(f"{asset_path}/") for path in tracked_paths
        )
        if not covered:
            raise ValueError(f"registered asset is not covered by tracked files: {asset_path}")
        registered.append(
            {
                "model_id": model["model_id"],
                "version": model["version"],
                "state": model["state"],
                "asset_path": asset_path,
            }
        )
    return registered, without_assets


def build_snapshot(
    root: Path,
    *,
    freeze_id: str,
    frozen_on: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    tracked_paths = _tracked_asset_paths(root)
    if not tracked_paths:
        raise ValueError("no tracked model assets found")

    files: list[dict[str, Any]] = []
    for relative_path in tracked_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "storage": _storage(root, relative_path),
                "git_index_blob_oid": _index_oid(root, relative_path),
            }
        )

    registered, without_assets = _registered_models(root, tracked_paths)
    return {
        "schema_version": "1.0",
        "freeze_id": freeze_id,
        "frozen_on": frozen_on,
        "scope": "all Git-tracked files under models/assets",
        "source_asset_commit": source_commit or _git(root, "rev-parse", "HEAD").strip(),
        "semantics": {
            "runtime_selection_changed": False,
            "admission_state_changed": False,
            "release_promotion": False,
        },
        "registered_models": registered,
        "registered_models_without_assets": without_assets,
        "summary": {
            "registered_models_with_assets": len(registered),
            "registered_models_without_assets": len(without_assets),
            "tracked_files": len(files),
            "git_lfs_files": sum(item["storage"] == "git-lfs" for item in files),
            "git_files": sum(item["storage"] == "git" for item in files),
            "total_bytes": sum(item["bytes"] for item in files),
        },
        "files": files,
    }


def verify_snapshot(root: Path, snapshot_path: Path) -> list[str]:
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    actual = build_snapshot(
        root,
        freeze_id=expected["freeze_id"],
        frozen_on=expected["frozen_on"],
        source_commit=expected["source_asset_commit"],
    )
    errors: list[str] = []
    for key in (
        "schema_version",
        "freeze_id",
        "frozen_on",
        "scope",
        "source_asset_commit",
        "semantics",
        "registered_models",
        "registered_models_without_assets",
        "summary",
        "files",
    ):
        if expected.get(key) != actual.get(key):
            errors.append(f"snapshot mismatch: {key}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze-id", default="all-tracked-model-assets-20260725")
    parser.add_argument("--frozen-on", default="2026-07-25")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if args.write == args.verify:
        raise SystemExit("choose exactly one of --write or --verify")
    if args.write:
        snapshot = build_snapshot(
            root,
            freeze_id=args.freeze_id,
            frozen_on=args.frozen_on,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(snapshot["summary"], ensure_ascii=False))
        return 0

    errors = verify_snapshot(root, output)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"verified {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
