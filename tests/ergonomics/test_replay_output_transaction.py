from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
from types import ModuleType
from uuid import uuid4

import pytest

from deskmate_advance.temporal.ergonomics.replay import ReplayValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _script_module() -> ModuleType:
    path = PROJECT_ROOT / "scripts/ergonomics/replay_part_a.py"
    spec = importlib.util.spec_from_file_location("a3_replay_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load replay CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_dir() -> Path:
    path = PROJECT_ROOT / "artifacts" / "pytest-a3-transaction" / uuid4().hex
    path.mkdir(parents=True)
    return path


def test_group_commit_restores_both_previous_outputs_after_second_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    output_dir = _artifact_dir()
    candidates = output_dir / "candidates.jsonl"
    summary = output_dir / "summary.json"
    candidates.write_text("old-candidates\n", encoding="utf-8")
    summary.write_text("old-summary\n", encoding="utf-8")
    transaction = module._OutputTransaction(
        project_root=PROJECT_ROOT,
        requested={"candidates": candidates, "summary": summary},
        protected_paths=(),
        overwrite=True,
    )
    transaction.handle("candidates").write("new-candidates\n")
    transaction.handle("summary").write("new-summary\n")

    real_replace = module.os.replace
    failed = False

    def fail_summary_install(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        nonlocal failed
        if (
            not failed
            and Path(target) == summary
            and str(source).endswith(".part-a-tmp")
        ):
            failed = True
            raise OSError("injected second-output failure")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_summary_install)
    with pytest.raises(OSError, match="injected second-output failure"):
        transaction.commit()

    assert candidates.read_text(encoding="utf-8") == "old-candidates\n"
    assert summary.read_text(encoding="utf-8") == "old-summary\n"
    assert sorted(item.name for item in output_dir.iterdir()) == [
        "candidates.jsonl",
        "summary.json",
    ]
    shutil.rmtree(output_dir)


def test_output_target_with_an_existing_hard_link_is_rejected() -> None:
    module = _script_module()
    output_dir = _artifact_dir()
    original = output_dir / "protected.txt"
    original.write_text("protected", encoding="utf-8")
    linked_output = output_dir / "linked.json"
    try:
        os.link(original, linked_output)
    except OSError as exc:
        shutil.rmtree(output_dir)
        pytest.skip(f"filesystem does not support hard links: {exc}")

    with pytest.raises(ReplayValidationError, match="multiple hard links"):
        module._OutputTransaction(
            project_root=PROJECT_ROOT,
            requested={"summary": linked_output},
            protected_paths=(),
            overwrite=True,
        )

    assert original.read_text(encoding="utf-8") == "protected"
    assert linked_output.read_text(encoding="utf-8") == "protected"
    linked_output.unlink()
    original.unlink()
    shutil.rmtree(output_dir)


def test_no_overwrite_commit_rolls_back_target_if_temp_unlink_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    output_dir = _artifact_dir()
    target = output_dir / "candidates.jsonl"
    transaction = module._OutputTransaction(
        project_root=PROJECT_ROOT,
        requested={"candidates": target},
        protected_paths=(),
        overwrite=False,
    )
    transaction.handle("candidates").write("candidate\n")
    temporary = transaction.staged_path("candidates")
    assert temporary is not None
    real_unlink = Path.unlink
    failed = False

    def fail_first_temp_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal failed
        if path == temporary and not failed:
            failed = True
            raise OSError("injected temp unlink failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)
    with pytest.raises(OSError, match="injected temp unlink failure"):
        transaction.commit()

    assert not target.exists()
    assert not temporary.exists()
    shutil.rmtree(output_dir)


def test_rollback_continues_restoring_backups_after_one_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    output_dir = _artifact_dir()
    candidates = output_dir / "candidates.jsonl"
    summary = output_dir / "summary.json"
    candidates.write_text("old-candidates\n", encoding="utf-8")
    summary.write_text("old-summary\n", encoding="utf-8")
    transaction = module._OutputTransaction(
        project_root=PROJECT_ROOT,
        requested={"candidates": candidates, "summary": summary},
        protected_paths=(),
        overwrite=True,
    )
    transaction.handle("candidates").write("new-candidates\n")
    transaction.handle("summary").write("new-summary\n")

    real_replace = module.os.replace
    install_failed = False

    def fail_summary_install(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        nonlocal install_failed
        if (
            not install_failed
            and Path(target) == summary
            and str(source).endswith(".part-a-tmp")
        ):
            install_failed = True
            raise OSError("injected summary install failure")
        real_replace(source, target)

    real_unlink = Path.unlink
    cleanup_failed = False

    def fail_first_candidate_cleanup(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal cleanup_failed
        if path == candidates and not cleanup_failed:
            cleanup_failed = True
            raise OSError("injected rollback cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(module.os, "replace", fail_summary_install)
    monkeypatch.setattr(Path, "unlink", fail_first_candidate_cleanup)
    with pytest.raises(ReplayValidationError, match="rollback encountered 1"):
        transaction.commit()

    assert candidates.read_text(encoding="utf-8") == "old-candidates\n"
    assert summary.read_text(encoding="utf-8") == "old-summary\n"
    assert sorted(item.name for item in output_dir.iterdir()) == [
        "candidates.jsonl",
        "summary.json",
    ]
    shutil.rmtree(output_dir)


def test_event_config_limit_is_checked_while_reading(tmp_path: Path) -> None:
    module = _script_module()
    oversized = tmp_path / "oversized-events.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))

    with pytest.raises(ReplayValidationError, match="exceeds 1 MiB"):
        module._load_event_config_verified(
            oversized,
            expected_sha256="0" * 64,
        )
