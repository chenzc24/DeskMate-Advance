from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from poker_dealer.runtime import DiagnosticRun, check_diagnostic_bundle


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runtime/run_hand.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_hand_diagnostics_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_hash_chain(path: Path) -> None:
    previous = "0" * 64
    for sequence, record in enumerate(_records(path)):
        assert record["sequence"] == sequence
        assert record["previous_hash"] == previous
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert record["record_hash"] == hashlib.sha256(encoded).hexdigest()
        previous = str(record["record_hash"])


def test_diagnostics_bundle_redacts_and_summarizes_artifacts(tmp_path, capsys) -> None:
    bundle = tmp_path / "field-run"
    diagnostics = DiagnosticRun(
        bundle,
        run_id="field-run",
        profile_id="robot_camera",
        mode="live",
        invocation={"stream_url": "https://example.invalid/video?token=secret"},
    )
    hand_log = diagnostics.hand_log_path("hand-001")
    hand_log.write_text("evidence\n", encoding="utf-8")
    diagnostics.register_artifact("test_artifact", hand_log)
    with diagnostics.capture_stdio():
        print("camera=https://example.invalid/video?token=secret")
        diagnostics.emit(
            "camera_connected",
            {"stream_url": "https://example.invalid/video?token=secret"},
        )
        diagnostics.metric(
            "runtime_step_duration", 12.5, {"hand_id": "hand-001"}
        )
        diagnostics.emit("runtime_result", {"completed": True})
    diagnostics.finish(0)

    assert "camera=https://example.invalid" in capsys.readouterr().out
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            bundle / "manifest.json",
            bundle / "runtime.jsonl",
            bundle / "stdout.log",
        )
    )
    assert "example.invalid" not in persisted
    assert "secret" not in persisted
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["first_failure"] is None
    assert summary["result"] == {"completed": True}
    assert summary["metrics"]["runtime_step_duration"]["p95"] == 12.5
    assert summary["artifacts"][0]["exists"] is True
    assert summary["artifacts"][0]["sha256"] == hashlib.sha256(
        hand_log.read_bytes()
    ).hexdigest()
    _verify_hash_chain(bundle / "runtime.jsonl")
    _verify_hash_chain(bundle / "metrics.jsonl")
    checked = check_diagnostic_bundle(bundle)
    assert checked.passed
    assert checked.runtime_records > 0


def test_diagnostics_bounds_and_first_failure_context(tmp_path) -> None:
    diagnostics = DiagnosticRun(
        tmp_path / "bounded",
        run_id="bounded",
        profile_id="laptop",
        mode="replay",
        max_records=3,
        max_bytes_per_stream=4_096,
    )
    diagnostics.emit("before_failure", {"state_version": 4})
    diagnostics.emit(
        "camera_failed",
        {"reason": "disconnected", "state_version": 4},
        level="error",
    )
    diagnostics.emit("after_limit", {"ignored": True})
    diagnostics.finish(4)

    summary = json.loads(
        (tmp_path / "bounded" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["first_failure"]["kind"] == "camera_failed"
    assert any(
        event["kind"] == "before_failure" for event in summary["causal_context"]
    )
    assert summary["truncated"]["runtime_jsonl"] is True


def test_formal_replay_cli_places_audited_logs_in_diagnostics_bundle(
    tmp_path, monkeypatch, capsys
) -> None:
    module = _load_script()
    bundle = tmp_path / "formal-replay"

    def create_diagnostics(args, mode):
        return DiagnosticRun(
            bundle,
            run_id="formal-replay",
            profile_id="laptop",
            mode=mode,
            invocation=vars(args),
        )

    monkeypatch.setattr(module, "_create_diagnostics", create_diagnostics)
    assert module.main(
        [
            "--profile",
            "laptop",
            "--mode",
            "replay",
            "--session-id",
            "diagnostic-test",
            "--hand-id",
            "hand-001",
            "--diagnostics",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["completed"] is True
    assert output["diagnostics_path"] == str(bundle.resolve())
    assert (bundle / "session.jsonl").is_file()
    assert (bundle / "hands" / "hand-001.jsonl").is_file()
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["result"]["session_log_check_passed"] is True
    assert summary["metrics"]["runtime_step_duration"]["count"] == 111
    returned = tmp_path / "returned-bundle"
    shutil.copytree(bundle, returned)
    assert check_diagnostic_bundle(returned).passed


def test_formal_cli_caught_error_writes_traceback_and_failure_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    module = _load_script()
    bundle = tmp_path / "failed-run"

    monkeypatch.setattr(
        module,
        "_create_diagnostics",
        lambda args, mode: DiagnosticRun(
            bundle,
            run_id="failed-run",
            profile_id="laptop",
            mode=mode,
        ),
    )
    assert module.main(
        ["--profile", "laptop", "--mode", "replay", "--max-hands", "0", "--diagnostics"]
    ) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["type"] == "runtime_error"
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["first_failure"]["kind"] == "uncaught_exception"
    runtime_text = (bundle / "runtime.jsonl").read_text(encoding="utf-8")
    assert "Traceback" in runtime_text
    assert check_diagnostic_bundle(bundle).passed


def test_diagnostics_checker_rejects_tampered_chain(tmp_path) -> None:
    bundle = tmp_path / "tampered"
    diagnostics = DiagnosticRun(
        bundle,
        run_id="tampered",
        profile_id="laptop",
        mode="preflight",
    )
    diagnostics.finish(0)
    lines = (bundle / "runtime.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["kind"] = "changed"
    lines[0] = json.dumps(record)
    (bundle / "runtime.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checked = check_diagnostic_bundle(bundle)
    assert not checked.passed
    assert any("runtime_record_hash_mismatch" in issue for issue in checked.issues)
