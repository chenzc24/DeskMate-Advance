"""Validate or run a privacy-safe Part A scalar JSONL replay."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from deskmate_advance.temporal.ergonomics.candidates import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateComponentContext,
    CandidateContext,
    CandidateJsonlFile,
    PartACandidateEmitter,
    candidate_context_sha256,
)
from deskmate_advance.temporal.ergonomics.evaluation import (
    ContinuousRuleEvaluator,
    EvaluationDataStatus,
)
from deskmate_advance.temporal.ergonomics.replay import (
    ReplayFile,
    ReplayLimits,
    ReplayValidationError,
    producer_bundle_sha256,
    sha256_file,
    validate_local_provenance,
)
from deskmate_advance.temporal.ergonomics.rules import (
    ErgonomicsEventConfig,
    ErgonomicsRuleEngine,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("replay", type=Path, help="scalar JSONL replay path")
    parser.add_argument(
        "--sha256",
        required=True,
        help="expected lowercase SHA-256 from the replay manifest",
    )
    parser.add_argument("--max-line-bytes", type=int, default=32 * 1024)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--max-file-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--event-config",
        type=Path,
        default=Path("configs/ergonomics/events.json"),
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("configs/ergonomics/perception.json"),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("models/manifest.yaml"),
    )
    parser.add_argument(
        "--skip-local-provenance",
        action="store_true",
        help="validate replay bytes/schema only; do not bind local configs/assets",
    )
    parser.add_argument(
        "--no-verify-assets",
        action="store_true",
        help="verify model manifest entries but do not hash local model bytes",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", description="fully validate without running temporal rules"
    )
    _add_common_arguments(validate_parser)
    run_parser = subparsers.add_parser(
        "run", description="validate, then replay through the Part A rule engine"
    )
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--summary",
        type=Path,
        help="optional compact JSON summary path; stdout is always emitted",
    )
    run_parser.add_argument(
        "--candidates",
        type=Path,
        help="optional deterministic Part A candidate JSONL output",
    )
    run_parser.add_argument(
        "--candidate-update-ms",
        type=int,
        default=1_000,
        help="bounded active/unknown candidate heartbeat in milliseconds",
    )
    run_parser.add_argument(
        "--data-status",
        choices=(
            EvaluationDataStatus.UNLABELED_SCREENING.value,
            EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST.value,
        ),
        default=None,
        help="optional assertion; must match the replay header",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing --summary/--candidates files",
    )
    return parser


def _under_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load_event_config_verified(
    path: Path,
    *,
    expected_sha256: str,
) -> ErgonomicsEventConfig:
    maximum_bytes = 1024 * 1024
    raw = bytearray()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            if len(raw) + len(chunk) > maximum_bytes:
                raise ReplayValidationError("event config exceeds 1 MiB safety limit")
            raw.extend(chunk)
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReplayValidationError("run event config does not match replay provenance")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ReplayValidationError(f"duplicate event-config key: {key}")
            output[key] = value
        return output

    try:
        data = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReplayValidationError(
                    f"non-finite event-config number: {value}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReplayValidationError("event config is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise ReplayValidationError("event config root must be an object")
    return ErgonomicsEventConfig.from_mapping(data)


def _open(args: argparse.Namespace) -> ReplayFile:
    limits = ReplayLimits(
        max_line_bytes=args.max_line_bytes,
        max_records=args.max_records,
        max_file_bytes=args.max_file_bytes,
    )
    replay = ReplayFile(
        args.replay,
        expected_sha256=args.sha256,
        limits=limits,
    )
    try:
        if not args.skip_local_provenance:
            project_root = args.project_root.resolve()
            validate_local_provenance(
                replay.header,
                project_root=project_root,
                event_config_path=_under_root(project_root, args.event_config),
                perception_config_path=_under_root(
                    project_root, args.perception_config
                ),
                model_manifest_path=_under_root(project_root, args.model_manifest),
                verify_assets=not args.no_verify_assets,
            )
    except Exception:
        replay.close()
        raise
    return replay


def _validation_payload(
    replay: ReplayFile,
    *,
    provenance_verified: bool,
    assets_verified: bool,
) -> dict[str, Any]:
    summary = replay.validate()
    return {
        "mode": "validate",
        "schema": "deskmate.ergonomics.scalar-replay/1.0",
        "data_status": replay.header.data_status,
        **asdict(summary),
        "privacy": {
            "structural_scalar_schema_verified": True,
            "declared_contains_images": False,
            "declared_contains_landmarks": False,
            "declared_contains_audio_samples": False,
            "direct_identifier_absence": "declared_not_verified",
        },
        "provenance": {
            "event_config_sha256": replay.header.provenance.event_config_sha256,
            "perception_config_sha256": (
                replay.header.provenance.perception_config_sha256
            ),
            "model_manifest_sha256": (
                replay.header.provenance.model_manifest_sha256
            ),
            "feature_bundle_sha256": (
                replay.header.provenance.feature_bundle_sha256
            ),
        },
        "verification": {
            "provenance_verified": provenance_verified,
            "assets_verified": assets_verified,
        },
    }


def _profile_sha256(replay: ReplayFile) -> str:
    payload = json.dumps(
        asdict(replay.header.calibration_profile),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _run_context(project_root: Path) -> dict[str, Any]:
    """Record compact checkout/environment identity without leaking paths."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "git command failed"
            raise ReplayValidationError(f"cannot record Git provenance: {detail}")
        return completed.stdout

    commit = git("rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ReplayValidationError("Git commit identity is not a hexadecimal hash")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).replace("\r\n", "\n")
    distributions: dict[str, str | None] = {}
    for distribution in (
        "deskmate-advance",
        "mediapipe",
        "numpy",
        "opencv-contrib-python",
        "sounddevice",
    ):
        try:
            distributions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "git": {
            "commit": commit,
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "project_definition_sha256": sha256_file(
                project_root / "pyproject.toml"
            ),
            "distributions": distributions,
        },
    }


def _candidate_context(
    replay: ReplayFile,
    config: ErgonomicsEventConfig,
    *,
    producer_sha256: str,
    provenance_verified: bool,
    assets_verified: bool,
) -> CandidateContext:
    provenance = replay.header.provenance
    components = (
        CandidateComponentContext(
            role="pose",
            model_id=provenance.pose_model.model_id,
            model_version=provenance.pose_model.model_version,
            asset_sha256=provenance.pose_model.asset_sha256,
            config_sha256=provenance.perception_config_sha256,
        ),
        CandidateComponentContext(
            role="face",
            model_id=provenance.face_model.model_id,
            model_version=provenance.face_model.model_version,
            asset_sha256=provenance.face_model.asset_sha256,
            config_sha256=provenance.perception_config_sha256,
        ),
        CandidateComponentContext(
            role="luminance",
            model_id="rgb_luminance_statistics",
            model_version="1.0",
            config_sha256=provenance.perception_config_sha256,
        ),
        CandidateComponentContext(
            role="audio",
            model_id="rms_dbfs_level",
            model_version="1.0",
            config_sha256=provenance.perception_config_sha256,
        ),
        CandidateComponentContext(
            role="rule",
            model_id="part_a_ergonomics_rules",
            model_version=config.schema_version,
            config_sha256=provenance.event_config_sha256,
        ),
    )
    return CandidateContext(
        producer_id="deskmate_advance.part_a_ergonomics",
        producer_version="a3-replay/1.0",
        rule_config_schema_version=config.schema_version,
        rule_config_status=config.status,
        rule_config_sha256=provenance.event_config_sha256,
        calibration_profile_sha256=_profile_sha256(replay),
        trace_id=f"replay:{replay.header.replay_id}",
        data_status=replay.header.data_status,
        producer_bundle_sha256=producer_sha256,
        feature_bundle_sha256=provenance.feature_bundle_sha256,
        model_manifest_sha256=provenance.model_manifest_sha256,
        provenance_verified=provenance_verified,
        assets_verified=assets_verified,
        input_artifact_sha256=replay.artifact_sha256,
        components=components,
    )


class _OutputTransaction:
    """Stage all outputs under artifacts/ and commit or roll back as a group."""

    def __init__(
        self,
        *,
        project_root: Path,
        requested: dict[str, Path | None],
        protected_paths: Sequence[Path],
        overwrite: bool,
    ) -> None:
        self.overwrite = overwrite
        self._targets: dict[str, Path] = {}
        self._temporaries: dict[str, Path] = {}
        self._handles: dict[str, Any] = {}
        self._committed: list[str] = []
        self._backups: dict[str, Path] = {}
        resolved_project_root = project_root.resolve()
        declared_artifacts_root = resolved_project_root / "artifacts"
        if declared_artifacts_root.exists() and (
            declared_artifacts_root.is_symlink()
            or (
                hasattr(declared_artifacts_root, "is_junction")
                and declared_artifacts_root.is_junction()
            )
        ):
            raise ReplayValidationError(
                "artifacts output root must not be a symlink or junction"
            )
        declared_artifacts_root.mkdir(parents=True, exist_ok=True)
        artifacts_root = declared_artifacts_root.resolve()
        if artifacts_root != declared_artifacts_root:
            raise ReplayValidationError("artifacts output root changed while resolving")
        protected = tuple(path.resolve() for path in protected_paths)
        for name, requested_path in requested.items():
            if requested_path is None:
                continue
            unresolved = (
                requested_path
                if requested_path.is_absolute()
                else project_root / requested_path
            )
            target = unresolved.resolve()
            if target == artifacts_root or artifacts_root not in target.parents:
                raise ReplayValidationError(
                    f"{name} output must stay under {artifacts_root}"
                )
            if target in self._targets.values():
                raise ReplayValidationError("output paths must be distinct")
            if target.exists() and target.is_dir():
                raise ReplayValidationError(f"output target is a directory: {target}")
            if target.exists() and target.stat().st_nlink > 1:
                raise ReplayValidationError(
                    f"output target has multiple hard links and is protected: {target}"
                )
            for protected_path in protected:
                if target == protected_path or (
                    target.exists()
                    and protected_path.exists()
                    and os.path.samefile(target, protected_path)
                ):
                    raise ReplayValidationError(
                        f"output aliases protected input: {target}"
                    )
            if target.exists() and not overwrite:
                raise ReplayValidationError(f"output already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._targets[name] = target
        try:
            for name, target in self._targets.items():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".part-a-tmp",
                    dir=target.parent,
                )
                temporary = Path(temporary_name)
                self._temporaries[name] = temporary
                self._handles[name] = os.fdopen(
                    descriptor,
                    "w+",
                    encoding="utf-8",
                    newline="\n",
                )
        except Exception:
            self.abort()
            raise

    def handle(self, name: str) -> Any | None:
        return self._handles.get(name)

    def staged_path(self, name: str) -> Path | None:
        return self._temporaries.get(name)

    def sha256(self, name: str) -> str | None:
        handle = self._handles.get(name)
        if handle is None:
            return None
        handle.flush()
        os.fsync(handle.fileno())
        digest = hashlib.sha256()
        with self._temporaries[name].open("rb") as raw:
            for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def commit(self) -> int:
        try:
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            for name, target in self._targets.items():
                if target.exists():
                    if not self.overwrite:
                        raise ReplayValidationError(f"output already exists: {target}")
                    descriptor, backup_name = tempfile.mkstemp(
                        prefix=f".{target.name}.",
                        suffix=".part-a-backup",
                        dir=target.parent,
                    )
                    os.close(descriptor)
                    backup = Path(backup_name)
                    os.replace(target, backup)
                    self._backups[name] = backup
            for name, target in self._targets.items():
                temporary = self._temporaries[name]
                if self.overwrite:
                    os.replace(temporary, target)
                    self._committed.append(name)
                else:
                    os.link(temporary, target)
                    self._committed.append(name)
                    temporary.unlink()
        except Exception as exc:
            rollback_failures = self._rollback_commit()
            if rollback_failures:
                raise ReplayValidationError(
                    "output commit failed and rollback encountered "
                    f"{rollback_failures} additional I/O error(s)"
                ) from exc
            raise
        cleanup_failures = 0
        for backup in self._backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                cleanup_failures += 1
        self._backups.clear()
        return cleanup_failures

    def abort(self) -> int:
        failures = 0
        for handle in self._handles.values():
            if not handle.closed:
                try:
                    handle.close()
                except OSError:
                    failures += 1
        for temporary in self._temporaries.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                failures += 1
        return failures

    def _rollback_commit(self) -> int:
        failures = 0
        for name in reversed(self._committed):
            try:
                self._targets[name].unlink(missing_ok=True)
            except OSError:
                failures += 1
        for name, backup in self._backups.items():
            if backup.exists():
                try:
                    os.replace(backup, self._targets[name])
                except OSError:
                    failures += 1
        failures += self.abort()
        self._committed.clear()
        self._backups.clear()
        return failures


def _run_payload(
    replay: ReplayFile,
    config: ErgonomicsEventConfig,
    *,
    candidate_handle: Any | None,
    candidate_update_ms: int,
    data_status: EvaluationDataStatus,
    producer_sha256: str,
    provenance_verified: bool,
    assets_verified: bool,
) -> dict[str, Any]:
    # Complete contract validation before temporal state can be advanced.
    validated = replay.validate()
    engine = ErgonomicsRuleEngine(
        config,
        profile=replay.header.calibration_profile,
    )
    emitter = PartACandidateEmitter(update_interval_ms=candidate_update_ms)
    candidate_context = _candidate_context(
        replay,
        config,
        producer_sha256=producer_sha256,
        provenance_verified=provenance_verified,
        assets_verified=assets_verified,
    )
    evaluator = ContinuousRuleEvaluator(
        maximum_evidence_gap_ms=config.maximum_evidence_gap_ms,
        data_status=data_status,
    )
    candidate_counts: Counter[str] = Counter()
    candidate_event_counts: dict[str, Counter[str]] = {
        event_name: Counter() for event_name in ErgonomicsRuleEngine.EVENT_NAMES
    }
    state_counts: dict[str, Counter[str]] = {
        event_name: Counter() for event_name in ErgonomicsRuleEngine.EVENT_NAMES
    }
    condition_counts: dict[str, Counter[str]] = {
        event_name: Counter() for event_name in ErgonomicsRuleEngine.EVENT_NAMES
    }
    last = None
    for sample in replay.iter_samples():
        last = engine.update(
            sample.snapshot,
            audio_level=sample.audio_level,
        )
        evaluator.add(last)
        candidates = emitter.emit(
            last,
            sequence_id=sample.snapshot.frame.sequence_id,
            context=candidate_context,
        )
        for candidate in candidates:
            candidate_counts[candidate.transition.value] += 1
            candidate_event_counts[candidate.event_name][
                candidate.transition.value
            ] += 1
            if candidate_handle is not None:
                candidate_handle.write(candidate.to_json() + "\n")
        for evaluation in last.evaluations:
            state_counts[evaluation.event_name][evaluation.semantic_state.value] += 1
            condition_counts[evaluation.event_name][evaluation.condition.value] += 1
    evaluation_summary = evaluator.finish()
    if last is None:
        raise ReplayValidationError("validated replay unexpectedly produced no samples")
    return {
        "mode": "run",
        "data_status": replay.header.data_status,
        **asdict(validated),
        "config_schema_version": config.schema_version,
        "config_status": config.status,
        "provenance": {
            "event_config_sha256": replay.header.provenance.event_config_sha256,
            "perception_config_sha256": (
                replay.header.provenance.perception_config_sha256
            ),
            "model_manifest_sha256": (
                replay.header.provenance.model_manifest_sha256
            ),
            "feature_bundle_sha256": (
                replay.header.provenance.feature_bundle_sha256
            ),
            "calibration_profile_sha256": _profile_sha256(replay),
            "producer_bundle_sha256": producer_sha256,
        },
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_context_sha256": candidate_context_sha256(candidate_context),
        "candidate_records": sum(candidate_counts.values()),
        "candidate_artifact_sha256": None,
        "candidate_transition_counts": dict(sorted(candidate_counts.items())),
        "candidate_events": {
            event_name: dict(sorted(candidate_event_counts[event_name].items()))
            for event_name in ErgonomicsRuleEngine.EVENT_NAMES
        },
        "verification": {
            "provenance_verified": provenance_verified,
            "assets_verified": assets_verified,
        },
        "privacy": {
            "structural_scalar_schema_verified": True,
            "declared_contains_images": False,
            "declared_contains_landmarks": False,
            "declared_contains_audio_samples": False,
            "direct_identifier_absence": "declared_not_verified",
        },
        "continuous_evaluation": asdict(evaluation_summary),
        "last_snapshot": {
            "source_id": last.source_id,
            "captured_at_ns": last.captured_at_ns,
        },
        "events": {
            event_name: {
                "semantic_state_counts": dict(sorted(state_counts[event_name].items())),
                "condition_counts": dict(
                    sorted(condition_counts[event_name].items())
                ),
                "final_semantic_state": last.evaluation(
                    event_name
                ).semantic_state.value,
                "final_phase": last.evaluation(event_name).phase.value,
            }
            for event_name in ErgonomicsRuleEngine.EVENT_NAMES
        },
    }


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    replay: ReplayFile | None = None
    try:
        replay = _open(args)
        provenance_verified = not args.skip_local_provenance
        assets_verified = provenance_verified and not args.no_verify_assets
        if args.command == "validate":
            payload = _validation_payload(
                replay,
                provenance_verified=provenance_verified,
                assets_verified=assets_verified,
            )
            payload["run_context"] = _run_context(args.project_root.resolve())
        elif args.command == "run":
            project_root = args.project_root.resolve()
            event_config_path = _under_root(project_root, args.event_config)
            perception_config_path = _under_root(
                project_root, args.perception_config
            )
            model_manifest_path = _under_root(project_root, args.model_manifest)
            if (
                args.data_status is not None
                and args.data_status != replay.header.data_status
            ):
                raise ReplayValidationError(
                    "--data-status does not match replay header"
                )
            if replay.header.data_status == EvaluationDataStatus.LABELED_EVIDENCE.value:
                raise ReplayValidationError(
                    "labeled replay requires the annotation-aware evaluation API"
                )
            transaction = _OutputTransaction(
                project_root=project_root,
                requested={
                    "candidates": args.candidates,
                    "summary": args.summary,
                },
                protected_paths=(
                    replay.path,
                    event_config_path,
                    perception_config_path,
                    model_manifest_path,
                ),
                overwrite=args.overwrite,
            )
            try:
                config = _load_event_config_verified(
                    event_config_path,
                    expected_sha256=(
                        replay.header.provenance.event_config_sha256
                    ),
                )
                payload = _run_payload(
                    replay,
                    config,
                    candidate_handle=transaction.handle("candidates"),
                    candidate_update_ms=args.candidate_update_ms,
                    data_status=EvaluationDataStatus(replay.header.data_status),
                    producer_sha256=producer_bundle_sha256(project_root),
                    provenance_verified=provenance_verified,
                    assets_verified=assets_verified,
                )
                payload["run_context"] = _run_context(project_root)
                payload["candidate_artifact_sha256"] = transaction.sha256(
                    "candidates"
                )
                candidate_path = transaction.staged_path("candidates")
                candidate_sha256 = payload["candidate_artifact_sha256"]
                if candidate_path is not None and candidate_sha256 is not None:
                    with CandidateJsonlFile(
                        candidate_path,
                        expected_sha256=candidate_sha256,
                    ) as candidate_artifact:
                        candidate_validation = candidate_artifact.validate()
                    if candidate_validation.records != payload["candidate_records"]:
                        raise ReplayValidationError(
                            "candidate producer/consumer record counts differ"
                        )
                    if candidate_validation.records > 0 and (
                        candidate_validation.source_id
                        != replay.header.camera.source_id
                        or candidate_validation.context_sha256
                        != payload["candidate_context_sha256"]
                        or candidate_validation.data_status
                        != replay.header.data_status
                        or candidate_validation.input_artifact_sha256
                        != replay.artifact_sha256
                    ):
                        raise ReplayValidationError(
                            "candidate producer/consumer run contexts differ"
                        )
                    payload["candidate_consumer_validation"] = asdict(
                        candidate_validation
                    )
                else:
                    payload["candidate_consumer_validation"] = None
                summary_handle = transaction.handle("summary")
                if summary_handle is not None:
                    summary_handle.write(_json_text(payload))
                cleanup_failures = transaction.commit()
                if cleanup_failures:
                    print(
                        "replay warning: committed outputs, but could not remove "
                        f"{cleanup_failures} rollback backup file(s)",
                        file=sys.stderr,
                    )
            except Exception:
                abort_failures = transaction.abort()
                if abort_failures:
                    print(
                        "replay warning: cleanup after an aborted output had "
                        f"{abort_failures} additional I/O error(s)",
                        file=sys.stderr,
                    )
                raise
        else:  # argparse requires one of the known commands.
            raise RuntimeError(f"unsupported command: {args.command}")
    except (OSError, UnicodeError, ReplayValidationError, ValueError) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2
    finally:
        if replay is not None:
            replay.close()
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
