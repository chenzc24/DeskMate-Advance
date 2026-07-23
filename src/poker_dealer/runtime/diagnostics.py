"""Bounded, privacy-safe diagnostics bundles for formal Runtime runs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
import traceback
from typing import Iterator, Mapping, Protocol, TextIO


_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|secret|token|credential|signed_url|stream_url)", re.I
)
_URL = re.compile(r"https?://[^\s]+", re.I)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_MAX_STRING = 8_192


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: object, *, key: str | None = None) -> object:
    """Return JSON-safe diagnostics data without credentials or live URLs."""

    if key is not None and _SENSITIVE_KEY.search(key):
        if value in (None, ""):
            return value
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
        return f"<redacted:sha256:{digest}>"
    if isinstance(value, Mapping):
        return {str(item): _redact(child, key=str(item)) for item, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        text = _URL.sub(
            lambda match: (
                "<redacted-url:sha256:"
                + hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()[:12]
                + ">"
            ),
            value,
        )
        return text if len(text) <= _MAX_STRING else text[:_MAX_STRING] + "…<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class DiagnosticSink(Protocol):
    """Observation-only interface accepted by Runtime loops."""

    def emit(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        level: str = "info",
    ) -> None: ...

    def metric(
        self,
        name: str,
        value: float,
        context: Mapping[str, object] | None = None,
        *,
        unit: str = "ms",
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DiagnosticArtifact:
    kind: str
    path: str
    exists: bool
    bytes: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class DiagnosticBundleCheck:
    passed: bool
    run_id: str | None
    status: str | None
    exit_code: int | None
    runtime_records: int
    metric_records: int
    first_failure: Mapping[str, object] | None
    issues: tuple[str, ...]


class _HashChainedJsonl:
    def __init__(self, path: Path, *, max_records: int, max_bytes: int) -> None:
        if max_records <= 0 or max_bytes <= 0:
            raise ValueError("diagnostic JSONL bounds must be positive")
        self.path = path
        self.max_records = max_records
        self.max_bytes = max_bytes
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self.records = 0
        self.bytes_written = 0
        self.previous_hash = "0" * 64
        self.truncated = False
        self._lock = threading.Lock()

    @staticmethod
    def _hash(unsigned: Mapping[str, object]) -> str:
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(self, value: Mapping[str, object]) -> bool:
        with self._lock:
            if self.truncated or self.records >= self.max_records:
                self.truncated = True
                return False
            unsigned = {
                "sequence": self.records,
                **value,
                "previous_hash": self.previous_hash,
            }
            record_hash = self._hash(unsigned)
            encoded = (
                json.dumps(
                    {**unsigned, "record_hash": record_hash},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            byte_count = len(encoded.encode("utf-8"))
            if self.bytes_written + byte_count > self.max_bytes:
                self.truncated = True
                return False
            self._stream.write(encoded)
            self._stream.flush()
            self.records += 1
            self.bytes_written += byte_count
            self.previous_hash = record_hash
            return True

    def close(self) -> None:
        self._stream.close()


class _Tee(TextIO):
    def __init__(self, primary: TextIO, capture: TextIO) -> None:
        self.primary = primary
        self.capture = capture

    @property
    def encoding(self) -> str | None:
        return self.primary.encoding

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        written = self.primary.write(text)
        self.capture.write(text)
        self.capture.flush()
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.capture.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def fileno(self) -> int:
        return self.primary.fileno()


class _BoundedTextCapture(TextIO):
    def __init__(self, path: Path, *, max_bytes: int) -> None:
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.truncated = False
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._lock:
            if self.truncated:
                return len(text)
            captured = str(_redact(text))
            encoded = captured.encode("utf-8")
            remaining = self.max_bytes - self.bytes_written
            if len(encoded) <= remaining:
                self._stream.write(captured)
                self._stream.flush()
                self.bytes_written += len(encoded)
                return len(text)
            marker = "\n<diagnostics stdio truncated at configured byte limit>\n"
            marker_bytes = marker.encode("utf-8")
            prefix_bytes = encoded[: max(0, remaining - len(marker_bytes))]
            prefix = prefix_bytes.decode("utf-8", errors="ignore")
            final = prefix + (marker if remaining >= len(marker_bytes) else "")
            self._stream.write(final)
            self._stream.flush()
            self.bytes_written += len(final.encode("utf-8"))
            self.truncated = True
            return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class DiagnosticRun:
    """One exclusive run bundle that never stores frames, audio or embeddings."""

    schema_version = "1.0"

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        profile_id: str,
        mode: str,
        invocation: Mapping[str, object] | None = None,
        max_records: int = 100_000,
        max_bytes_per_stream: int = 32 * 1024 * 1024,
    ) -> None:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError(
                "diagnostics run_id must contain only letters, digits, dot, dash or underscore"
            )
        self.root = root.resolve()
        self.run_id = run_id
        self.profile_id = profile_id
        self.mode = mode
        self.started_monotonic_ns = time.monotonic_ns()
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(exist_ok=False)
        (self.root / "hands").mkdir()
        (self.root / "configs").mkdir()
        self._events = _HashChainedJsonl(
            self.root / "runtime.jsonl",
            max_records=max_records,
            max_bytes=max_bytes_per_stream,
        )
        self._metrics = _HashChainedJsonl(
            self.root / "metrics.jsonl",
            max_records=max_records,
            max_bytes=max_bytes_per_stream,
        )
        self._stdout = _BoundedTextCapture(
            self.root / "stdout.log", max_bytes=max_bytes_per_stream
        )
        self._stderr = _BoundedTextCapture(
            self.root / "stderr.log", max_bytes=max_bytes_per_stream
        )
        self._event_counts: Counter[str] = Counter()
        self._metric_samples: dict[str, list[float]] = defaultdict(list)
        self._metric_units: dict[str, str] = {}
        self._first_failure: dict[str, object] | None = None
        self._recent_events: deque[dict[str, object]] = deque(maxlen=12)
        self._first_failure_context: list[dict[str, object]] = []
        self._result: object = {}
        self._artifacts: dict[Path, str] = {}
        self._config_files: dict[str, dict[str, object]] = {}
        self._closed = False
        self._write_error: str | None = None
        self._manifest: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": run_id,
            "profile_id": profile_id,
            "mode": mode,
            "started_at_utc": self.started_at_utc,
            "invocation": _redact(invocation or {}),
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "process_id": os.getpid(),
            },
            "privacy": {
                "frames_saved": False,
                "audio_saved": False,
                "embeddings_persisted": False,
                "urls_redacted": True,
            },
            "bounds": {
                "max_records_per_jsonl": max_records,
                "max_bytes_per_jsonl": max_bytes_per_stream,
            },
        }
        self._write_json(self.root / "manifest.json", self._manifest, exclusive=True)
        self.emit(
            "run_started",
            {
                "run_id": run_id,
                "profile_id": profile_id,
                "mode": mode,
                "process_id": os.getpid(),
            },
        )
        self._snapshot_process("start")

    @property
    def hand_directory(self) -> Path:
        return self.root / "hands"

    @property
    def session_log_path(self) -> Path:
        return self.root / "session.jsonl"

    def hand_log_path(self, hand_id: str) -> Path:
        if not _SAFE_ID.fullmatch(hand_id):
            raise ValueError("diagnostic hand_id contains unsupported characters")
        return self.hand_directory / f"{hand_id}.jsonl"

    def add_config(self, label: str, path: Path) -> None:
        source = path.resolve()
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)
        snapshot = self.root / "configs" / f"{safe_label}.json"
        exists = source.is_file()
        record: dict[str, object] = {
            "label": label,
            "source_path": str(source),
            "source_exists": exists,
            "source_bytes": source.stat().st_size if exists else None,
            "source_sha256": _sha256_file(source) if exists else None,
            "snapshot_path": None,
            "snapshot_bytes": None,
            "snapshot_sha256": None,
        }
        if exists:
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
                snapshot_value = {
                    "diagnostics_snapshot": True,
                    "source_sha256": record["source_sha256"],
                    "content": _redact(value),
                }
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                snapshot_value = {
                    "diagnostics_snapshot": True,
                    "source_sha256": record["source_sha256"],
                    "content_omitted": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            self._write_json(snapshot, snapshot_value, exclusive=True)
            record["snapshot_path"] = str(snapshot.relative_to(self.root))
            record["snapshot_bytes"] = snapshot.stat().st_size
            record["snapshot_sha256"] = _sha256_file(snapshot)
        self._config_files[label] = record

    def register_artifact(self, kind: str, path: Path) -> None:
        self._artifacts[path.resolve()] = kind

    def emit(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        level: str = "info",
    ) -> None:
        if self._closed:
            return
        value = {
            "observed_at_ns": time.monotonic_ns(),
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "kind": kind,
            "payload": _redact(payload or {}),
        }
        try:
            written = self._events.append(value)
        except OSError as exc:
            self._write_error = f"{type(exc).__name__}: {exc}"
            return
        if written:
            self._event_counts[kind] += 1
            self._recent_events.append(dict(value))
        if kind == "runtime_result":
            self._result = value["payload"]
        if level in {"error", "critical"} and self._first_failure is None:
            self._first_failure = {
                "kind": kind,
                "level": level,
                "observed_at_ns": value["observed_at_ns"],
                "payload": value["payload"],
            }
            self._first_failure_context = list(self._recent_events)

    def metric(
        self,
        name: str,
        value: float,
        context: Mapping[str, object] | None = None,
        *,
        unit: str = "ms",
    ) -> None:
        if self._closed:
            return
        numeric = float(value)
        record = {
            "observed_at_ns": time.monotonic_ns(),
            "name": name,
            "value": numeric,
            "unit": unit,
            "context": _redact(context or {}),
        }
        try:
            written = self._metrics.append(record)
        except OSError as exc:
            self._write_error = f"{type(exc).__name__}: {exc}"
            return
        if written:
            samples = self._metric_samples[name]
            if len(samples) < 100_000:
                samples.append(numeric)
            self._metric_units[name] = unit

    def record_exception(
        self,
        exc: BaseException,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self.emit(
            "uncaught_exception",
            {
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "context": dict(context or {}),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
            level="error",
        )

    @contextmanager
    def operation(
        self, kind: str, payload: Mapping[str, object] | None = None
    ) -> Iterator[None]:
        started = time.monotonic_ns()
        self.emit(f"{kind}_started", payload)
        try:
            yield
        except BaseException as exc:
            elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
            self.metric(f"{kind}_duration", elapsed_ms, payload)
            self.emit(
                f"{kind}_failed",
                {
                    **dict(payload or {}),
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "elapsed_ms": elapsed_ms,
                },
                level="error",
            )
            raise
        elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
        self.metric(f"{kind}_duration", elapsed_ms, payload)
        self.emit(f"{kind}_completed", {**dict(payload or {}), "elapsed_ms": elapsed_ms})

    @contextmanager
    def capture_stdio(self) -> Iterator[None]:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, self._stdout)
        sys.stderr = _Tee(original_stderr, self._stderr)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def finish(
        self,
        exit_code: int,
        *,
        result: Mapping[str, object] | None = None,
    ) -> None:
        if self._closed:
            return
        status = "passed" if exit_code == 0 else "failed"
        if exit_code != 0 and self._first_failure is None:
            self.emit(
                "nonzero_exit",
                {"exit_code": exit_code, "result": dict(result or {})},
                level="error",
            )
        self._snapshot_process("finish")
        elapsed_ms = (time.monotonic_ns() - self.started_monotonic_ns) / 1_000_000
        self.metric("run_duration", elapsed_ms, unit="ms")
        self.emit(
            "run_finished",
            {"exit_code": exit_code, "status": status, "elapsed_ms": elapsed_ms},
            level="info" if exit_code == 0 else "error",
        )
        event_truncated = self._events.truncated
        metric_truncated = self._metrics.truncated
        stdout_truncated = self._stdout.truncated
        stderr_truncated = self._stderr.truncated
        self._events.close()
        self._metrics.close()
        self._stdout.flush()
        self._stderr.flush()
        self._stdout.close()
        self._stderr.close()
        artifacts = self._artifact_records()
        configs = self._config_records()
        summary = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "mode": self.mode,
            "status": status,
            "exit_code": exit_code,
            "started_at_utc": self.started_at_utc,
            "duration_ms": elapsed_ms,
            "event_counts": dict(sorted(self._event_counts.items())),
            "metrics": self._metric_summary(),
            "first_failure": self._first_failure,
            "causal_context": self._first_failure_context,
            "truncated": {
                "runtime_jsonl": event_truncated,
                "metrics_jsonl": metric_truncated,
                "stdout_log": stdout_truncated,
                "stderr_log": stderr_truncated,
            },
            "diagnostics_write_error": self._write_error,
            "result": _redact(result if result is not None else self._result),
            "artifacts": [asdict(artifact) for artifact in artifacts],
        }
        self._write_json(self.root / "summary.json", summary, exclusive=True)
        self._manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._manifest["status"] = status
        self._manifest["exit_code"] = exit_code
        self._manifest["config_files"] = configs
        self._manifest["artifacts"] = [asdict(artifact) for artifact in artifacts]
        self._write_json(self.root / "manifest.json", self._manifest, exclusive=False)
        self._closed = True

    def _artifact_records(self) -> list[DiagnosticArtifact]:
        records: list[DiagnosticArtifact] = []
        for path, kind in sorted(self._artifacts.items(), key=lambda item: str(item[0])):
            exists = path.is_file()
            try:
                recorded_path = str(path.relative_to(self.root))
            except ValueError:
                recorded_path = str(path)
            records.append(
                DiagnosticArtifact(
                    kind=kind,
                    path=recorded_path,
                    exists=exists,
                    bytes=path.stat().st_size if exists else None,
                    sha256=_sha256_file(path) if exists else None,
                )
            )
        return records

    def _config_records(self) -> list[dict[str, object]]:
        return [dict(record) for _, record in sorted(self._config_files.items())]

    def _metric_summary(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, samples in sorted(self._metric_samples.items()):
            ordered = sorted(samples)
            result[name] = {
                "count": len(ordered),
                "unit": self._metric_units[name],
                "min": ordered[0],
                "p50": self._percentile(ordered, 0.50),
                "p95": self._percentile(ordered, 0.95),
                "p99": self._percentile(ordered, 0.99),
                "max": ordered[-1],
            }
        return result

    def _snapshot_process(self, point: str) -> None:
        context = {"point": point, "process_id": os.getpid()}
        self.metric(
            "process_cpu_time",
            time.process_time() * 1_000,
            context,
            unit="ms",
        )
        self.metric(
            "active_thread_count",
            float(threading.active_count()),
            context,
            unit="threads",
        )
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process(os.getpid())
            self.metric(
                "resident_memory",
                float(process.memory_info().rss),
                context,
                unit="bytes",
            )
        except (ImportError, OSError, RuntimeError):
            pass

    @staticmethod
    def _percentile(ordered: list[float], fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
        return ordered[index]

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, object], *, exclusive: bool) -> None:
        mode = "x" if exclusive else "w"
        with path.open(mode, encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")


def check_diagnostic_bundle(root: Path) -> DiagnosticBundleCheck:
    """Independently verify diagnostics chains and referenced artifact hashes."""

    bundle = root.resolve()
    issues: list[str] = []
    required = {
        "manifest": bundle / "manifest.json",
        "summary": bundle / "summary.json",
        "runtime": bundle / "runtime.jsonl",
        "metrics": bundle / "metrics.jsonl",
        "stdout": bundle / "stdout.log",
        "stderr": bundle / "stderr.log",
    }
    for label, path in required.items():
        if not path.is_file():
            issues.append(f"missing_{label}:{path.name}")
    if issues:
        return DiagnosticBundleCheck(
            False, None, None, None, 0, 0, None, tuple(issues)
        )
    try:
        manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return DiagnosticBundleCheck(
            False,
            None,
            None,
            None,
            0,
            0,
            None,
            (f"diagnostic_json_unreadable:{type(exc).__name__}:{exc}",),
        )
    if manifest.get("schema_version") != DiagnosticRun.schema_version:
        issues.append("manifest_schema_version_unsupported")
    if summary.get("schema_version") != DiagnosticRun.schema_version:
        issues.append("summary_schema_version_unsupported")
    if manifest.get("run_id") != summary.get("run_id"):
        issues.append("run_id_mismatch")
    runtime_records = _check_jsonl_chain(required["runtime"], "runtime", issues)
    metric_records = _check_jsonl_chain(required["metrics"], "metrics", issues)
    _check_references(
        bundle,
        manifest.get("config_files"),
        "config",
        issues,
        path_key="snapshot_path",
        hash_key="snapshot_sha256",
        allow_missing=True,
    )
    _check_references(bundle, manifest.get("artifacts"), "artifact", issues)
    _check_authoritative_logs(bundle, manifest.get("artifacts"), issues)
    for path in required.values():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(f"diagnostic_file_unreadable:{path.name}:{exc}")
            continue
        if _URL.search(text):
            issues.append(f"unredacted_url:{path.name}")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not False
        for key in ("frames_saved", "audio_saved", "embeddings_persisted")
    ):
        issues.append("privacy_manifest_invalid")
    raw_exit = summary.get("exit_code")
    exit_code = raw_exit if isinstance(raw_exit, int) else None
    if exit_code is None:
        issues.append("summary_exit_code_missing")
    status = summary.get("status") if isinstance(summary.get("status"), str) else None
    first_failure = summary.get("first_failure")
    return DiagnosticBundleCheck(
        passed=not issues,
        run_id=str(summary.get("run_id")) if summary.get("run_id") else None,
        status=status,
        exit_code=exit_code,
        runtime_records=runtime_records,
        metric_records=metric_records,
        first_failure=first_failure if isinstance(first_failure, Mapping) else None,
        issues=tuple(issues),
    )


def _check_jsonl_chain(path: Path, label: str, issues: list[str]) -> int:
    previous = "0" * 64
    count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        issues.append(f"{label}_jsonl_unreadable:{exc}")
        return 0
    for sequence, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{label}_jsonl_invalid:{sequence}:{exc}")
            continue
        if record.get("sequence") != count or record.get("previous_hash") != previous:
            issues.append(f"{label}_chain_discontinuity:{sequence}")
        claimed = record.get("record_hash")
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        actual = _HashChainedJsonl._hash(unsigned)
        if claimed != actual:
            issues.append(f"{label}_record_hash_mismatch:{sequence}")
        previous = str(claimed)
        count += 1
    if count == 0:
        issues.append(f"{label}_jsonl_empty")
    return count


def _check_references(
    bundle: Path,
    value: object,
    label: str,
    issues: list[str],
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
    allow_missing: bool = False,
) -> None:
    if not isinstance(value, list):
        issues.append(f"{label}_references_missing")
        return
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            issues.append(f"{label}_reference_invalid:{index}")
            continue
        raw_path = record.get(path_key)
        if allow_missing and raw_path in (None, ""):
            continue
        if not isinstance(raw_path, str) or not raw_path:
            issues.append(f"{label}_path_missing:{index}")
            continue
        supplied = Path(raw_path)
        path = supplied if supplied.is_absolute() else bundle / supplied
        if not path.is_file():
            issues.append(f"{label}_file_missing:{index}")
            continue
        if record.get(hash_key) != _sha256_file(path):
            issues.append(f"{label}_hash_mismatch:{index}")


def _check_authoritative_logs(
    bundle: Path, value: object, issues: list[str]
) -> None:
    if not isinstance(value, list):
        return
    from .event_log import RuntimeEventLog, check_runtime_hand_log
    from .session_log import SessionEventLog, check_session_log

    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            continue
        raw_path = record.get("path")
        kind = record.get("kind")
        if not isinstance(raw_path, str):
            continue
        supplied = Path(raw_path)
        path = supplied if supplied.is_absolute() else bundle / supplied
        if not path.is_file():
            continue
        try:
            if kind == "hand_log":
                checked = check_runtime_hand_log(
                    RuntimeEventLog.from_path(path), allow_voided=True
                )
                if not checked.passed:
                    issues.append(f"hand_log_recheck_failed:{index}")
            elif kind == "session_log":
                checked = check_session_log(
                    SessionEventLog.from_path(path), verify_hand_logs=False
                )
                if not checked.passed:
                    issues.append(f"session_log_recheck_failed:{index}")
        except (OSError, TypeError, ValueError) as exc:
            issues.append(f"{kind}_recheck_error:{index}:{type(exc).__name__}:{exc}")


__all__ = [
    "DiagnosticArtifact",
    "DiagnosticBundleCheck",
    "DiagnosticRun",
    "DiagnosticSink",
    "check_diagnostic_bundle",
]
