"""Manifest, timestamp and bounded metric helpers for Part A A2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator

import numpy as np

from deskmate_advance.perception.ergonomics import ObservationState


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SPLITS = {"train", "selection", "test"}
_ALLOWED_LICENSE = {"project_recording_approved", "redistribution_approved"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RecordingRecord:
    sample_id: str
    media_path: Path
    media_sha256: str
    timestamp_path: Path
    timestamp_sha256: str
    participant_id: str
    session_id: str
    device_id: str
    scenario: str
    scenario_tags: tuple[str, ...]
    split: str


@dataclass(frozen=True, slots=True)
class TimestampEntry:
    frame_index: int
    captured_at_ns: int
    dropped_before: int


def _owned_raw_path(project_root: Path, value: str, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the project root")
    resolved = (project_root / relative).resolve()
    raw_root = (project_root / "data" / "raw").resolve()
    if resolved != raw_root and raw_root not in resolved.parents:
        raise ValueError(f"{field} must resolve under data/raw")
    return resolved


def load_recording_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    verify_files: bool = True,
) -> list[RecordingRecord]:
    records: list[RecordingRecord] = []
    sample_ids: set[str] = set()
    content_splits: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            required = {
                "sample_id",
                "local_path",
                "sha256",
                "timestamp_sidecar_path",
                "timestamp_sidecar_sha256",
                "participant_id",
                "session_id",
                "device_id",
                "scenario",
                "scenario_tags",
                "split",
                "consent_status",
                "license_status",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(
                    f"manifest line {line_number} missing fields: {', '.join(missing)}"
                )
            sample_id = str(row["sample_id"]).strip()
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"duplicate or empty sample_id on line {line_number}")
            sample_ids.add(sample_id)
            media_hash = str(row["sha256"])
            timestamp_hash = str(row["timestamp_sidecar_sha256"])
            if not _SHA256.fullmatch(media_hash) or not _SHA256.fullmatch(timestamp_hash):
                raise ValueError(f"invalid SHA-256 on manifest line {line_number}")
            split = str(row["split"])
            if split not in _ALLOWED_SPLITS:
                raise ValueError(f"invalid split on manifest line {line_number}: {split}")
            previous_split = content_splits.setdefault(media_hash, split)
            if previous_split != split:
                raise ValueError("identical media hash appears in multiple splits")
            if row["consent_status"] != "confirmed":
                raise ValueError(f"unconfirmed consent on manifest line {line_number}")
            if row["license_status"] not in _ALLOWED_LICENSE:
                raise ValueError(f"unapproved license on manifest line {line_number}")
            tags = row["scenario_tags"]
            if not isinstance(tags, list) or not all(
                isinstance(item, str) and item.strip() for item in tags
            ):
                raise ValueError(f"invalid scenario_tags on manifest line {line_number}")
            media_path = _owned_raw_path(project_root, row["local_path"], "local_path")
            timestamp_path = _owned_raw_path(
                project_root,
                row["timestamp_sidecar_path"],
                "timestamp_sidecar_path",
            )
            if verify_files:
                for path, expected, label in (
                    (media_path, media_hash, "media"),
                    (timestamp_path, timestamp_hash, "timestamp sidecar"),
                ):
                    if not path.is_file():
                        raise ValueError(f"missing {label}: {path}")
                    if sha256_file(path) != expected:
                        raise ValueError(f"SHA-256 mismatch for {label}: {path}")
            records.append(
                RecordingRecord(
                    sample_id=sample_id,
                    media_path=media_path,
                    media_sha256=media_hash,
                    timestamp_path=timestamp_path,
                    timestamp_sha256=timestamp_hash,
                    participant_id=str(row["participant_id"]),
                    session_id=str(row["session_id"]),
                    device_id=str(row["device_id"]),
                    scenario=str(row["scenario"]),
                    scenario_tags=tuple(tags),
                    split=split,
                )
            )
    if not records:
        raise ValueError("recording manifest is empty")
    return records


def iter_timestamp_sidecar(path: Path) -> Iterator[TimestampEntry]:
    previous_ns: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for expected_index, raw_line in enumerate(handle):
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid timestamp JSON on line {expected_index + 1}"
                ) from exc
            frame_index = row.get("frame_index")
            captured_at_ns = row.get("captured_at_ns")
            dropped_before = row.get("dropped_before", 0)
            if frame_index != expected_index:
                raise ValueError(
                    f"timestamp frame index must be contiguous at {expected_index}"
                )
            if not isinstance(captured_at_ns, int) or captured_at_ns < 0:
                raise ValueError(f"invalid captured_at_ns at frame {expected_index}")
            if previous_ns is not None and (
                captured_at_ns <= previous_ns
                or captured_at_ns // 1_000_000 <= previous_ns // 1_000_000
            ):
                raise ValueError(
                    f"timestamps must increase at millisecond resolution at frame {expected_index}"
                )
            if not isinstance(dropped_before, int) or dropped_before < 0:
                raise ValueError(f"invalid dropped_before at frame {expected_index}")
            previous_ns = captured_at_ns
            yield TimestampEntry(frame_index, captured_at_ns, dropped_before)


class Reservoir:
    """Deterministic bounded reservoir for approximate latency percentiles."""

    def __init__(self, capacity: int = 10_000, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._values: list[float] = []
        self._seen = 0
        self._random = random.Random(seed)

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self._seen += 1
        if len(self._values) < self.capacity:
            self._values.append(float(value))
            return
        replacement = self._random.randrange(self._seen)
        if replacement < self.capacity:
            self._values[replacement] = float(value)

    def summary(self) -> dict[str, float | int | None]:
        if not self._values:
            return {
                "seen": self._seen,
                "sampled": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "p99": None,
            }
        values = np.asarray(self._values, dtype=np.float64)
        return {
            "seen": self._seen,
            "sampled": len(self._values),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }


class ComponentMetrics:
    """Bounded state, latency and feature-availability aggregation."""

    def __init__(self, feature_names: Iterable[str]) -> None:
        self.total = 0
        self.states: Counter[str] = Counter()
        self.features: Counter[str] = Counter()
        self.feature_names = tuple(feature_names)
        self.latency_ms = Reservoir()
        self.timestamp_delta_ms = Reservoir()
        self.dropped_before = 0
        self._previous_timestamp_ns: int | None = None
        self._invalid_started_ns: int | None = None
        self.max_contiguous_invalid_ms = 0.0

    def begin_recording(self) -> None:
        self._finish_invalid_run(self._previous_timestamp_ns)
        self._previous_timestamp_ns = None
        self._invalid_started_ns = None

    def add(
        self,
        *,
        state: ObservationState,
        timestamp_ns: int,
        inference_ms: float,
        dropped_before: int,
        available: dict[str, bool],
    ) -> None:
        self.total += 1
        self.states[state.value] += 1
        self.latency_ms.add(inference_ms)
        self.dropped_before += dropped_before
        if self._previous_timestamp_ns is not None:
            self.timestamp_delta_ms.add(
                (timestamp_ns - self._previous_timestamp_ns) / 1_000_000
            )
        if state is ObservationState.VALID:
            self._finish_invalid_run(timestamp_ns)
        elif self._invalid_started_ns is None:
            self._invalid_started_ns = timestamp_ns
        self._previous_timestamp_ns = timestamp_ns
        for name in self.feature_names:
            if available.get(name, False):
                self.features[name] += 1

    def summary(self) -> dict[str, Any]:
        self._finish_invalid_run(self._previous_timestamp_ns)
        state_counts = {state.value: self.states[state.value] for state in ObservationState}
        return {
            "processed": self.total,
            "state_counts": state_counts,
            "state_rates": {
                name: (count / self.total if self.total else None)
                for name, count in state_counts.items()
            },
            "feature_available_rates": {
                name: (self.features[name] / self.total if self.total else None)
                for name in self.feature_names
            },
            "inference_ms": self.latency_ms.summary(),
            "timestamp_delta_ms": self.timestamp_delta_ms.summary(),
            "dropped_before_total": self.dropped_before,
            "max_contiguous_invalid_ms": self.max_contiguous_invalid_ms,
        }

    def _finish_invalid_run(self, ended_ns: int | None) -> None:
        if self._invalid_started_ns is None or ended_ns is None:
            return
        duration_ms = max(0.0, (ended_ns - self._invalid_started_ns) / 1_000_000)
        self.max_contiguous_invalid_ms = max(
            self.max_contiguous_invalid_ms, duration_ms
        )
        self._invalid_started_ns = None
