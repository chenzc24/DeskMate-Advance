"""Strict, replay-bound truth annotations for Part A continuous evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .evaluation import Annotation, AnnotationLabel, EVENT_NAMES


ANNOTATION_SCHEMA_NAME = "deskmate.ergonomics.annotations"
ANNOTATION_SCHEMA_VERSION = "1.0"
_HARD_MAX_FILE_BYTES = 16 * 1024 * 1024
_HARD_MAX_ANNOTATIONS = 100_000
_HARD_MAX_JSON_DEPTH = 12
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class AnnotationValidationError(ValueError):
    """Raised when an annotation artifact violates the v1 contract."""


@dataclass(frozen=True, slots=True)
class AnnotationLimits:
    max_file_bytes: int = 4 * 1024 * 1024
    max_annotations: int = 10_000

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            ("max_file_bytes", self.max_file_bytes, _HARD_MAX_FILE_BYTES),
            ("max_annotations", self.max_annotations, _HARD_MAX_ANNOTATIONS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > ceiling:
                raise ValueError(f"{name} exceeds the immutable safety ceiling")


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    annotation_set_id: str
    replay_sha256: str
    source_id: str
    artifact_sha256: str
    annotations: tuple[Annotation, ...]

    def __post_init__(self) -> None:
        _safe_id(self.annotation_set_id, "annotation_set_id")
        _sha256(self.replay_sha256, "replay_sha256")
        _safe_id(self.source_id, "source_id")
        _sha256(self.artifact_sha256, "artifact_sha256")
        if not isinstance(self.annotations, tuple) or not self.annotations:
            raise AnnotationValidationError("annotations must be a non-empty tuple")
        if not all(isinstance(item, Annotation) for item in self.annotations):
            raise AnnotationValidationError(
                "annotations must contain only Annotation records"
            )
        _validate_annotation_set(self.annotations)


class AnnotationJsonFile:
    """Read one bounded annotation JSON object and verify its exact identity."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        limits: AnnotationLimits | None = None,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise ValueError("expected_sha256 must be 64 lowercase hex characters")
        if limits is not None and not isinstance(limits, AnnotationLimits):
            raise TypeError("limits must be AnnotationLimits")
        self.path = path.resolve()
        self.expected_sha256 = expected_sha256
        self.limits = limits or AnnotationLimits()

    def load(
        self,
        *,
        replay_sha256: str,
        source_id: str,
    ) -> AnnotationSet:
        if not isinstance(replay_sha256, str) or not _SHA256.fullmatch(replay_sha256):
            raise ValueError("replay_sha256 must be 64 lowercase hex characters")
        if not isinstance(source_id, str) or not _SAFE_ID.fullmatch(source_id):
            raise ValueError("source_id must be a bounded safe identifier")
        raw = self._read_bounded()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != self.expected_sha256:
            raise AnnotationValidationError("annotation artifact SHA-256 mismatch")
        root = _decode_strict_json(raw)
        _enforce_depth(root)
        annotation_set = _parse_root(root, actual_sha256)
        if annotation_set.replay_sha256 != replay_sha256:
            raise AnnotationValidationError(
                "annotation artifact is bound to a different replay"
            )
        if annotation_set.source_id != source_id:
            raise AnnotationValidationError(
                "annotation artifact source does not match replay source"
            )
        if len(annotation_set.annotations) > self.limits.max_annotations:
            raise AnnotationValidationError("annotation count exceeds configured limit")
        return annotation_set

    def _read_bounded(self) -> bytes:
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            raise AnnotationValidationError(
                f"cannot stat annotation artifact: {exc}"
            ) from exc
        if size <= 0:
            raise AnnotationValidationError("annotation artifact must not be empty")
        if size > self.limits.max_file_bytes:
            raise AnnotationValidationError("annotation artifact exceeds byte limit")
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(self.limits.max_file_bytes + 1)
        except OSError as exc:
            raise AnnotationValidationError(
                f"cannot read annotation artifact: {exc}"
            ) from exc
        if len(raw) != size:
            raise AnnotationValidationError("annotation artifact changed while reading")
        if len(raw) > self.limits.max_file_bytes:
            raise AnnotationValidationError("annotation artifact exceeds byte limit")
        return raw


def _decode_strict_json(raw: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise AnnotationValidationError(
                    f"duplicate annotation JSON key: {key}"
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise AnnotationValidationError(
            f"non-finite annotation JSON number: {value}"
        )

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except AnnotationValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AnnotationValidationError(
            "annotation artifact is not strict UTF-8 JSON"
        ) from exc


def _enforce_depth(root: Any) -> None:
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > _HARD_MAX_JSON_DEPTH:
            raise AnnotationValidationError("annotation JSON nesting is too deep")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnnotationValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AnnotationValidationError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AnnotationValidationError(f"{label} must be a bounded safe identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AnnotationValidationError(
            f"{label} must be 64 lowercase hex characters"
        )
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnnotationValidationError(f"{label} must be an integer")
    if value < 0:
        raise AnnotationValidationError(f"{label} must be non-negative")
    return value


def _parse_root(value: Any, artifact_sha256: str) -> AnnotationSet:
    root = _mapping(value, "annotation root")
    _exact_keys(
        root,
        {
            "schema",
            "schema_version",
            "annotation_set_id",
            "replay_sha256",
            "source_id",
            "clock",
            "privacy",
            "annotations",
        },
        "annotation root",
    )
    if (
        root["schema"] != ANNOTATION_SCHEMA_NAME
        or root["schema_version"] != ANNOTATION_SCHEMA_VERSION
    ):
        raise AnnotationValidationError("unsupported annotation schema")
    clock = _mapping(root["clock"], "annotation clock")
    _exact_keys(clock, {"kind", "origin", "unit"}, "annotation clock")
    if clock != {
        "kind": "monotonic",
        "origin": "session_relative",
        "unit": "ns",
    }:
        raise AnnotationValidationError(
            "annotation clock must be monotonic session-relative nanoseconds"
        )
    privacy = _mapping(root["privacy"], "annotation privacy")
    _exact_keys(
        privacy,
        {"contains_direct_identifiers"},
        "annotation privacy",
    )
    if privacy["contains_direct_identifiers"] is not False:
        raise AnnotationValidationError(
            "annotation privacy.contains_direct_identifiers must be false"
        )
    rows = root["annotations"]
    if not isinstance(rows, list) or not rows:
        raise AnnotationValidationError("annotations must be a non-empty array")
    if len(rows) > _HARD_MAX_ANNOTATIONS:
        raise AnnotationValidationError("annotation count exceeds safety ceiling")
    annotations = tuple(_parse_annotation(row, index) for index, row in enumerate(rows))
    _validate_annotation_set(annotations)
    return AnnotationSet(
        annotation_set_id=_safe_id(root["annotation_set_id"], "annotation_set_id"),
        replay_sha256=_sha256(root["replay_sha256"], "replay_sha256"),
        source_id=_safe_id(root["source_id"], "source_id"),
        artifact_sha256=artifact_sha256,
        annotations=annotations,
    )


def _parse_annotation(value: Any, index: int) -> Annotation:
    label = f"annotations[{index}]"
    row = _mapping(value, label)
    _exact_keys(
        row,
        {
            "event_id",
            "event_name",
            "label",
            "onset_ns",
            "offset_ns",
            "eligible_at_ns",
        },
        label,
    )
    event_name = row["event_name"]
    if not isinstance(event_name, str) or event_name not in EVENT_NAMES:
        raise AnnotationValidationError(f"{label}.event_name is unsupported")
    try:
        annotation_label = AnnotationLabel(row["label"])
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{label}.label is unsupported") from exc
    eligible = row["eligible_at_ns"]
    if eligible is not None:
        eligible = _integer(eligible, f"{label}.eligible_at_ns")
    try:
        return Annotation(
            event_id=_safe_id(row["event_id"], f"{label}.event_id"),
            event_name=event_name,
            label=annotation_label,
            onset_ns=_integer(row["onset_ns"], f"{label}.onset_ns"),
            offset_ns=_integer(row["offset_ns"], f"{label}.offset_ns"),
            eligible_at_ns=eligible,
        )
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{label}: {exc}") from exc


def _validate_annotation_set(annotations: tuple[Annotation, ...]) -> None:
    event_ids: set[str] = set()
    by_lane: dict[str, list[Annotation]] = {name: [] for name in EVENT_NAMES}
    for annotation in annotations:
        if annotation.event_id in event_ids:
            raise AnnotationValidationError(
                f"duplicate annotation event_id: {annotation.event_id}"
            )
        event_ids.add(annotation.event_id)
        by_lane[annotation.event_name].append(annotation)
    for event_name, lane in by_lane.items():
        ordered = sorted(lane, key=lambda item: item.onset_ns)
        for previous, current in zip(ordered, ordered[1:]):
            if current.onset_ns < previous.offset_ns:
                raise AnnotationValidationError(
                    f"overlapping annotations for {event_name}"
                )
