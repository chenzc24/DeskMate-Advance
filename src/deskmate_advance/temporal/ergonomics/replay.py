"""Strict, bounded scalar replay for Part A ergonomics.

The replay contract deliberately stops at scalar evidence.  It contains no
camera pixels, face/pose landmarks, audio samples, wall-clock timestamps, or
direct participant identifiers.  The narrow records below duck-type the
attributes consumed by :class:`CalibrationCollector` and
:class:`ErgonomicsRuleEngine`; no synthetic ``FramePacket`` is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from deskmate_advance.perception.ergonomics import (
    AudioLevelObservation,
    ObservationState,
)

from .calibration import CalibrationProfile


SCHEMA_NAME = "deskmate.ergonomics.scalar-replay"
SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_MAX_SIGNED_64 = (1 << 63) - 1
_HARD_MAX_LINE_BYTES = 64 * 1024
_HARD_MAX_RECORDS = 200_000
_HARD_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_PROVENANCE_JSON_BYTES = 4 * 1024 * 1024
_DBFS_ABSOLUTE_TOLERANCE = 1e-6
_MANIFEST_MODEL_IDS = {
    "pose": "pose_landmarker",
    "face": "face_landmarker",
}
_FEATURE_BUNDLE_PATHS = (
    "src/deskmate_advance/features/ergonomics/audio_live.py",
    "src/deskmate_advance/features/ergonomics/face.py",
    "src/deskmate_advance/features/ergonomics/live.py",
    "src/deskmate_advance/features/ergonomics/pose.py",
    "src/deskmate_advance/perception/ergonomics/landmarkers.py",
    "src/deskmate_advance/perception/ergonomics/observations.py",
    "src/deskmate_advance/perception/ergonomics/signals.py",
)
_PRODUCER_BUNDLE_PATHS = (
    "scripts/ergonomics/replay_part_a.py",
    "src/deskmate_advance/temporal/ergonomics/__init__.py",
    "src/deskmate_advance/temporal/ergonomics/blink.py",
    "src/deskmate_advance/temporal/ergonomics/calibration.py",
    "src/deskmate_advance/temporal/ergonomics/candidates.py",
    "src/deskmate_advance/temporal/ergonomics/core.py",
    "src/deskmate_advance/temporal/ergonomics/evaluation.py",
    "src/deskmate_advance/temporal/ergonomics/replay.py",
    "src/deskmate_advance/temporal/ergonomics/rules.py",
)
REPLAY_DATA_STATUSES = frozenset(
    {"labeled_evidence", "unlabeled_screening", "synthetic_contract_test"}
)


class ReplayValidationError(ValueError):
    """Raised when replay bytes violate the frozen scalar contract."""


@dataclass(frozen=True, slots=True)
class ReplayLimits:
    """Hard limits that keep malformed or untrusted streams bounded."""

    max_line_bytes: int = 32 * 1024
    max_records: int = 100_000
    max_file_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_line_bytes", self.max_line_bytes),
            ("max_records", self.max_records),
            ("max_file_bytes", self.max_file_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value, ceiling in (
            ("max_line_bytes", self.max_line_bytes, _HARD_MAX_LINE_BYTES),
            ("max_records", self.max_records, _HARD_MAX_RECORDS),
            ("max_file_bytes", self.max_file_bytes, _HARD_MAX_FILE_BYTES),
        ):
            if value > ceiling:
                raise ValueError(f"{name} exceeds the immutable safety ceiling")


@dataclass(frozen=True, slots=True)
class ReplaySource:
    source_id: str
    device_id: str
    device_index: int


@dataclass(frozen=True, slots=True)
class ReplayModelProvenance:
    model_id: str
    model_version: str
    asset_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayProvenance:
    event_config_sha256: str
    perception_config_sha256: str
    model_manifest_sha256: str
    feature_bundle_sha256: str
    pose_model: ReplayModelProvenance
    face_model: ReplayModelProvenance


@dataclass(frozen=True, slots=True)
class ReplayStaleThresholds:
    pose_ms: int
    face_ms: int
    luminance_ms: int
    audio_ms: int


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    replay_id: str
    data_status: str
    record_count: int
    camera: ReplaySource
    audio: ReplaySource | None
    stale: ReplayStaleThresholds
    provenance: ReplayProvenance
    calibration_profile: CalibrationProfile


@dataclass(frozen=True, slots=True)
class ScalarFrame:
    """Only frame metadata used by temporal logic; intentionally no image."""

    sequence_id: int
    captured_at_ns: int
    source_id: str
    device_index: int
    dropped_before: int


@dataclass(frozen=True, slots=True)
class ScalarPoseFeatures:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    state: ObservationState
    model_id: str
    model_version: str
    asset_sha256: str
    config_sha256: str
    dropped_before: int
    dt_ns: int | None
    temporal_gap: bool
    shoulder_tilt_deg: float | None
    torso_lean_from_vertical_deg: float | None
    upper_body_motion_per_second: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ScalarFaceFeatures:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    state: ObservationState
    model_id: str
    model_version: str
    asset_sha256: str
    config_sha256: str
    dropped_before: int
    dt_ns: int | None
    geometry_state: ObservationState
    rotation_state: ObservationState
    blink_state: ObservationState
    face_bbox_area_ratio: float | None
    raw_rotation_xyz_deg: tuple[float, float, float] | None
    eye_blink_left: float | None
    eye_blink_right: float | None
    eye_blink_mean: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ScalarLuminanceObservation:
    source_id: str
    sequence_id: int
    captured_at_ns: int
    state: ObservationState
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ErgonomicsScalarSnapshot:
    """Privacy-safe duck type for the current calibration/rule input surface."""

    frame: ScalarFrame
    luminance: ScalarLuminanceObservation
    luminance_ran: bool
    luminance_age_ms: float
    luminance_stale: bool
    pose_observation: None
    pose_features: ScalarPoseFeatures | None
    pose_ran: bool
    pose_age_ms: float | None
    pose_stale: bool
    face_observation: None
    face_features: ScalarFaceFeatures | None
    face_ran: bool
    face_age_ms: float | None
    face_stale: bool


@dataclass(frozen=True, slots=True)
class ReplaySample:
    snapshot: ErgonomicsScalarSnapshot
    audio_level: AudioLevelObservation | None
    audio_stale: bool


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    replay_id: str
    artifact_sha256: str
    records: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    dropped_before_total: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_bundle_sha256(project_root: Path) -> str:
    """Hash a canonical map of the scalar-producing feature source files."""

    entries = {
        relative: sha256_file(project_root / relative)
        for relative in _FEATURE_BUNDLE_PATHS
    }
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def producer_bundle_sha256(project_root: Path) -> str:
    """Hash every Part A module that changes replay-to-candidate semantics."""

    entries = {
        relative: sha256_file(project_root / relative)
        for relative in _PRODUCER_BUNDLE_PATHS
    }
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class ReplayFile:
    """A pre-hashed replay artifact with repeatable bounded sample iteration."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        limits: ReplayLimits | None = None,
    ) -> None:
        self.path = path.resolve()
        self.limits = limits or ReplayLimits()
        self.expected_sha256 = _hash_value(expected_sha256, "expected_sha256")
        if not self.path.is_file():
            raise ReplayValidationError(f"replay file does not exist: {self.path}")
        self._snapshot_handle = tempfile.TemporaryFile(mode="w+b")
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with self.path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    total_bytes += len(chunk)
                    if total_bytes > self.limits.max_file_bytes:
                        raise ReplayValidationError(
                            "replay exceeds configured total-byte limit"
                        )
                    digest.update(chunk)
                    self._snapshot_handle.write(chunk)
            self._snapshot_handle.flush()
            self._snapshot_handle.seek(0)
        except Exception:
            self._snapshot_handle.close()
            raise
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != self.expected_sha256:
            self._snapshot_handle.close()
            raise ReplayValidationError(
                "replay SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {actual_sha256}"
            )
        self.artifact_sha256 = actual_sha256
        self.artifact_bytes = total_bytes
        try:
            self.header = self._read_header()
            if self.header.record_count > self.limits.max_records:
                raise ReplayValidationError(
                    "header record_count exceeds configured replay limit"
                )
        except Exception:
            self._snapshot_handle.close()
            raise

    def validate(self) -> ReplaySummary:
        """Parse the complete artifact without retaining its samples."""

        count = 0
        first_timestamp_ns: int | None = None
        last_timestamp_ns: int | None = None
        dropped_total = 0
        for sample in self.iter_samples():
            timestamp_ns = sample.snapshot.frame.captured_at_ns
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns
            dropped_total += sample.snapshot.frame.dropped_before
            count += 1
        if first_timestamp_ns is None or last_timestamp_ns is None:
            raise ReplayValidationError("replay contains no snapshot records")
        return ReplaySummary(
            replay_id=self.header.replay_id,
            artifact_sha256=self.artifact_sha256,
            records=count,
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
            dropped_before_total=dropped_total,
        )

    def iter_samples(self) -> Iterator[ReplaySample]:
        """Yield a latest-only scalar stream, rechecking bytes while parsing."""

        state = _ReplayParserState(self.header)
        digest = hashlib.sha256()
        record_count = 0
        self._snapshot_handle.seek(0)
        for line_number, raw_line in _iter_bounded_lines(
            self._snapshot_handle,
            max_line_bytes=self.limits.max_line_bytes,
        ):
            digest.update(raw_line)
            row = _decode_json_line(raw_line, line_number)
            if line_number == 1:
                parsed_header = _parse_header(row, line_number)
                if parsed_header != self.header:
                    raise ReplayValidationError(
                        "header changed after artifact verification"
                    )
                continue
            record_count += 1
            if record_count > self.limits.max_records:
                raise ReplayValidationError("replay record limit exceeded")
            if record_count > self.header.record_count:
                raise ReplayValidationError(
                    "replay has more records than header.record_count"
                )
            yield state.parse_snapshot(row, line_number)
        if digest.hexdigest() != self.expected_sha256:
            raise ReplayValidationError("replay bytes changed during parsing")
        if record_count != self.header.record_count:
            raise ReplayValidationError(
                "replay record count does not match header: "
                f"expected {self.header.record_count}, got {record_count}"
            )

    def _read_header(self) -> ReplayHeader:
        self._snapshot_handle.seek(0)
        iterator = _iter_bounded_lines(
            self._snapshot_handle,
            max_line_bytes=self.limits.max_line_bytes,
        )
        try:
            line_number, raw_line = next(iterator)
        except StopIteration as exc:
            raise ReplayValidationError("replay file is empty") from exc
        if line_number != 1:
            raise ReplayValidationError("replay header must be the first line")
        return _parse_header(_decode_json_line(raw_line, line_number), line_number)

    def close(self) -> None:
        self._snapshot_handle.close()

    def __enter__(self) -> ReplayFile:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_local_provenance(
    header: ReplayHeader,
    *,
    project_root: Path,
    event_config_path: Path | None = None,
    perception_config_path: Path | None = None,
    model_manifest_path: Path | None = None,
    verify_assets: bool = True,
) -> None:
    """Bind replay producer hashes to the current offline project checkout."""

    root = project_root.resolve()
    event_path = (event_config_path or root / "configs/ergonomics/events.json").resolve()
    perception_path = (
        perception_config_path or root / "configs/ergonomics/perception.json"
    ).resolve()
    manifest_path = (model_manifest_path or root / "models/manifest.yaml").resolve()
    event_config = _read_verified_json(
        event_path,
        header.provenance.event_config_sha256,
        "event config",
    )
    perception_config = _read_verified_json(
        perception_path,
        header.provenance.perception_config_sha256,
        "perception config",
    )
    manifest = _read_verified_json(
        manifest_path,
        header.provenance.model_manifest_sha256,
        "model manifest",
    )
    event_config = _mapping(event_config, "event config root")
    perception_config = _mapping(perception_config, "perception config root")
    manifest = _mapping(manifest, "model manifest root")
    for label, document in (
        ("event config", event_config),
        ("perception config", perception_config),
        ("model manifest", manifest),
    ):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ReplayValidationError(f"{label} schema_version must be {SCHEMA_VERSION}")

    try:
        live_probe = perception_config["live_probe"]
        pose_config = perception_config["pose"]
        face_config = perception_config["face"]
    except (KeyError, TypeError) as exc:
        raise ReplayValidationError("perception config lacks replay provenance fields") from exc
    if not all(isinstance(item, dict) for item in (live_probe, pose_config, face_config)):
        raise ReplayValidationError("perception replay provenance sections must be objects")
    try:
        configured_stale_ms = _positive_int(
            live_probe["stale_after_ms"],
            "perception config live_probe.stale_after_ms",
        )
        luminance_stale_ms = _positive_int(
            event_config["luminance_stale_after_ms"],
            "event config luminance_stale_after_ms",
        )
        audio_stale_ms = _positive_int(
            event_config["audio_stale_after_ms"],
            "event config audio_stale_after_ms",
        )
    except KeyError as exc:
        raise ReplayValidationError("replay configs lack stale-threshold fields") from exc
    if (
        configured_stale_ms != header.stale.pose_ms
        or configured_stale_ms != header.stale.face_ms
        or luminance_stale_ms != header.stale.luminance_ms
        or audio_stale_ms != header.stale.audio_ms
    ):
        raise ReplayValidationError("replay stale thresholds do not match bound configs")

    expected_pose = header.provenance.pose_model
    pose_candidates = tuple(
        _perception_model_entry(
            pose_config,
            prefix=prefix,
            label=f"perception config pose.{prefix}",
            root=root,
        )
        for prefix in ("primary", "fallback")
    )
    matching_pose = [
        item for item in pose_candidates if item[:3] == _model_identity(expected_pose)
    ]
    if len(matching_pose) != 1:
        raise ReplayValidationError(
            "pose model provenance must resolve exactly once in perception config"
        )
    pose_asset_path = matching_pose[0][3]

    expected_face = header.provenance.face_model
    face_entry = _perception_model_entry(
        face_config,
        prefix=None,
        label="perception config face",
        root=root,
    )
    if face_entry[:3] != _model_identity(expected_face):
        raise ReplayValidationError("face model provenance does not match perception config")
    face_asset_path = face_entry[3]

    actual_bundle = feature_bundle_sha256(root)
    if actual_bundle != header.provenance.feature_bundle_sha256:
        raise ReplayValidationError(
            "feature bundle SHA-256 mismatch: "
            f"expected {header.provenance.feature_bundle_sha256}, got {actual_bundle}"
        )

    if not isinstance(manifest.get("models"), list):
        raise ReplayValidationError("model manifest does not contain a models list")
    manifest_models = [
        _manifest_model_entry(item, index=index, root=root)
        for index, item in enumerate(manifest["models"])
    ]
    for label, expected_model, configured_asset_path in (
        ("pose", header.provenance.pose_model, pose_asset_path),
        ("face", header.provenance.face_model, face_asset_path),
    ):
        candidates = [
            item
            for item in manifest_models
            if item[0] == _MANIFEST_MODEL_IDS[label]
            and item[1] == expected_model.model_version
            and item[2] == expected_model.asset_sha256
        ]
        if len(candidates) != 1:
            raise ReplayValidationError(
                f"{label} model provenance does not resolve uniquely in manifest"
            )
        candidate = candidates[0]
        manifest_asset_path = candidate[3]
        if configured_asset_path != manifest_asset_path:
            raise ReplayValidationError(
                f"{label} asset path does not match perception config and manifest"
            )
        if verify_assets:
            if (
                not manifest_asset_path.is_file()
                or sha256_file(manifest_asset_path) != expected_model.asset_sha256
            ):
                raise ReplayValidationError(f"{label} model asset is missing or has changed")


def _read_verified_json(path: Path, expected_sha256: str, label: str) -> Any:
    """Hash and strictly parse the exact same bounded byte snapshot."""

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total_bytes += len(chunk)
                if total_bytes > _MAX_PROVENANCE_JSON_BYTES:
                    raise ReplayValidationError(
                        f"{label} exceeds {_MAX_PROVENANCE_JSON_BYTES} bytes"
                    )
                digest.update(chunk)
                chunks.append(chunk)
    except ReplayValidationError:
        raise
    except OSError as exc:
        raise ReplayValidationError(f"missing or unreadable {label}: {path}") from exc
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReplayValidationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        text = b"".join(chunks).decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except UnicodeDecodeError as exc:
        raise ReplayValidationError(f"{label} is not UTF-8") from exc
    except ReplayValidationError as exc:
        raise ReplayValidationError(f"{label} is not strict JSON: {exc}") from exc
    except RecursionError as exc:
        raise ReplayValidationError(f"{label} JSON nesting is too deep") from exc
    except json.JSONDecodeError as exc:
        raise ReplayValidationError(f"{label} is not valid JSON") from exc


def _model_identity(model: ReplayModelProvenance) -> tuple[str, str, str]:
    return model.model_id, model.model_version, model.asset_sha256


def _perception_model_entry(
    config: Mapping[str, Any],
    *,
    prefix: str | None,
    label: str,
    root: Path,
) -> tuple[str, str, str, Path]:
    field_prefix = f"{prefix}_" if prefix is not None else ""
    try:
        model_id = _safe_id(config[f"{field_prefix}model_id"], f"{label}.model_id")
        model_version = _safe_id(
            config[f"{field_prefix}model_version"], f"{label}.model_version"
        )
        asset_sha256 = _hash_value(
            config[f"{field_prefix}asset_sha256"], f"{label}.asset_sha256"
        )
        asset_path = _normalized_model_path(
            root,
            config[f"{field_prefix}asset"],
            f"{label}.asset",
        )
    except KeyError as exc:
        raise ReplayValidationError(f"{label} identity is incomplete") from exc
    return model_id, model_version, asset_sha256, asset_path


def _manifest_model_entry(
    value: Any,
    *,
    index: int,
    root: Path,
) -> tuple[str, str, str, Path]:
    row = _mapping(value, f"model manifest models[{index}]")
    label = f"model manifest models[{index}]"
    try:
        model_id = _safe_id(row["model_id"], f"{label}.model_id")
        model_version = _safe_id(row["version"], f"{label}.version")
        asset_sha256 = _hash_value(row["sha256"], f"{label}.sha256")
        asset_path = _normalized_model_path(root, row["local_path"], f"{label}.local_path")
    except KeyError as exc:
        raise ReplayValidationError(f"{label} identity is incomplete") from exc
    return model_id, model_version, asset_sha256, asset_path


def _normalized_model_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReplayValidationError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ReplayValidationError(f"{label} must be relative to project root")
    asset_path = (root / relative).resolve()
    models_root = (root / "models").resolve()
    if asset_path == models_root or models_root not in asset_path.parents:
        raise ReplayValidationError(f"{label} escapes models/")
    return asset_path


class _ReplayParserState:
    def __init__(self, header: ReplayHeader) -> None:
        self.header = header
        self._expected_sequence = 0
        self._last_frame_ns: int | None = None
        self._pose: ScalarPoseFeatures | None = None
        self._face: ScalarFaceFeatures | None = None
        self._luminance: ScalarLuminanceObservation | None = None
        self._last_audio: AudioLevelObservation | None = None

    def parse_snapshot(self, value: Any, line_number: int) -> ReplaySample:
        row = _mapping(value, f"line {line_number}")
        _exact_keys(
            row,
            {
                "record_type",
                "schema_version",
                "source_id",
                "device_index",
                "sequence_id",
                "captured_at_ns",
                "dropped_before",
                "pose",
                "face",
                "luminance",
                "audio",
            },
            f"snapshot line {line_number}",
        )
        if row["record_type"] != "snapshot":
            raise ReplayValidationError(
                f"line {line_number}: only snapshot records may follow the header"
            )
        if row["schema_version"] != SCHEMA_VERSION:
            raise ReplayValidationError(f"line {line_number}: schema version mismatch")
        if row["source_id"] != self.header.camera.source_id:
            raise ReplayValidationError(f"line {line_number}: camera source mismatch")
        device_index = _non_negative_int(
            row["device_index"], f"line {line_number} device_index"
        )
        if device_index != self.header.camera.device_index:
            raise ReplayValidationError(f"line {line_number}: camera device mismatch")
        sequence_id = _non_negative_int(
            row["sequence_id"], f"line {line_number} sequence_id"
        )
        if sequence_id != self._expected_sequence:
            raise ReplayValidationError(
                f"line {line_number}: sequence_id must be {self._expected_sequence}"
            )
        timestamp_ns = _non_negative_int(
            row["captured_at_ns"], f"line {line_number} captured_at_ns"
        )
        if self._last_frame_ns is not None and timestamp_ns <= self._last_frame_ns:
            raise ReplayValidationError(
                f"line {line_number}: camera timestamps must increase strictly"
            )
        if self._last_frame_ns is None and (
            self.header.calibration_profile.window_ended_at_ns > timestamp_ns
        ):
            raise ReplayValidationError(
                f"line {line_number}: calibration profile ends after replay begins"
            )
        dropped_before = _non_negative_int(
            row["dropped_before"], f"line {line_number} dropped_before"
        )
        frame = ScalarFrame(
            sequence_id=sequence_id,
            captured_at_ns=timestamp_ns,
            source_id=self.header.camera.source_id,
            device_index=device_index,
            dropped_before=dropped_before,
        )

        pose, pose_ran, pose_stale, pose_age_ms = self._parse_pose(
            row["pose"], frame, line_number
        )
        face, face_ran, face_stale, face_age_ms = self._parse_face(
            row["face"], frame, line_number
        )
        luminance, luminance_ran, luminance_stale, luminance_age_ms = (
            self._parse_luminance(row["luminance"], frame, line_number)
        )
        audio_level, audio_stale = self._parse_audio(
            row["audio"], frame, line_number
        )

        self._expected_sequence += 1
        self._last_frame_ns = timestamp_ns
        return ReplaySample(
            snapshot=ErgonomicsScalarSnapshot(
                frame=frame,
                luminance=luminance,
                luminance_ran=luminance_ran,
                luminance_age_ms=luminance_age_ms,
                luminance_stale=luminance_stale,
                pose_observation=None,
                pose_features=pose,
                pose_ran=pose_ran,
                pose_age_ms=pose_age_ms,
                pose_stale=pose_stale,
                face_observation=None,
                face_features=face,
                face_ran=face_ran,
                face_age_ms=face_age_ms,
                face_stale=face_stale,
            ),
            audio_level=audio_level,
            audio_stale=audio_stale,
        )

    def _parse_pose(
        self,
        value: Any,
        frame: ScalarFrame,
        line_number: int,
    ) -> tuple[ScalarPoseFeatures | None, bool, bool, float | None]:
        envelope = _modality_envelope(value, "pose", line_number)
        ran = envelope["ran"]
        update = envelope["update"]
        if ran:
            update_row = _mapping(update, f"line {line_number} pose.update")
            _exact_keys(
                update_row,
                {
                    "captured_at_ns",
                    "state",
                    "valid",
                    "reason",
                    "temporal_gap",
                    "shoulder_tilt_deg",
                    "torso_lean_from_vertical_deg",
                    "upper_body_motion_per_second",
                },
                f"line {line_number} pose.update",
            )
            captured_at_ns = _fresh_timestamp(update_row, frame, "pose", line_number)
            state = _observation_state(update_row["state"], "pose", line_number)
            _valid_matches_state(update_row["valid"], state, "pose", line_number)
            reason = _reason(update_row["reason"], state, "pose", line_number)
            temporal_gap = _boolean(
                update_row["temporal_gap"], f"line {line_number} pose.temporal_gap"
            )
            dt_ns = (
                captured_at_ns - self._pose.captured_at_ns
                if self._pose is not None
                else None
            )
            gap_requires_flag = frame.dropped_before > 0 or (
                dt_ns is not None
                and dt_ns
                > self.header.stale.pose_ms * _NANOSECONDS_PER_MILLISECOND
            )
            if gap_requires_flag and not temporal_gap:
                raise ReplayValidationError(
                    f"line {line_number}: pose temporal_gap must expose drop/gap"
                )
            shoulder = _optional_finite(
                update_row["shoulder_tilt_deg"],
                f"line {line_number} pose.shoulder_tilt_deg",
            )
            torso = _optional_finite(
                update_row["torso_lean_from_vertical_deg"],
                f"line {line_number} pose.torso_lean_from_vertical_deg",
            )
            motion = _optional_finite(
                update_row["upper_body_motion_per_second"],
                f"line {line_number} pose.upper_body_motion_per_second",
                minimum=0.0,
            )
            if state is not ObservationState.VALID and any(
                item is not None for item in (shoulder, torso, motion)
            ):
                raise ReplayValidationError(
                    f"line {line_number}: invalid pose cannot carry scalar evidence"
                )
            model = self.header.provenance.pose_model
            self._pose = ScalarPoseFeatures(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=captured_at_ns,
                state=state,
                model_id=model.model_id,
                model_version=model.model_version,
                asset_sha256=model.asset_sha256,
                config_sha256=self.header.provenance.perception_config_sha256,
                dropped_before=frame.dropped_before,
                dt_ns=dt_ns,
                temporal_gap=temporal_gap,
                shoulder_tilt_deg=shoulder,
                torso_lean_from_vertical_deg=torso,
                upper_body_motion_per_second=motion,
                reason=reason,
            )
        elif update is not None:
            raise ReplayValidationError(
                f"line {line_number}: pose update requires ran=true"
            )
        return self._finish_modality(
            self._pose,
            ran=ran,
            declared_stale=envelope["stale"],
            frame=frame,
            threshold_ms=self.header.stale.pose_ms,
            label="pose",
            line_number=line_number,
        )

    def _parse_face(
        self,
        value: Any,
        frame: ScalarFrame,
        line_number: int,
    ) -> tuple[ScalarFaceFeatures | None, bool, bool, float | None]:
        envelope = _modality_envelope(value, "face", line_number)
        ran = envelope["ran"]
        update = envelope["update"]
        if ran:
            update_row = _mapping(update, f"line {line_number} face.update")
            _exact_keys(
                update_row,
                {
                    "captured_at_ns",
                    "state",
                    "valid",
                    "reason",
                    "geometry_state",
                    "rotation_state",
                    "blink_state",
                    "face_bbox_area_ratio",
                    "raw_rotation_xyz_deg",
                    "eye_blink_left",
                    "eye_blink_right",
                    "eye_blink_mean",
                },
                f"line {line_number} face.update",
            )
            captured_at_ns = _fresh_timestamp(update_row, frame, "face", line_number)
            state = _observation_state(update_row["state"], "face", line_number)
            _valid_matches_state(update_row["valid"], state, "face", line_number)
            reason = _reason(update_row["reason"], state, "face", line_number)
            geometry_state = _observation_state(
                update_row["geometry_state"], "face.geometry", line_number
            )
            rotation_state = _observation_state(
                update_row["rotation_state"], "face.rotation", line_number
            )
            blink_state = _observation_state(
                update_row["blink_state"], "face.blink", line_number
            )
            if state is not ObservationState.VALID and any(
                item is ObservationState.VALID
                for item in (geometry_state, rotation_state, blink_state)
            ):
                raise ReplayValidationError(
                    f"line {line_number}: invalid face cannot have valid components"
                )
            area = _optional_finite(
                update_row["face_bbox_area_ratio"],
                f"line {line_number} face.face_bbox_area_ratio",
                minimum=0.0,
                minimum_exclusive=True,
            )
            if (geometry_state is ObservationState.VALID) != (area is not None):
                raise ReplayValidationError(
                    f"line {line_number}: face geometry state/value mismatch"
                )
            rotation = _rotation(
                update_row["raw_rotation_xyz_deg"], line_number=line_number
            )
            if (rotation_state is ObservationState.VALID) != (rotation is not None):
                raise ReplayValidationError(
                    f"line {line_number}: face rotation state/value mismatch"
                )
            left = _optional_finite(
                update_row["eye_blink_left"],
                f"line {line_number} face.eye_blink_left",
                minimum=0.0,
                maximum=1.0,
            )
            right = _optional_finite(
                update_row["eye_blink_right"],
                f"line {line_number} face.eye_blink_right",
                minimum=0.0,
                maximum=1.0,
            )
            mean = _optional_finite(
                update_row["eye_blink_mean"],
                f"line {line_number} face.eye_blink_mean",
                minimum=0.0,
                maximum=1.0,
            )
            blink_values_valid = left is not None and right is not None and mean is not None
            if (blink_state is ObservationState.VALID) != blink_values_valid:
                raise ReplayValidationError(
                    f"line {line_number}: face blink state/value mismatch"
                )
            if blink_values_valid and not math.isclose(
                mean, (left + right) / 2, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ReplayValidationError(
                    f"line {line_number}: eye_blink_mean must equal both-eye mean"
                )
            model = self.header.provenance.face_model
            self._face = ScalarFaceFeatures(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=captured_at_ns,
                state=state,
                model_id=model.model_id,
                model_version=model.model_version,
                asset_sha256=model.asset_sha256,
                config_sha256=self.header.provenance.perception_config_sha256,
                dropped_before=frame.dropped_before,
                dt_ns=(
                    captured_at_ns - self._face.captured_at_ns
                    if self._face is not None
                    else None
                ),
                geometry_state=geometry_state,
                rotation_state=rotation_state,
                blink_state=blink_state,
                face_bbox_area_ratio=area,
                raw_rotation_xyz_deg=rotation,
                eye_blink_left=left,
                eye_blink_right=right,
                eye_blink_mean=mean,
                reason=reason,
            )
        elif update is not None:
            raise ReplayValidationError(
                f"line {line_number}: face update requires ran=true"
            )
        return self._finish_modality(
            self._face,
            ran=ran,
            declared_stale=envelope["stale"],
            frame=frame,
            threshold_ms=self.header.stale.face_ms,
            label="face",
            line_number=line_number,
        )

    def _parse_luminance(
        self,
        value: Any,
        frame: ScalarFrame,
        line_number: int,
    ) -> tuple[ScalarLuminanceObservation, bool, bool, float]:
        envelope = _modality_envelope(value, "luminance", line_number)
        ran = envelope["ran"]
        update = envelope["update"]
        if ran:
            update_row = _mapping(update, f"line {line_number} luminance.update")
            _exact_keys(
                update_row,
                {"captured_at_ns", "state", "valid", "reason", "mean", "p90"},
                f"line {line_number} luminance.update",
            )
            captured_at_ns = _fresh_timestamp(
                update_row, frame, "luminance", line_number
            )
            state = _observation_state(
                update_row["state"], "luminance", line_number
            )
            _valid_matches_state(
                update_row["valid"], state, "luminance", line_number
            )
            reason = _reason(
                update_row["reason"], state, "luminance", line_number
            )
            mean = _optional_finite(
                update_row["mean"],
                f"line {line_number} luminance.mean",
                minimum=0.0,
                maximum=255.0,
            )
            p90 = _optional_finite(
                update_row["p90"],
                f"line {line_number} luminance.p90",
                minimum=0.0,
                maximum=255.0,
            )
            if state is ObservationState.VALID:
                if mean is None or p90 is None:
                    raise ReplayValidationError(
                        f"line {line_number}: valid luminance requires mean and p90"
                    )
            elif mean is not None or p90 is not None:
                raise ReplayValidationError(
                    f"line {line_number}: invalid luminance cannot carry values"
                )
            self._luminance = ScalarLuminanceObservation(
                source_id=frame.source_id,
                sequence_id=frame.sequence_id,
                captured_at_ns=captured_at_ns,
                state=state,
                mean=mean,
                median=None,
                p10=None,
                p90=p90,
                reason=reason,
            )
        elif update is not None:
            raise ReplayValidationError(
                f"line {line_number}: luminance update requires ran=true"
            )
        record, ran, stale, age = self._finish_modality(
            self._luminance,
            ran=ran,
            declared_stale=envelope["stale"],
            frame=frame,
            threshold_ms=self.header.stale.luminance_ms,
            label="luminance",
            line_number=line_number,
        )
        if record is None or age is None:
            raise ReplayValidationError(
                f"line {line_number}: luminance must be initialized on the first frame"
            )
        return record, ran, stale, age

    def _parse_audio(
        self,
        value: Any,
        frame: ScalarFrame,
        line_number: int,
    ) -> tuple[AudioLevelObservation | None, bool]:
        if value is None:
            return None, True
        if self.header.audio is None:
            raise ReplayValidationError(
                f"line {line_number}: audio evidence conflicts with disabled header"
            )
        row = _mapping(value, f"line {line_number} audio")
        _exact_keys(
            row,
            {
                "source_id",
                "device_index",
                "state",
                "valid",
                "stale",
                "reason",
                "window_started_at_ns",
                "window_ended_at_ns",
                "sample_rate_hz",
                "sample_count",
                "rms",
                "dbfs",
            },
            f"line {line_number} audio",
        )
        if row["source_id"] != self.header.audio.source_id:
            raise ReplayValidationError(f"line {line_number}: audio source mismatch")
        device_index = _non_negative_int(
            row["device_index"], f"line {line_number} audio.device_index"
        )
        if device_index != self.header.audio.device_index:
            raise ReplayValidationError(f"line {line_number}: audio device mismatch")
        state = _observation_state(row["state"], "audio", line_number)
        _valid_matches_state(row["valid"], state, "audio", line_number)
        reason = _reason(row["reason"], state, "audio", line_number)
        start_ns = _non_negative_int(
            row["window_started_at_ns"],
            f"line {line_number} audio.window_started_at_ns",
        )
        end_ns = _non_negative_int(
            row["window_ended_at_ns"],
            f"line {line_number} audio.window_ended_at_ns",
        )
        if end_ns <= start_ns:
            raise ReplayValidationError(
                f"line {line_number}: audio window must increase strictly"
            )
        sample_rate_hz = _positive_int(
            row["sample_rate_hz"], f"line {line_number} audio.sample_rate_hz"
        )
        sample_count = _non_negative_int(
            row["sample_count"], f"line {line_number} audio.sample_count"
        )
        if state is ObservationState.VALID and sample_count == 0:
            raise ReplayValidationError(
                f"line {line_number}: valid audio requires samples"
            )
        if state is ObservationState.VALID:
            expected_duration_ns = max(
                1,
                round(sample_count * 1_000_000_000 / sample_rate_hz),
            )
            if end_ns - start_ns != expected_duration_ns:
                raise ReplayValidationError(
                    f"line {line_number}: audio window duration does not match "
                    "sample_count/sample_rate_hz"
                )
        rms = _optional_finite(
            row["rms"],
            f"line {line_number} audio.rms",
            minimum=0.0,
            maximum=1.0,
        )
        dbfs = _optional_finite(
            row["dbfs"],
            f"line {line_number} audio.dbfs",
            maximum=0.0,
        )
        if state is ObservationState.VALID:
            if rms is None or dbfs is None:
                raise ReplayValidationError(
                    f"line {line_number}: valid audio requires rms and dbfs"
                )
            if rms == 0.0:
                if dbfs >= 0.0:
                    raise ReplayValidationError(
                        f"line {line_number}: silent audio dbfs must be negative"
                    )
            else:
                expected_dbfs = 20.0 * math.log10(rms)
                if not math.isclose(
                    dbfs,
                    expected_dbfs,
                    rel_tol=1e-9,
                    abs_tol=_DBFS_ABSOLUTE_TOLERANCE,
                ):
                    raise ReplayValidationError(
                        f"line {line_number}: audio rms and dbfs are inconsistent"
                    )
        elif rms is not None or dbfs is not None:
            raise ReplayValidationError(
                f"line {line_number}: invalid audio cannot carry level values"
            )
        expected_stale = (
            end_ns > frame.captured_at_ns
            or frame.captured_at_ns - end_ns
            > self.header.stale.audio_ms * _NANOSECONDS_PER_MILLISECOND
        )
        declared_stale = _boolean(
            row["stale"], f"line {line_number} audio.stale"
        )
        if declared_stale != expected_stale:
            raise ReplayValidationError(
                f"line {line_number}: audio stale flag does not match timestamps"
            )
        observation = AudioLevelObservation(
                source_id=self.header.audio.source_id,
                window_started_at_ns=start_ns,
                window_ended_at_ns=end_ns,
                sample_rate_hz=sample_rate_hz,
                sample_count=sample_count,
                state=state,
                rms=rms,
                dbfs=dbfs,
                reason=reason,
        )
        previous = self._last_audio
        if previous is not None:
            if observation.window_ended_at_ns < previous.window_ended_at_ns:
                raise ReplayValidationError(
                    f"line {line_number}: audio window timestamps must not regress"
                )
            if (
                observation.window_ended_at_ns == previous.window_ended_at_ns
                and observation != previous
            ):
                raise ReplayValidationError(
                    f"line {line_number}: repeated audio window changed content"
                )
        self._last_audio = observation
        return observation, declared_stale

    @staticmethod
    def _finish_modality(
        record: Any | None,
        *,
        ran: bool,
        declared_stale: Any,
        frame: ScalarFrame,
        threshold_ms: int,
        label: str,
        line_number: int,
    ) -> tuple[Any | None, bool, bool, float | None]:
        stale = _boolean(
            declared_stale, f"line {line_number} {label}.stale"
        )
        age_ms = (
            (frame.captured_at_ns - record.captured_at_ns)
            / _NANOSECONDS_PER_MILLISECOND
            if record is not None
            else None
        )
        if age_ms is not None and age_ms < 0:
            raise ReplayValidationError(
                f"line {line_number}: {label} timestamp is later than frame"
            )
        expected_stale = age_ms is None or age_ms > threshold_ms
        if stale != expected_stale:
            raise ReplayValidationError(
                f"line {line_number}: {label} stale flag does not match age"
            )
        return record, ran, stale, age_ms


def _parse_header(value: Any, line_number: int) -> ReplayHeader:
    row = _mapping(value, f"line {line_number}")
    _exact_keys(
        row,
        {
            "record_type",
            "schema",
            "schema_version",
            "replay_id",
            "data_status",
            "record_count",
            "clock",
            "camera",
            "audio",
            "stale_after_ms",
            "provenance",
            "calibration_profile",
            "privacy",
        },
        "replay header",
    )
    if row["record_type"] != "header":
        raise ReplayValidationError("first JSONL record must be a header")
    if row["schema"] != SCHEMA_NAME or row["schema_version"] != SCHEMA_VERSION:
        raise ReplayValidationError("unsupported scalar replay schema")
    replay_id = _safe_id(row["replay_id"], "replay_id")
    data_status = row["data_status"]
    if not isinstance(data_status, str) or data_status not in REPLAY_DATA_STATUSES:
        raise ReplayValidationError("unsupported replay data_status")
    record_count = _positive_int(row["record_count"], "record_count")

    clock = _mapping(row["clock"], "clock")
    _exact_keys(clock, {"kind", "origin", "unit"}, "clock")
    if clock != {
        "kind": "monotonic",
        "origin": "session_relative",
        "unit": "ns",
    }:
        raise ReplayValidationError(
            "clock must be monotonic, session-relative nanoseconds"
        )
    camera = _parse_source(row["camera"], "camera")
    audio = (
        _parse_source(row["audio"], "audio") if row["audio"] is not None else None
    )

    stale_row = _mapping(row["stale_after_ms"], "stale_after_ms")
    _exact_keys(stale_row, {"pose", "face", "luminance", "audio"}, "stale_after_ms")
    stale = ReplayStaleThresholds(
        pose_ms=_positive_int(stale_row["pose"], "stale_after_ms.pose"),
        face_ms=_positive_int(stale_row["face"], "stale_after_ms.face"),
        luminance_ms=_positive_int(
            stale_row["luminance"], "stale_after_ms.luminance"
        ),
        audio_ms=_positive_int(stale_row["audio"], "stale_after_ms.audio"),
    )

    provenance_row = _mapping(row["provenance"], "provenance")
    _exact_keys(
        provenance_row,
        {
            "event_config_sha256",
            "perception_config_sha256",
            "model_manifest_sha256",
            "feature_bundle_sha256",
            "pose_model",
            "face_model",
        },
        "provenance",
    )
    provenance = ReplayProvenance(
        event_config_sha256=_hash_value(
            provenance_row["event_config_sha256"], "event_config_sha256"
        ),
        perception_config_sha256=_hash_value(
            provenance_row["perception_config_sha256"], "perception_config_sha256"
        ),
        model_manifest_sha256=_hash_value(
            provenance_row["model_manifest_sha256"], "model_manifest_sha256"
        ),
        feature_bundle_sha256=_hash_value(
            provenance_row["feature_bundle_sha256"], "feature_bundle_sha256"
        ),
        pose_model=_parse_model(provenance_row["pose_model"], "pose_model"),
        face_model=_parse_model(provenance_row["face_model"], "face_model"),
    )

    privacy = _mapping(row["privacy"], "privacy")
    privacy_fields = {
        "contains_images",
        "contains_landmarks",
        "contains_audio_samples",
        "contains_direct_identifiers",
    }
    _exact_keys(privacy, privacy_fields, "privacy")
    for key in privacy_fields:
        if _boolean(privacy[key], f"privacy.{key}"):
            raise ReplayValidationError(f"privacy.{key} must be false")

    profile = _parse_profile(row["calibration_profile"], camera)
    return ReplayHeader(
        replay_id=replay_id,
        data_status=data_status,
        record_count=record_count,
        camera=camera,
        audio=audio,
        stale=stale,
        provenance=provenance,
        calibration_profile=profile,
    )


def _parse_source(value: Any, label: str) -> ReplaySource:
    row = _mapping(value, label)
    _exact_keys(row, {"source_id", "device_id", "device_index"}, label)
    return ReplaySource(
        source_id=_safe_id(row["source_id"], f"{label}.source_id"),
        device_id=_safe_id(row["device_id"], f"{label}.device_id"),
        device_index=_non_negative_int(row["device_index"], f"{label}.device_index"),
    )


def _parse_model(value: Any, label: str) -> ReplayModelProvenance:
    row = _mapping(value, label)
    _exact_keys(row, {"model_id", "model_version", "asset_sha256"}, label)
    return ReplayModelProvenance(
        model_id=_safe_id(row["model_id"], f"{label}.model_id"),
        model_version=_safe_id(row["model_version"], f"{label}.model_version"),
        asset_sha256=_hash_value(row["asset_sha256"], f"{label}.asset_sha256"),
    )


def _parse_profile(value: Any, camera: ReplaySource) -> CalibrationProfile:
    row = _mapping(value, "calibration_profile")
    fields = {
        "source_id",
        "device_index",
        "window_started_at_ns",
        "window_ended_at_ns",
        "pose_samples",
        "torso_lean_samples",
        "face_samples",
        "eye_open_samples",
        "luminance_samples",
        "shoulder_tilt_deg",
        "torso_lean_from_vertical_deg",
        "face_bbox_area_ratio",
        "head_rotation_x_deg",
        "head_rotation_y_deg",
        "eye_open_score",
        "mean_luminance",
        "p90_luminance",
    }
    _exact_keys(row, fields, "calibration_profile")
    if row["source_id"] != camera.source_id:
        raise ReplayValidationError("calibration profile source mismatch")
    device_index = _non_negative_int(
        row["device_index"], "calibration_profile.device_index"
    )
    if device_index != camera.device_index:
        raise ReplayValidationError("calibration profile device mismatch")
    started_at_ns = _non_negative_int(
        row["window_started_at_ns"], "calibration_profile.window_started_at_ns"
    )
    ended_at_ns = _non_negative_int(
        row["window_ended_at_ns"], "calibration_profile.window_ended_at_ns"
    )
    if ended_at_ns <= started_at_ns:
        raise ReplayValidationError("calibration profile window must increase")
    pose_samples = _positive_int(row["pose_samples"], "profile.pose_samples")
    torso_samples = _non_negative_int(
        row["torso_lean_samples"], "profile.torso_lean_samples"
    )
    face_samples = _positive_int(row["face_samples"], "profile.face_samples")
    eye_samples = _non_negative_int(
        row["eye_open_samples"], "profile.eye_open_samples"
    )
    luminance_samples = _positive_int(
        row["luminance_samples"], "profile.luminance_samples"
    )
    shoulder = _finite(row["shoulder_tilt_deg"], "profile.shoulder_tilt_deg")
    torso = _optional_finite(
        row["torso_lean_from_vertical_deg"],
        "profile.torso_lean_from_vertical_deg",
    )
    if (torso is None) != (torso_samples == 0):
        raise ReplayValidationError("profile torso baseline/sample count mismatch")
    area = _finite(
        row["face_bbox_area_ratio"],
        "profile.face_bbox_area_ratio",
        minimum=0.0,
        minimum_exclusive=True,
    )
    head_x = _finite(row["head_rotation_x_deg"], "profile.head_rotation_x_deg")
    head_y = _finite(row["head_rotation_y_deg"], "profile.head_rotation_y_deg")
    eye_open = _optional_finite(
        row["eye_open_score"],
        "profile.eye_open_score",
        minimum=0.0,
        maximum=1.0,
    )
    if (eye_open is None) != (eye_samples == 0):
        raise ReplayValidationError("profile eye baseline/sample count mismatch")
    mean = _finite(
        row["mean_luminance"],
        "profile.mean_luminance",
        minimum=0.0,
        maximum=255.0,
    )
    p90 = _finite(
        row["p90_luminance"],
        "profile.p90_luminance",
        minimum=0.0,
        maximum=255.0,
    )
    return CalibrationProfile(
        source_id=camera.source_id,
        device_index=device_index,
        window_started_at_ns=started_at_ns,
        window_ended_at_ns=ended_at_ns,
        pose_samples=pose_samples,
        torso_lean_samples=torso_samples,
        face_samples=face_samples,
        eye_open_samples=eye_samples,
        luminance_samples=luminance_samples,
        shoulder_tilt_deg=shoulder,
        torso_lean_from_vertical_deg=torso,
        face_bbox_area_ratio=area,
        head_rotation_x_deg=head_x,
        head_rotation_y_deg=head_y,
        eye_open_score=eye_open,
        mean_luminance=mean,
        p90_luminance=p90,
    )


def _modality_envelope(value: Any, label: str, line_number: int) -> Mapping[str, Any]:
    row = _mapping(value, f"line {line_number} {label}")
    _exact_keys(row, {"ran", "stale", "update"}, f"line {line_number} {label}")
    ran = _boolean(row["ran"], f"line {line_number} {label}.ran")
    if ran != (row["update"] is not None):
        raise ReplayValidationError(
            f"line {line_number}: {label}.ran must match update presence"
        )
    _boolean(row["stale"], f"line {line_number} {label}.stale")
    return row


def _fresh_timestamp(
    row: Mapping[str, Any],
    frame: ScalarFrame,
    label: str,
    line_number: int,
) -> int:
    value = _non_negative_int(
        row["captured_at_ns"], f"line {line_number} {label}.captured_at_ns"
    )
    if value != frame.captured_at_ns:
        raise ReplayValidationError(
            f"line {line_number}: fresh {label} timestamp must match frame"
        )
    return value


def _rotation(value: Any, *, line_number: int) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ReplayValidationError(
            f"line {line_number}: raw_rotation_xyz_deg must be a three-item list"
        )
    output = tuple(
        _finite(item, f"line {line_number} raw_rotation_xyz_deg[{index}]")
        for index, item in enumerate(value)
    )
    return output  # type: ignore[return-value]


def _observation_state(value: Any, label: str, line_number: int) -> ObservationState:
    if not isinstance(value, str):
        raise ReplayValidationError(f"line {line_number}: {label} state must be text")
    try:
        return ObservationState(value)
    except ValueError as exc:
        raise ReplayValidationError(
            f"line {line_number}: invalid {label} state: {value}"
        ) from exc


def _valid_matches_state(
    value: Any,
    state: ObservationState,
    label: str,
    line_number: int,
) -> None:
    valid = _boolean(value, f"line {line_number} {label}.valid")
    if valid != (state is ObservationState.VALID):
        raise ReplayValidationError(
            f"line {line_number}: {label}.valid does not match state"
        )


def _reason(
    value: Any,
    state: ObservationState,
    label: str,
    line_number: int,
) -> str | None:
    if state is ObservationState.VALID:
        if value is not None:
            raise ReplayValidationError(
                f"line {line_number}: valid {label} reason must be null"
            )
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ReplayValidationError(
            f"line {line_number}: invalid {label} requires a bounded reason"
        )
    return value


def _iter_bounded_lines(
    handle: Any,
    *,
    max_line_bytes: int,
) -> Iterator[tuple[int, bytes]]:
    line_number = 0
    while True:
        raw_line = handle.readline(max_line_bytes + 1)
        if not raw_line:
            return
        line_number += 1
        if len(raw_line) > max_line_bytes:
            raise ReplayValidationError(
                f"line {line_number} exceeds {max_line_bytes} bytes"
            )
        if not raw_line.strip():
            raise ReplayValidationError(f"blank JSONL line at {line_number}")
        yield line_number, raw_line


def _decode_json_line(raw_line: bytes, line_number: int) -> Any:
    try:
        text = raw_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReplayValidationError(f"line {line_number} is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (json.JSONDecodeError, ReplayValidationError, RecursionError) as exc:
        if isinstance(exc, ReplayValidationError):
            raise ReplayValidationError(f"line {line_number}: {exc}") from exc
        if isinstance(exc, RecursionError):
            raise ReplayValidationError(
                f"line {line_number}: JSON nesting is too deep"
            ) from exc
        raise ReplayValidationError(f"invalid JSON on line {line_number}") from exc


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ReplayValidationError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_non_finite_constant(value: str) -> None:
    raise ReplayValidationError(f"non-finite JSON number: {value}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReplayValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing={','.join(missing)}")
    if unknown:
        details.append(f"unknown={','.join(unknown)}")
    raise ReplayValidationError(f"{label} fields do not match schema ({'; '.join(details)})")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ReplayValidationError(
            f"{label} must be a 1-64 character opaque safe identifier"
        )
    return value


def _hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReplayValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReplayValidationError(f"{label} must be boolean")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SIGNED_64
    ):
        raise ReplayValidationError(
            f"{label} must be a non-negative signed-64-bit integer"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    output = _non_negative_int(value, label)
    if output == 0:
        raise ReplayValidationError(f"{label} must be positive")
    return output


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayValidationError(f"{label} must be numeric")
    try:
        output = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReplayValidationError(f"{label} must be finite") from exc
    if not math.isfinite(output):
        raise ReplayValidationError(f"{label} must be finite")
    if minimum is not None and (
        output <= minimum if minimum_exclusive else output < minimum
    ):
        qualifier = "greater than" if minimum_exclusive else "at least"
        raise ReplayValidationError(f"{label} must be {qualifier} {minimum}")
    if maximum is not None and output > maximum:
        raise ReplayValidationError(f"{label} must be at most {maximum}")
    return output


def _optional_finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _finite(
        value,
        label,
        minimum=minimum,
        maximum=maximum,
        minimum_exclusive=minimum_exclusive,
    )
