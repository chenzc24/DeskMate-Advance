"""Versioned Part A event candidates, deliberately below ``UnifiedEvent``.

This module exposes controller-independent, replayable candidate records.  It
does not define acknowledgement, controller receipt, cross-feature priority,
suggested actions, or hardware commands.  Each ergonomic condition owns an
independent episode lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO, Iterator

from .core import ConditionState, SemanticState, TemporalPhase
from .rules import ErgonomicsRuleSnapshot, RuleEvaluation


CANDIDATE_SCHEMA_VERSION = "part-a-event-candidate/1.0"
PART_A_EVENT_NAMES = (
    "static_too_long",
    "bad_posture",
    "screen_too_close",
    "head_off_center",
    "low_blink_rate",
    "environment_too_dark",
    "environment_too_bright",
    "noise_too_high",
)

JsonScalar = str | float | int | bool | None

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EPISODE_ID = re.compile(r"^parta-[0-9a-f]{32}$")
_MAX_EVIDENCE_ITEMS = 32
_MAX_EVIDENCE_TEXT = 512
_HARD_MAX_CANDIDATE_LINE_BYTES = 64 * 1024
_HARD_MAX_CANDIDATE_RECORDS = 200_000
_HARD_MAX_CANDIDATE_FILE_BYTES = 512 * 1024 * 1024
_HARD_MAX_JSON_DEPTH = 32
_DATA_STATUSES = frozenset(
    {"labeled_evidence", "unlabeled_screening", "synthetic_contract_test"}
)

_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "event_name",
        "transition",
        "semantic_state",
        "condition",
        "episode_id",
        "source_id",
        "sequence_id",
        "observed_at_ns",
        "confirmed_at_ns",
        "duration_ms",
        "confidence",
        "reason_code",
        "supporting_evidence",
        "context",
    }
)
_CONFIDENCE_FIELDS = frozenset({"value", "status", "method_id"})
_CONTEXT_FIELDS = frozenset(
    {
        "producer_id",
        "producer_version",
        "rule_config_schema_version",
        "rule_config_status",
        "rule_config_sha256",
        "calibration_profile_sha256",
        "trace_id",
        "data_status",
        "producer_bundle_sha256",
        "feature_bundle_sha256",
        "model_manifest_sha256",
        "provenance_verified",
        "assets_verified",
        "input_artifact_sha256",
        "components",
    }
)
_COMPONENT_FIELDS = frozenset(
    {"role", "model_id", "model_version", "asset_sha256", "config_sha256"}
)
_FORBIDDEN_EVIDENCE_TOKENS = frozenset(
    {
        "action",
        "actuation",
        "arduino",
        "command",
        "controller",
        "motor",
        "pwm",
        "servo",
        "speed",
        "velocity",
    }
)
_FORBIDDEN_EVIDENCE_PATTERN = re.compile(
    r"motor|servo|arduino|pwm|velocity|command|controller|actuat",
    re.IGNORECASE,
)
_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "absolute_distance_claimed",
        "age_ms",
        "blink_count",
        "blinks_per_minute",
        "cooldown_remaining_ms",
        "dbfs",
        "direction_sign_calibrated",
        "entry_or_exit_evidence_ms",
        "enter_below",
        "evidence_continuous",
        "exit_above",
        "face_area_ratio_to_baseline",
        "mean_luminance",
        "motion_body_per_second",
        "p90_luminance",
        "raw_x_delta_deg",
        "raw_y_delta_deg",
        "shoulder_delta_deg",
        "spl_calibrated",
        "temporal_phase",
        "torso_coverage",
        "torso_delta_deg",
        "valid_eye_ms",
        "window_ended_at_ns",
        "window_started_at_ns",
    }
)


class CandidateValidationError(ValueError):
    """Raised when candidate JSONL violates the bounded consumer contract."""


@dataclass(frozen=True, slots=True)
class CandidateJsonlLimits:
    """Configurable limits capped by immutable candidate-stream ceilings."""

    max_line_bytes: int = 32 * 1024
    max_records: int = 100_000
    max_file_bytes: int = 256 * 1024 * 1024
    max_json_depth: int = 16

    def __post_init__(self) -> None:
        values = (
            ("max_line_bytes", self.max_line_bytes, _HARD_MAX_CANDIDATE_LINE_BYTES),
            ("max_records", self.max_records, _HARD_MAX_CANDIDATE_RECORDS),
            ("max_file_bytes", self.max_file_bytes, _HARD_MAX_CANDIDATE_FILE_BYTES),
            ("max_json_depth", self.max_json_depth, _HARD_MAX_JSON_DEPTH),
        )
        for name, value, ceiling in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > ceiling:
                raise ValueError(f"{name} exceeds the immutable safety ceiling")


@dataclass(frozen=True, slots=True)
class CandidateJsonlSummary:
    """Compact result of validating a complete candidate artifact."""

    artifact_sha256: str
    artifact_bytes: int
    records: int
    source_id: str | None
    context_sha256: str | None
    data_status: str | None
    input_artifact_sha256: str | None
    first_sequence_id: int | None
    last_sequence_id: int | None
    first_observed_at_ns: int | None
    last_observed_at_ns: int | None


class CandidateTransition(StrEnum):
    """Candidate-lane transitions; these are not UnifiedEvent states."""

    START = "start"
    UPDATE = "update"
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    CLEAR = "clear"


class ConfidenceStatus(StrEnum):
    """Whether a semantic confidence value has empirical calibration."""

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConfidenceEstimate:
    """Probability that the candidate's semantic claim is correct.

    Raw landmark scores, rule margins, and evidence validity are not semantic
    confidence.  Until held-out calibration exists, known claims explicitly
    carry ``UNCALIBRATED`` with a null value and unknown claims carry
    ``UNAVAILABLE``.
    """

    value: float | None
    status: ConfidenceStatus
    method_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConfidenceStatus):
            raise TypeError("confidence status must be a ConfidenceStatus")
        if self.status is ConfidenceStatus.CALIBRATED:
            raise ValueError(
                "calibrated confidence is unsupported by candidate schema v1; "
                "a calibration-provenance schema must be versioned first"
            )
        if self.value is not None:
            raise ValueError("non-calibrated confidence value must be null")
        if self.method_id is not None:
            raise ValueError("non-calibrated confidence method_id must be null")

    @classmethod
    def uncalibrated(cls) -> ConfidenceEstimate:
        return cls(value=None, status=ConfidenceStatus.UNCALIBRATED)

    @classmethod
    def unavailable(cls) -> ConfidenceEstimate:
        return cls(value=None, status=ConfidenceStatus.UNAVAILABLE)

    def to_mapping(self) -> dict[str, JsonScalar]:
        return {
            "value": float(self.value) if self.value is not None else None,
            "status": self.status.value,
            "method_id": self.method_id,
        }


@dataclass(frozen=True, slots=True)
class CandidateComponentContext:
    """One learned model or deterministic algorithm used by a candidate."""

    role: str
    model_id: str
    model_version: str
    asset_sha256: str | None = None
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.role, "component role")
        _validate_text(self.model_id, "component model_id")
        _validate_text(self.model_version, "component model_version")
        _validate_optional_sha256(self.asset_sha256, "component asset_sha256")
        _validate_optional_sha256(self.config_sha256, "component config_sha256")

    def to_mapping(self) -> dict[str, JsonScalar]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "asset_sha256": self.asset_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Traceable producer, configuration, calibration, and model context."""

    producer_id: str
    producer_version: str
    rule_config_schema_version: str
    rule_config_status: str
    rule_config_sha256: str
    trace_id: str
    data_status: str
    producer_bundle_sha256: str
    feature_bundle_sha256: str
    model_manifest_sha256: str
    provenance_verified: bool
    assets_verified: bool
    components: tuple[CandidateComponentContext, ...]
    input_artifact_sha256: str | None = None
    calibration_profile_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.producer_id, "producer_id")
        _validate_text(self.producer_version, "producer_version")
        _validate_text(
            self.rule_config_schema_version, "rule_config_schema_version"
        )
        _validate_text(self.rule_config_status, "rule_config_status")
        _validate_sha256(self.rule_config_sha256, "rule_config_sha256")
        _validate_text(self.trace_id, "trace_id")
        if self.data_status not in _DATA_STATUSES:
            raise ValueError("unsupported candidate data_status")
        _validate_sha256(self.producer_bundle_sha256, "producer_bundle_sha256")
        _validate_sha256(self.feature_bundle_sha256, "feature_bundle_sha256")
        _validate_sha256(self.model_manifest_sha256, "model_manifest_sha256")
        if not isinstance(self.provenance_verified, bool):
            raise TypeError("provenance_verified must be a boolean")
        if not isinstance(self.assets_verified, bool):
            raise TypeError("assets_verified must be a boolean")
        if self.assets_verified and not self.provenance_verified:
            raise ValueError("assets cannot be verified without provenance")
        _validate_optional_sha256(
            self.input_artifact_sha256, "input_artifact_sha256"
        )
        _validate_optional_sha256(
            self.calibration_profile_sha256, "calibration_profile_sha256"
        )
        if not isinstance(self.components, tuple):
            raise TypeError("components must be a tuple")
        if not self.components:
            raise ValueError("components must not be empty")
        if len(self.components) > 16:
            raise ValueError("components must remain compact")
        roles: set[str] = set()
        for component in self.components:
            if not isinstance(component, CandidateComponentContext):
                raise TypeError("components must contain CandidateComponentContext")
            if component.role in roles:
                raise ValueError("component roles must be unique")
            roles.add(component.role)

    def to_mapping(self) -> dict[str, Any]:
        components = sorted(
            self.components,
            key=lambda item: (item.role, item.model_id, item.model_version),
        )
        return {
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "rule_config_schema_version": self.rule_config_schema_version,
            "rule_config_status": self.rule_config_status,
            "rule_config_sha256": self.rule_config_sha256,
            "calibration_profile_sha256": self.calibration_profile_sha256,
            "trace_id": self.trace_id,
            "data_status": self.data_status,
            "producer_bundle_sha256": self.producer_bundle_sha256,
            "feature_bundle_sha256": self.feature_bundle_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "provenance_verified": self.provenance_verified,
            "assets_verified": self.assets_verified,
            "input_artifact_sha256": self.input_artifact_sha256,
            "components": [item.to_mapping() for item in components],
        }


@dataclass(frozen=True, slots=True)
class PartAEventCandidate:
    """One immutable Part A candidate transition.

    ``CLEAR`` means only that the feature condition completed its configured
    exit confirmation.  It is not acknowledgement, UI dismissal, controller
    receipt, or the final UnifiedEvent ``cleared`` state.
    """

    schema_version: str
    event_name: str
    transition: CandidateTransition
    semantic_state: SemanticState
    condition: ConditionState
    episode_id: str | None
    source_id: str
    sequence_id: int
    observed_at_ns: int
    confirmed_at_ns: int | None
    duration_ms: float
    confidence: ConfidenceEstimate
    reason_code: str | None
    supporting_evidence: tuple[tuple[str, JsonScalar], ...]
    context: CandidateContext

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported Part A candidate schema_version")
        if self.event_name not in PART_A_EVENT_NAMES:
            raise ValueError("event_name is not in the Part A vocabulary")
        if not isinstance(self.transition, CandidateTransition):
            raise TypeError("transition must be a CandidateTransition")
        if not isinstance(self.semantic_state, SemanticState):
            raise TypeError("semantic_state must be a SemanticState")
        if not isinstance(self.condition, ConditionState):
            raise TypeError("condition must be a ConditionState")
        _validate_text(self.source_id, "source_id")
        _validate_non_negative_int(self.sequence_id, "sequence_id")
        _validate_non_negative_int(self.observed_at_ns, "observed_at_ns")
        if self.confirmed_at_ns is not None:
            _validate_non_negative_int(self.confirmed_at_ns, "confirmed_at_ns")
            if self.confirmed_at_ns > self.observed_at_ns:
                raise ValueError("confirmed_at_ns must not exceed observed_at_ns")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or not math.isfinite(self.duration_ms)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be finite and non-negative")
        if not isinstance(self.confidence, ConfidenceEstimate):
            raise TypeError("confidence must be a ConfidenceEstimate")
        if self.reason_code is not None:
            _validate_text(self.reason_code, "reason_code")
        _validate_evidence(self.supporting_evidence)
        if not isinstance(self.context, CandidateContext):
            raise TypeError("context must be a CandidateContext")
        self._validate_transition_contract()

    def _validate_transition_contract(self) -> None:
        has_episode = self.episode_id is not None
        if has_episode:
            if not _EPISODE_ID.fullmatch(self.episode_id or ""):
                raise ValueError("episode_id must be a deterministic Part A ID")
            if self.confirmed_at_ns is None:
                raise ValueError("episode candidates require confirmed_at_ns")
            wall_duration_ms = (
                self.observed_at_ns - self.confirmed_at_ns
            ) / 1_000_000
            if self.duration_ms > wall_duration_ms + 1e-6:
                raise ValueError("episode duration cannot exceed elapsed wall time")
        elif self.confirmed_at_ns is not None:
            raise ValueError("confirmed_at_ns requires episode_id")

        if self.transition is CandidateTransition.START:
            if (
                self.semantic_state is not SemanticState.WARNING
                or self.condition is not ConditionState.TRUE
                or not has_episode
                or self.confirmed_at_ns != self.observed_at_ns
                or self.duration_ms != 0
            ):
                raise ValueError("START must begin one confirmed warning episode")
        elif self.transition is CandidateTransition.UPDATE:
            if self.semantic_state is not SemanticState.WARNING or not has_episode:
                raise ValueError("UPDATE must belong to a warning episode")
            if self.condition not in {ConditionState.TRUE, ConditionState.FALSE}:
                raise ValueError("UPDATE requires known condition evidence")
        elif self.transition is CandidateTransition.UNKNOWN:
            if (
                self.semantic_state is not SemanticState.UNKNOWN
                or self.condition is not ConditionState.UNKNOWN
                or not self.reason_code
            ):
                raise ValueError("UNKNOWN requires unknown evidence and a reason")
            if not has_episode and self.duration_ms != 0:
                raise ValueError("idle UNKNOWN duration must be zero")
        elif self.transition is CandidateTransition.AVAILABLE:
            if (
                self.semantic_state is not SemanticState.NORMAL
                or self.condition not in {ConditionState.TRUE, ConditionState.FALSE}
                or has_episode
                or self.duration_ms != 0
                or self.reason_code != "evidence_available"
            ):
                raise ValueError("AVAILABLE must restore an idle known-evidence lane")
        elif self.transition is CandidateTransition.CLEAR:
            if (
                self.semantic_state is not SemanticState.NORMAL
                or self.condition is not ConditionState.FALSE
                or not has_episode
                or self.reason_code != "condition_exit_confirmed"
            ):
                raise ValueError("CLEAR must be a confirmed condition exit")

        if self.transition is CandidateTransition.UNKNOWN:
            if self.confidence.status is not ConfidenceStatus.UNAVAILABLE:
                raise ValueError("UNKNOWN confidence must be unavailable")
        elif self.confidence.status is ConfidenceStatus.UNAVAILABLE:
            raise ValueError("known candidate confidence cannot be unavailable")

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible mapping."""

        evidence = {
            key: value for key, value in sorted(self.supporting_evidence)
        }
        return {
            "schema_version": self.schema_version,
            "event_name": self.event_name,
            "transition": self.transition.value,
            "semantic_state": self.semantic_state.value,
            "condition": self.condition.value,
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "sequence_id": self.sequence_id,
            "observed_at_ns": self.observed_at_ns,
            "confirmed_at_ns": self.confirmed_at_ns,
            "duration_ms": float(self.duration_ms),
            "confidence": self.confidence.to_mapping(),
            "reason_code": self.reason_code,
            "supporting_evidence": evidence,
            "context": self.context.to_mapping(),
        }

    def to_json(self) -> str:
        """Serialize identically across equivalent replay runs."""

        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def candidate_from_mapping(
    mapping: object,
    *,
    location: str = "candidate",
) -> PartAEventCandidate:
    """Strictly reconstruct one candidate from its versioned JSON mapping.

    This is intentionally an exact schema reader: producer-only additions must
    receive a schema migration before a consumer will accept them.
    """

    try:
        candidate_mapping = _require_object(mapping, location)
        _require_exact_fields(candidate_mapping, _CANDIDATE_FIELDS, location)

        confidence_mapping = _require_object(
            candidate_mapping["confidence"], f"{location}.confidence"
        )
        _require_exact_fields(
            confidence_mapping, _CONFIDENCE_FIELDS, f"{location}.confidence"
        )
        confidence = ConfidenceEstimate(
            value=_require_optional_number(
                confidence_mapping["value"], f"{location}.confidence.value"
            ),
            status=_parse_enum(
                ConfidenceStatus,
                confidence_mapping["status"],
                f"{location}.confidence.status",
            ),
            method_id=_require_optional_text_type(
                confidence_mapping["method_id"],
                f"{location}.confidence.method_id",
            ),
        )

        context_mapping = _require_object(
            candidate_mapping["context"], f"{location}.context"
        )
        _require_exact_fields(context_mapping, _CONTEXT_FIELDS, f"{location}.context")
        raw_components = context_mapping["components"]
        if not isinstance(raw_components, list):
            raise CandidateValidationError(
                f"{location}.context.components must be an array"
            )
        components: list[CandidateComponentContext] = []
        for index, raw_component in enumerate(raw_components):
            component_location = f"{location}.context.components[{index}]"
            component_mapping = _require_object(raw_component, component_location)
            _require_exact_fields(
                component_mapping, _COMPONENT_FIELDS, component_location
            )
            components.append(
                CandidateComponentContext(
                    role=_require_text_type(
                        component_mapping["role"], f"{component_location}.role"
                    ),
                    model_id=_require_text_type(
                        component_mapping["model_id"],
                        f"{component_location}.model_id",
                    ),
                    model_version=_require_text_type(
                        component_mapping["model_version"],
                        f"{component_location}.model_version",
                    ),
                    asset_sha256=_require_optional_text_type(
                        component_mapping["asset_sha256"],
                        f"{component_location}.asset_sha256",
                    ),
                    config_sha256=_require_optional_text_type(
                        component_mapping["config_sha256"],
                        f"{component_location}.config_sha256",
                    ),
                )
            )
        context = CandidateContext(
            producer_id=_require_text_type(
                context_mapping["producer_id"], f"{location}.context.producer_id"
            ),
            producer_version=_require_text_type(
                context_mapping["producer_version"],
                f"{location}.context.producer_version",
            ),
            rule_config_schema_version=_require_text_type(
                context_mapping["rule_config_schema_version"],
                f"{location}.context.rule_config_schema_version",
            ),
            rule_config_status=_require_text_type(
                context_mapping["rule_config_status"],
                f"{location}.context.rule_config_status",
            ),
            rule_config_sha256=_require_text_type(
                context_mapping["rule_config_sha256"],
                f"{location}.context.rule_config_sha256",
            ),
            calibration_profile_sha256=_require_optional_text_type(
                context_mapping["calibration_profile_sha256"],
                f"{location}.context.calibration_profile_sha256",
            ),
            trace_id=_require_text_type(
                context_mapping["trace_id"], f"{location}.context.trace_id"
            ),
            data_status=_require_text_type(
                context_mapping["data_status"], f"{location}.context.data_status"
            ),
            producer_bundle_sha256=_require_text_type(
                context_mapping["producer_bundle_sha256"],
                f"{location}.context.producer_bundle_sha256",
            ),
            feature_bundle_sha256=_require_text_type(
                context_mapping["feature_bundle_sha256"],
                f"{location}.context.feature_bundle_sha256",
            ),
            model_manifest_sha256=_require_text_type(
                context_mapping["model_manifest_sha256"],
                f"{location}.context.model_manifest_sha256",
            ),
            provenance_verified=_require_bool(
                context_mapping["provenance_verified"],
                f"{location}.context.provenance_verified",
            ),
            assets_verified=_require_bool(
                context_mapping["assets_verified"],
                f"{location}.context.assets_verified",
            ),
            input_artifact_sha256=_require_optional_text_type(
                context_mapping["input_artifact_sha256"],
                f"{location}.context.input_artifact_sha256",
            ),
            components=tuple(components),
        )

        evidence_mapping = _require_object(
            candidate_mapping["supporting_evidence"],
            f"{location}.supporting_evidence",
        )
        evidence: list[tuple[str, JsonScalar]] = []
        for key, value in evidence_mapping.items():
            if not _is_json_scalar(value):
                raise CandidateValidationError(
                    f"{location}.supporting_evidence.{key} must be a JSON scalar"
                )
            evidence.append((key, value))

        return PartAEventCandidate(
            schema_version=_require_text_type(
                candidate_mapping["schema_version"],
                f"{location}.schema_version",
            ),
            event_name=_require_text_type(
                candidate_mapping["event_name"], f"{location}.event_name"
            ),
            transition=_parse_enum(
                CandidateTransition,
                candidate_mapping["transition"],
                f"{location}.transition",
            ),
            semantic_state=_parse_enum(
                SemanticState,
                candidate_mapping["semantic_state"],
                f"{location}.semantic_state",
            ),
            condition=_parse_enum(
                ConditionState,
                candidate_mapping["condition"],
                f"{location}.condition",
            ),
            episode_id=_require_optional_text_type(
                candidate_mapping["episode_id"], f"{location}.episode_id"
            ),
            source_id=_require_text_type(
                candidate_mapping["source_id"], f"{location}.source_id"
            ),
            sequence_id=_require_int(
                candidate_mapping["sequence_id"], f"{location}.sequence_id"
            ),
            observed_at_ns=_require_int(
                candidate_mapping["observed_at_ns"],
                f"{location}.observed_at_ns",
            ),
            confirmed_at_ns=_require_optional_int(
                candidate_mapping["confirmed_at_ns"],
                f"{location}.confirmed_at_ns",
            ),
            duration_ms=_require_number(
                candidate_mapping["duration_ms"], f"{location}.duration_ms"
            ),
            confidence=confidence,
            reason_code=_require_optional_text_type(
                candidate_mapping["reason_code"], f"{location}.reason_code"
            ),
            supporting_evidence=tuple(evidence),
            context=context,
        )
    except CandidateValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"{location} is invalid: {exc}") from exc


@dataclass(slots=True)
class _CandidateConsumerLane:
    episode_id: str | None = None
    confirmed_at_ns: int | None = None
    last_duration_ms: float = 0.0
    idle_unknown: bool = False
    active_unknown: bool = False
    last_observed_at_ns: int | None = None


class CandidateJsonlFile:
    """Immutable, pre-hashed view of a bounded candidate JSONL artifact.

    The source path is copied once into an anonymous temporary file.  All
    parsing then uses those staged bytes, so replacement of the source path
    cannot alter a validated consumer view.
    """

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        limits: CandidateJsonlLimits | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.limits = limits or CandidateJsonlLimits()
        if not isinstance(self.limits, CandidateJsonlLimits):
            raise TypeError("limits must be CandidateJsonlLimits")
        if expected_sha256 is not None:
            try:
                _validate_sha256(expected_sha256, "expected_sha256")
            except ValueError as exc:
                raise CandidateValidationError(str(exc)) from exc
        self.expected_sha256 = expected_sha256
        if not self.path.is_file():
            raise CandidateValidationError(
                f"candidate JSONL file does not exist: {self.path}"
            )

        self._snapshot_handle: BinaryIO = tempfile.TemporaryFile(mode="w+b")
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with self.path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    total_bytes += len(chunk)
                    if total_bytes > self.limits.max_file_bytes:
                        raise CandidateValidationError(
                            "candidate JSONL exceeds configured total-byte limit"
                        )
                    digest.update(chunk)
                    self._snapshot_handle.write(chunk)
            self._snapshot_handle.flush()
            self._snapshot_handle.seek(0)
        except Exception:
            self._snapshot_handle.close()
            raise

        self.artifact_sha256 = digest.hexdigest()
        self.artifact_bytes = total_bytes
        if (
            self.expected_sha256 is not None
            and self.artifact_sha256 != self.expected_sha256
        ):
            self._snapshot_handle.close()
            raise CandidateValidationError(
                "candidate JSONL SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {self.artifact_sha256}"
            )

    def iter_candidates(self) -> Iterator[PartAEventCandidate]:
        """Yield candidates while enforcing whole-stream invariants."""

        digest = hashlib.sha256()
        record_count = 0
        source_id: str | None = None
        context_sha256: str | None = None
        last_sequence_id: int | None = None
        last_observed_at_ns: int | None = None
        events_in_sequence: set[str] = set()
        lanes = {name: _CandidateConsumerLane() for name in PART_A_EVENT_NAMES}
        self._snapshot_handle.seek(0)
        for line_number, raw_line in _iter_bounded_candidate_lines(
            self._snapshot_handle,
            max_line_bytes=self.limits.max_line_bytes,
        ):
            digest.update(raw_line)
            record_count += 1
            if record_count > self.limits.max_records:
                raise CandidateValidationError(
                    "candidate JSONL record limit exceeded"
                )
            mapping = _decode_candidate_json_line(
                raw_line,
                line_number=line_number,
                max_json_depth=self.limits.max_json_depth,
            )
            candidate = candidate_from_mapping(
                mapping, location=f"candidate line {line_number}"
            )

            if source_id is None:
                source_id = candidate.source_id
            elif candidate.source_id != source_id:
                raise CandidateValidationError(
                    f"candidate line {line_number} changed source_id"
                )
            fingerprint = _context_sha256(candidate.context)
            if context_sha256 is None:
                context_sha256 = fingerprint
            elif fingerprint != context_sha256:
                raise CandidateValidationError(
                    f"candidate line {line_number} changed immutable stream context"
                )

            if (
                last_sequence_id is not None
                and candidate.sequence_id < last_sequence_id
            ):
                raise CandidateValidationError(
                    f"candidate line {line_number} sequence_id regressed"
                )
            if (
                last_observed_at_ns is not None
                and candidate.observed_at_ns < last_observed_at_ns
            ):
                raise CandidateValidationError(
                    f"candidate line {line_number} observed_at_ns regressed"
                )
            if (
                last_sequence_id is not None
                and candidate.sequence_id > last_sequence_id
                and last_observed_at_ns is not None
                and candidate.observed_at_ns == last_observed_at_ns
            ):
                raise CandidateValidationError(
                    f"candidate line {line_number} timestamp did not increase "
                    "with sequence_id"
                )
            if candidate.sequence_id != last_sequence_id:
                events_in_sequence.clear()
            elif candidate.observed_at_ns != last_observed_at_ns:
                raise CandidateValidationError(
                    f"candidate line {line_number} changed timestamp within a sequence"
                )
            if candidate.event_name in events_in_sequence:
                raise CandidateValidationError(
                    f"candidate line {line_number} duplicates an event within a sequence"
                )
            _accept_candidate_lane_transition(
                lanes[candidate.event_name],
                candidate,
                line_number=line_number,
            )
            events_in_sequence.add(candidate.event_name)
            last_sequence_id = candidate.sequence_id
            last_observed_at_ns = candidate.observed_at_ns
            yield candidate

        if digest.hexdigest() != self.artifact_sha256:
            raise CandidateValidationError(
                "staged candidate JSONL bytes changed during parsing"
            )

    def validate(self) -> CandidateJsonlSummary:
        """Parse the complete staged artifact without retaining its records."""

        records = 0
        source_id: str | None = None
        first_context: CandidateContext | None = None
        first_sequence_id: int | None = None
        last_sequence_id: int | None = None
        first_observed_at_ns: int | None = None
        last_observed_at_ns: int | None = None
        for candidate in self.iter_candidates():
            if first_sequence_id is None:
                source_id = candidate.source_id
                first_context = candidate.context
                first_sequence_id = candidate.sequence_id
                first_observed_at_ns = candidate.observed_at_ns
            last_sequence_id = candidate.sequence_id
            last_observed_at_ns = candidate.observed_at_ns
            records += 1
        return CandidateJsonlSummary(
            artifact_sha256=self.artifact_sha256,
            artifact_bytes=self.artifact_bytes,
            records=records,
            source_id=source_id,
            context_sha256=(
                candidate_context_sha256(first_context)
                if first_context is not None
                else None
            ),
            data_status=(
                first_context.data_status if first_context is not None else None
            ),
            input_artifact_sha256=(
                first_context.input_artifact_sha256
                if first_context is not None
                else None
            ),
            first_sequence_id=first_sequence_id,
            last_sequence_id=last_sequence_id,
            first_observed_at_ns=first_observed_at_ns,
            last_observed_at_ns=last_observed_at_ns,
        )

    def close(self) -> None:
        self._snapshot_handle.close()

    def __enter__(self) -> CandidateJsonlFile:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(slots=True)
class _LaneState:
    episode_id: str | None = None
    confirmed_at_ns: int | None = None
    last_semantic_state: SemanticState | None = None
    last_phase: TemporalPhase | None = None
    last_reason: str | None = None
    last_emitted_at_ns: int | None = None
    last_duration_ms: float = 0.0


class PartACandidateEmitter:
    """Convert complete rule snapshots to independent bounded candidates."""

    def __init__(self, *, update_interval_ms: int = 1_000) -> None:
        if (
            isinstance(update_interval_ms, bool)
            or not isinstance(update_interval_ms, int)
        ):
            raise TypeError("update_interval_ms must be an integer")
        if update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")
        self.update_interval_ms = update_interval_ms
        self._update_interval_ns = update_interval_ms * 1_000_000
        self._lanes = {name: _LaneState() for name in PART_A_EVENT_NAMES}
        self._source_id: str | None = None
        self._last_sequence_id: int | None = None
        self._last_timestamp_ns: int | None = None
        self._session_context_sha256: str | None = None

    def emit(
        self,
        snapshot: ErgonomicsRuleSnapshot,
        *,
        sequence_id: int,
        context: CandidateContext,
    ) -> tuple[PartAEventCandidate, ...]:
        """Consume one complete eight-lane snapshot and emit transition deltas."""

        evaluations = self._validate_snapshot(snapshot, sequence_id, context)
        self._accept_context(context)
        self._accept_position(snapshot.source_id, sequence_id, snapshot.captured_at_ns)
        emitted: list[PartAEventCandidate] = []
        for event_name in PART_A_EVENT_NAMES:
            evaluation = evaluations[event_name]
            candidate = self._advance_lane(
                evaluation,
                source_id=snapshot.source_id,
                sequence_id=sequence_id,
                context=context,
            )
            if candidate is not None:
                emitted.append(candidate)
        return tuple(emitted)

    def mark_unavailable(
        self,
        *,
        source_id: str,
        sequence_id: int,
        observed_at_ns: int,
        reason_code: str,
        context: CandidateContext,
    ) -> tuple[PartAEventCandidate, ...]:
        """Emit bounded UNKNOWN health observations without clearing episodes."""

        if not isinstance(context, CandidateContext):
            raise TypeError("context must be a CandidateContext")
        _validate_text(reason_code, "reason_code")
        self._validate_context(context)
        self._validate_position(source_id, sequence_id, observed_at_ns)
        self._accept_context(context)
        self._accept_position(source_id, sequence_id, observed_at_ns)
        emitted: list[PartAEventCandidate] = []
        for event_name in PART_A_EVENT_NAMES:
            lane = self._lanes[event_name]
            should_emit = (
                lane.last_semantic_state is not SemanticState.UNKNOWN
                or lane.last_reason != reason_code
                or self._heartbeat_due(lane, observed_at_ns)
            )
            if should_emit:
                emitted.append(
                    self._build_candidate(
                        event_name=event_name,
                        transition=CandidateTransition.UNKNOWN,
                        semantic_state=SemanticState.UNKNOWN,
                        condition=ConditionState.UNKNOWN,
                        lane=lane,
                        source_id=source_id,
                        sequence_id=sequence_id,
                        observed_at_ns=observed_at_ns,
                        duration_ms=(
                            lane.last_duration_ms if lane.episode_id is not None else 0.0
                        ),
                        reason_code=reason_code,
                        evidence=(("evidence_continuous", False),),
                        context=context,
                    )
                )
                lane.last_emitted_at_ns = observed_at_ns
            lane.last_semantic_state = SemanticState.UNKNOWN
            lane.last_reason = reason_code
        return tuple(emitted)

    def _advance_lane(
        self,
        evaluation: RuleEvaluation,
        *,
        source_id: str,
        sequence_id: int,
        context: CandidateContext,
    ) -> PartAEventCandidate | None:
        lane = self._lanes[evaluation.event_name]
        timestamp_ns = evaluation.observed_at_ns
        transition: CandidateTransition | None = None
        reason_code = evaluation.reason

        if evaluation.semantic_state is SemanticState.WARNING:
            if lane.episode_id is None:
                lane.confirmed_at_ns = timestamp_ns
                lane.episode_id = _episode_id(
                    source_id=source_id,
                    event_name=evaluation.event_name,
                    sequence_id=sequence_id,
                    confirmed_at_ns=timestamp_ns,
                    config_sha256=context.rule_config_sha256,
                    context_sha256=self._required_context_sha256(),
                )
                transition = CandidateTransition.START
            elif (
                lane.last_semantic_state is SemanticState.UNKNOWN
                or lane.last_phase is not evaluation.phase
                or self._heartbeat_due(lane, timestamp_ns)
            ):
                transition = CandidateTransition.UPDATE
        elif evaluation.semantic_state is SemanticState.UNKNOWN:
            reason_code = reason_code or "evidence_unavailable"
            if (
                lane.last_semantic_state is not SemanticState.UNKNOWN
                or lane.last_reason != reason_code
                or self._heartbeat_due(lane, timestamp_ns)
            ):
                transition = CandidateTransition.UNKNOWN
        elif lane.episode_id is not None:
            if (
                evaluation.condition is not ConditionState.FALSE
                or evaluation.phase not in {TemporalPhase.IDLE, TemporalPhase.COOLDOWN}
            ):
                raise ValueError(
                    f"active {evaluation.event_name} episode disappeared without "
                    "a confirmed condition exit"
                )
            if evaluation.active_duration_ms < lane.last_duration_ms:
                raise ValueError("clear snapshot lost the terminal active duration")
            transition = CandidateTransition.CLEAR
            reason_code = "condition_exit_confirmed"
        elif lane.last_semantic_state is SemanticState.UNKNOWN:
            transition = CandidateTransition.AVAILABLE
            reason_code = "evidence_available"

        candidate: PartAEventCandidate | None = None
        if transition is not None:
            duration_ms = (
                0.0
                if transition is CandidateTransition.START
                else evaluation.active_duration_ms
                if lane.episode_id is not None
                else 0.0
            )
            candidate = self._build_candidate(
                event_name=evaluation.event_name,
                transition=transition,
                semantic_state=evaluation.semantic_state,
                condition=evaluation.condition,
                lane=lane,
                source_id=source_id,
                sequence_id=sequence_id,
                observed_at_ns=timestamp_ns,
                duration_ms=duration_ms,
                reason_code=reason_code,
                evidence=_candidate_evidence(evaluation),
                context=context,
            )
            lane.last_emitted_at_ns = timestamp_ns

        lane.last_semantic_state = evaluation.semantic_state
        lane.last_phase = evaluation.phase
        lane.last_reason = reason_code
        lane.last_duration_ms = evaluation.active_duration_ms
        if transition is CandidateTransition.CLEAR:
            lane.episode_id = None
            lane.confirmed_at_ns = None
            lane.last_duration_ms = 0.0
        return candidate

    @staticmethod
    def _build_candidate(
        *,
        event_name: str,
        transition: CandidateTransition,
        semantic_state: SemanticState,
        condition: ConditionState,
        lane: _LaneState,
        source_id: str,
        sequence_id: int,
        observed_at_ns: int,
        duration_ms: float,
        reason_code: str | None,
        evidence: tuple[tuple[str, JsonScalar], ...],
        context: CandidateContext,
    ) -> PartAEventCandidate:
        confidence = (
            ConfidenceEstimate.unavailable()
            if transition is CandidateTransition.UNKNOWN
            else ConfidenceEstimate.uncalibrated()
        )
        return PartAEventCandidate(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            event_name=event_name,
            transition=transition,
            semantic_state=semantic_state,
            condition=condition,
            episode_id=lane.episode_id,
            source_id=source_id,
            sequence_id=sequence_id,
            observed_at_ns=observed_at_ns,
            confirmed_at_ns=lane.confirmed_at_ns,
            duration_ms=float(duration_ms),
            confidence=confidence,
            reason_code=reason_code,
            supporting_evidence=evidence,
            context=context,
        )

    def _validate_snapshot(
        self,
        snapshot: ErgonomicsRuleSnapshot,
        sequence_id: int,
        context: CandidateContext,
    ) -> dict[str, RuleEvaluation]:
        if not isinstance(snapshot, ErgonomicsRuleSnapshot):
            raise TypeError("snapshot must be an ErgonomicsRuleSnapshot")
        if not isinstance(context, CandidateContext):
            raise TypeError("context must be a CandidateContext")
        self._validate_context(context)
        if snapshot.config_schema_version != context.rule_config_schema_version:
            raise ValueError("snapshot and candidate config schema versions differ")
        if snapshot.config_status != context.rule_config_status:
            raise ValueError("snapshot and candidate config statuses differ")
        self._validate_position(snapshot.source_id, sequence_id, snapshot.captured_at_ns)
        if not isinstance(snapshot.evaluations, tuple):
            raise TypeError("snapshot evaluations must be a tuple")
        if len(snapshot.evaluations) != len(PART_A_EVENT_NAMES):
            raise ValueError("snapshot must contain every Part A lane exactly once")
        evaluations: dict[str, RuleEvaluation] = {}
        for evaluation in snapshot.evaluations:
            self._validate_evaluation(evaluation, snapshot.captured_at_ns)
            if evaluation.event_name in evaluations:
                raise ValueError("snapshot contains a duplicate Part A lane")
            evaluations[evaluation.event_name] = evaluation
        if set(evaluations) != set(PART_A_EVENT_NAMES):
            raise ValueError("snapshot Part A lane vocabulary does not match schema")
        for event_name in PART_A_EVENT_NAMES:
            self._validate_lane_transition(evaluations[event_name])
        return evaluations

    def _validate_context(self, context: CandidateContext) -> None:
        fingerprint = _context_sha256(context)
        if (
            self._session_context_sha256 is not None
            and fingerprint != self._session_context_sha256
        ):
            raise ValueError("candidate session context changed within one stream")

    def _accept_context(self, context: CandidateContext) -> None:
        if self._session_context_sha256 is None:
            self._session_context_sha256 = _context_sha256(context)

    def _required_context_sha256(self) -> str:
        if self._session_context_sha256 is None:
            raise RuntimeError("candidate session context was not initialized")
        return self._session_context_sha256

    def _validate_lane_transition(self, evaluation: RuleEvaluation) -> None:
        lane = self._lanes[evaluation.event_name]
        if evaluation.semantic_state is SemanticState.WARNING:
            expected_condition = (
                ConditionState.TRUE
                if evaluation.phase is TemporalPhase.ACTIVE
                else ConditionState.FALSE
            )
            if evaluation.condition is not expected_condition:
                raise ValueError("warning condition does not match its temporal phase")
            if lane.episode_id is None and evaluation.phase is not TemporalPhase.ACTIVE:
                raise ValueError("candidate stream cannot begin inside an exiting episode")
            if lane.episode_id is None and evaluation.active_duration_ms != 0:
                raise ValueError("new candidate episode must begin at zero duration")
            if (
                lane.episode_id is not None
                and evaluation.active_duration_ms < lane.last_duration_ms
            ):
                raise ValueError("active candidate duration must not decrease")
            return
        if evaluation.semantic_state is SemanticState.UNKNOWN:
            if lane.episode_id is None and evaluation.active_duration_ms != 0:
                raise ValueError("idle unknown duration must be zero")
            if lane.episode_id is not None:
                if evaluation.phase is not TemporalPhase.ACTIVE:
                    raise ValueError("active unknown must retain its active temporal phase")
                if evaluation.active_duration_ms < lane.last_duration_ms:
                    raise ValueError("active candidate duration must not decrease")
            return
        if evaluation.semantic_state is not SemanticState.NORMAL:
            return
        if lane.episode_id is None:
            return
        if (
            evaluation.condition is not ConditionState.FALSE
            or evaluation.phase not in {TemporalPhase.IDLE, TemporalPhase.COOLDOWN}
        ):
            raise ValueError(
                f"active {evaluation.event_name} episode disappeared without "
                "a confirmed condition exit"
            )
        if evaluation.active_duration_ms < lane.last_duration_ms:
            raise ValueError("clear snapshot lost the terminal active duration")

    @staticmethod
    def _validate_evaluation(evaluation: RuleEvaluation, timestamp_ns: int) -> None:
        if not isinstance(evaluation, RuleEvaluation):
            raise TypeError("evaluations must contain RuleEvaluation")
        if evaluation.event_name not in PART_A_EVENT_NAMES:
            raise ValueError("evaluation event_name is outside Part A vocabulary")
        if evaluation.observed_at_ns != timestamp_ns:
            raise ValueError("evaluation timestamp must match its rule snapshot")
        if not isinstance(evaluation.condition, ConditionState):
            raise TypeError("evaluation condition must be a ConditionState")
        if not isinstance(evaluation.semantic_state, SemanticState):
            raise TypeError("evaluation semantic_state must be a SemanticState")
        if not isinstance(evaluation.phase, TemporalPhase):
            raise TypeError("evaluation phase must be a TemporalPhase")
        for label, value in (
            ("evidence_elapsed_ms", evaluation.evidence_elapsed_ms),
            ("active_duration_ms", evaluation.active_duration_ms),
            ("cooldown_remaining_ms", evaluation.cooldown_remaining_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"evaluation {label} must be finite and non-negative")
        if evaluation.reason is not None:
            _validate_text(evaluation.reason, "evaluation reason")
        _validate_evidence(evaluation.evidence)
        if (
            evaluation.condition is ConditionState.UNKNOWN
        ) != (evaluation.semantic_state is SemanticState.UNKNOWN):
            raise ValueError("unknown condition and semantic state must agree")
        if evaluation.semantic_state is SemanticState.WARNING and evaluation.phase not in {
            TemporalPhase.ACTIVE,
            TemporalPhase.EXITING,
        }:
            raise ValueError("warning evaluation must be active or exiting")
        if evaluation.semantic_state is SemanticState.NORMAL and evaluation.phase not in {
            TemporalPhase.IDLE,
            TemporalPhase.ENTERING,
            TemporalPhase.COOLDOWN,
        }:
            raise ValueError("normal evaluation has an invalid temporal phase")

    def _validate_position(
        self, source_id: str, sequence_id: int, timestamp_ns: int
    ) -> None:
        _validate_text(source_id, "source_id")
        _validate_non_negative_int(sequence_id, "sequence_id")
        _validate_non_negative_int(timestamp_ns, "observed_at_ns")
        if self._source_id is not None and source_id != self._source_id:
            raise ValueError("candidate emitter cannot switch source")
        if self._last_sequence_id is not None and sequence_id <= self._last_sequence_id:
            raise ValueError("candidate sequence IDs must increase strictly")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("candidate timestamps must increase strictly")

    def _accept_position(
        self, source_id: str, sequence_id: int, timestamp_ns: int
    ) -> None:
        self._source_id = source_id
        self._last_sequence_id = sequence_id
        self._last_timestamp_ns = timestamp_ns

    def _heartbeat_due(self, lane: _LaneState, timestamp_ns: int) -> bool:
        return (
            lane.last_emitted_at_ns is None
            or timestamp_ns - lane.last_emitted_at_ns >= self._update_interval_ns
        )


def _candidate_evidence(
    evaluation: RuleEvaluation,
) -> tuple[tuple[str, JsonScalar], ...]:
    evidence = dict(evaluation.evidence)
    evidence.update(
        {
            "temporal_phase": evaluation.phase.value,
            "entry_or_exit_evidence_ms": float(evaluation.evidence_elapsed_ms),
            "cooldown_remaining_ms": float(evaluation.cooldown_remaining_ms),
        }
    )
    return tuple(sorted(evidence.items()))


def _episode_id(
    *,
    source_id: str,
    event_name: str,
    sequence_id: int,
    confirmed_at_ns: int,
    config_sha256: str,
    context_sha256: str,
) -> str:
    payload = "\x1f".join(
        (
            CANDIDATE_SCHEMA_VERSION,
            source_id,
            event_name,
            str(sequence_id),
            str(confirmed_at_ns),
            config_sha256,
            context_sha256,
        )
    ).encode("utf-8")
    return f"parta-{hashlib.sha256(payload).hexdigest()[:32]}"


def _context_sha256(context: CandidateContext) -> str:
    """Fingerprint immutable run context while allowing per-record trace IDs."""

    mapping = context.to_mapping()
    mapping.pop("trace_id", None)
    payload = json.dumps(
        mapping,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_context_sha256(context: CandidateContext) -> str:
    """Return the immutable consumer fingerprint for one candidate context."""

    if not isinstance(context, CandidateContext):
        raise TypeError("context must be a CandidateContext")
    return _context_sha256(context)


def _validate_evidence(
    evidence: tuple[tuple[str, JsonScalar], ...],
) -> None:
    if not isinstance(evidence, tuple):
        raise TypeError("supporting evidence must be a tuple")
    if len(evidence) > _MAX_EVIDENCE_ITEMS:
        raise ValueError("supporting evidence must remain compact")
    keys: set[str] = set()
    for item in evidence:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("supporting evidence items must be key/value tuples")
        key, value = item
        _validate_text(key, "supporting evidence key", maximum=128)
        if key in keys:
            raise ValueError("supporting evidence keys must be unique")
        normalized_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).casefold()
        key_tokens = {
            token for token in re.split(r"[^a-z0-9]+", normalized_key) if token
        }
        if (
            key_tokens & _FORBIDDEN_EVIDENCE_TOKENS
            or _FORBIDDEN_EVIDENCE_PATTERN.search(key)
        ):
            raise ValueError("supporting evidence cannot contain control fields")
        if key not in _ALLOWED_EVIDENCE_KEYS:
            raise ValueError(
                "supporting evidence key is not in candidate schema v1 allowlist"
            )
        keys.add(key)
        if value is None or isinstance(value, (str, bool, int)):
            if isinstance(value, str) and len(value) > _MAX_EVIDENCE_TEXT:
                raise ValueError("supporting evidence text is too long")
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        raise TypeError("supporting evidence values must be finite JSON scalars")


def _validate_text(
    value: str | None, label: str, *, maximum: int = 256
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > maximum:
        raise ValueError(f"{label} is too long")


def _validate_non_negative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _validate_optional_sha256(value: str | None, label: str) -> None:
    if value is not None:
        _validate_sha256(value, label)


def _accept_candidate_lane_transition(
    lane: _CandidateConsumerLane,
    candidate: PartAEventCandidate,
    *,
    line_number: int,
) -> None:
    location = f"candidate line {line_number} ({candidate.event_name})"
    transition = candidate.transition

    if transition is CandidateTransition.START:
        if lane.episode_id is not None:
            raise CandidateValidationError(
                f"{location} has duplicate START while an episode is active"
            )
        lane.episode_id = candidate.episode_id
        lane.confirmed_at_ns = candidate.confirmed_at_ns
        lane.last_duration_ms = candidate.duration_ms
        lane.idle_unknown = False
        lane.active_unknown = False
        lane.last_observed_at_ns = candidate.observed_at_ns
        return

    if transition is CandidateTransition.AVAILABLE:
        if lane.episode_id is not None:
            raise CandidateValidationError(
                f"{location} cannot make an active episode AVAILABLE"
            )
        if not lane.idle_unknown:
            raise CandidateValidationError(
                f"{location} has AVAILABLE without a preceding idle UNKNOWN"
            )
        lane.idle_unknown = False
        lane.active_unknown = False
        lane.last_observed_at_ns = None
        return

    if transition is CandidateTransition.UNKNOWN and candidate.episode_id is None:
        if lane.episode_id is not None:
            raise CandidateValidationError(
                f"{location} dropped the active episode on UNKNOWN"
            )
        lane.idle_unknown = True
        lane.active_unknown = False
        lane.last_observed_at_ns = None
        return

    if lane.episode_id is None:
        raise CandidateValidationError(
            f"{location} has {transition.value.upper()} before START"
        )
    if candidate.episode_id != lane.episode_id:
        raise CandidateValidationError(
            f"{location} changed the active episode_id"
        )
    if candidate.confirmed_at_ns != lane.confirmed_at_ns:
        raise CandidateValidationError(
            f"{location} changed the active confirmed_at_ns"
        )
    if candidate.duration_ms < lane.last_duration_ms:
        raise CandidateValidationError(
            f"{location} active duration regressed"
        )
    if lane.active_unknown and not math.isclose(
            candidate.duration_ms,
            lane.last_duration_ms,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
        raise CandidateValidationError(
            f"{location} duration changed during or immediately after active UNKNOWN"
        )
    if lane.last_observed_at_ns is None:
        raise CandidateValidationError(f"{location} lost the prior episode timestamp")
    maximum_increment_ms = (
        candidate.observed_at_ns - lane.last_observed_at_ns
    ) / 1_000_000
    if candidate.duration_ms - lane.last_duration_ms > maximum_increment_ms + 1e-6:
        raise CandidateValidationError(
            f"{location} duration advanced faster than wall time"
        )

    lane.last_duration_ms = candidate.duration_ms
    lane.idle_unknown = False
    lane.active_unknown = transition is CandidateTransition.UNKNOWN
    lane.last_observed_at_ns = candidate.observed_at_ns
    if transition is CandidateTransition.CLEAR:
        lane.episode_id = None
        lane.confirmed_at_ns = None
        lane.last_duration_ms = 0.0
        lane.active_unknown = False
        lane.last_observed_at_ns = None


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateValidationError(f"{location} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CandidateValidationError(f"{location} keys must be strings")
    return value


def _require_exact_fields(
    mapping: dict[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise CandidateValidationError(f"{location} has {' and '.join(details)}")


def _require_text_type(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise CandidateValidationError(f"{location} must be text")
    return value


def _require_optional_text_type(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _require_text_type(value, location)


def _require_bool(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateValidationError(f"{location} must be a boolean")
    return value


def _require_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateValidationError(f"{location} must be an integer")
    return value


def _require_optional_int(value: object, location: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, location)


def _require_number(value: object, location: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateValidationError(f"{location} must be a number")
    if not math.isfinite(value):
        raise CandidateValidationError(f"{location} must be finite")
    return value


def _require_optional_number(value: object, location: str) -> float | int | None:
    if value is None:
        return None
    return _require_number(value, location)


def _parse_enum(enum_type: type[StrEnum], value: object, location: str) -> Any:
    text = _require_text_type(value, location)
    try:
        return enum_type(text)
    except ValueError as exc:
        raise CandidateValidationError(
            f"{location} has unsupported value {text!r}"
        ) from exc


def _is_json_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _iter_bounded_candidate_lines(
    handle: BinaryIO, *, max_line_bytes: int
) -> Iterator[tuple[int, bytes]]:
    line_number = 0
    while True:
        raw_line = handle.readline(max_line_bytes + 1)
        if not raw_line:
            return
        line_number += 1
        if len(raw_line) > max_line_bytes:
            raise CandidateValidationError(
                f"candidate line {line_number} exceeds configured byte limit"
            )
        yield line_number, raw_line


def _decode_candidate_json_line(
    raw_line: bytes,
    *,
    line_number: int,
    max_json_depth: int,
) -> dict[str, Any]:
    _validate_json_depth(
        raw_line, line_number=line_number, max_json_depth=max_json_depth
    )
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CandidateValidationError(
            f"candidate line {line_number} is not valid UTF-8"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except CandidateValidationError as exc:
        raise CandidateValidationError(
            f"candidate line {line_number} is invalid: {exc}"
        ) from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CandidateValidationError(
            f"candidate line {line_number} is not valid bounded JSON"
        ) from exc
    return _require_object(value, f"candidate line {line_number}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise CandidateValidationError(f"duplicate JSON key {key!r}")
        mapping[key] = value
    return mapping


def _reject_non_finite_json_constant(value: str) -> None:
    raise CandidateValidationError(f"non-finite JSON constant {value!r}")


def _validate_json_depth(
    raw_line: bytes, *, line_number: int, max_json_depth: int
) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in raw_line:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):  # { [
            depth += 1
            if depth > max_json_depth:
                raise CandidateValidationError(
                    f"candidate line {line_number} exceeds JSON nesting limit"
                )
        elif byte in (0x7D, 0x5D):  # } ]
            depth -= 1
            if depth < 0:
                raise CandidateValidationError(
                    f"candidate line {line_number} has unbalanced JSON nesting"
                )
