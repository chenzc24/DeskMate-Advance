from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from deskmate_advance.temporal.ergonomics import (
    CANDIDATE_SCHEMA_VERSION,
    PART_A_EVENT_NAMES,
    CandidateComponentContext,
    CandidateContext,
    CandidateTransition,
    ConditionState,
    ConfidenceEstimate,
    ConfidenceStatus,
    ErgonomicsRuleEngine,
    ErgonomicsRuleSnapshot,
    PartACandidateEmitter,
    SemanticState,
    TemporalPhase,
)
from deskmate_advance.temporal.ergonomics.rules import RuleEvaluation
from deskmate_advance.temporal.ergonomics.candidates import (
    CandidateJsonlFile,
    CandidateJsonlLimits,
    CandidateValidationError,
    candidate_from_mapping,
)


_CONFIG_HASH = "a" * 64
_ASSET_HASH = "b" * 64


def _ms(value: int) -> int:
    return value * 1_000_000


def _context(trace_id: str = "trace-1") -> CandidateContext:
    return CandidateContext(
        producer_id="part_a_ergonomics_rules",
        producer_version="0.1.0",
        rule_config_schema_version="1.0",
        rule_config_status="development_defaults_not_acceptance_thresholds",
        rule_config_sha256=_CONFIG_HASH,
        calibration_profile_sha256="c" * 64,
        trace_id=trace_id,
        data_status="synthetic_contract_test",
        producer_bundle_sha256="e" * 64,
        feature_bundle_sha256="f" * 64,
        model_manifest_sha256="1" * 64,
        provenance_verified=True,
        assets_verified=True,
        input_artifact_sha256="2" * 64,
        components=(
            CandidateComponentContext(
                role="rule",
                model_id="ergonomics_temporal_rules",
                model_version="0.1.0",
                config_sha256=_CONFIG_HASH,
            ),
            CandidateComponentContext(
                role="pose",
                model_id="mediapipe_pose_landmarker_full",
                model_version="0.10.35",
                asset_sha256=_ASSET_HASH,
                config_sha256="d" * 64,
            ),
        ),
    )


def _evaluation(
    event_name: str,
    timestamp_ns: int,
    *,
    condition: ConditionState = ConditionState.FALSE,
    semantic_state: SemanticState = SemanticState.NORMAL,
    phase: TemporalPhase = TemporalPhase.IDLE,
    active_duration_ms: float = 0.0,
    reason: str | None = None,
    evidence: tuple[tuple[str, str | float | int | bool | None], ...] = (
        ("p90_luminance", 2.0),
        ("mean_luminance", 1.0),
    ),
) -> RuleEvaluation:
    return RuleEvaluation(
        event_name=event_name,
        condition=condition,
        semantic_state=semantic_state,
        phase=phase,
        observed_at_ns=timestamp_ns,
        evidence_elapsed_ms=0.0,
        active_duration_ms=active_duration_ms,
        cooldown_remaining_ms=0.0,
        reason=reason,
        evidence=evidence,
    )


def _snapshot(
    timestamp_ns: int,
    *,
    overrides: dict[str, RuleEvaluation] | None = None,
    source_id: str = "camera-0",
) -> ErgonomicsRuleSnapshot:
    overrides = overrides or {}
    evaluations = tuple(
        overrides.get(event_name, _evaluation(event_name, timestamp_ns))
        for event_name in PART_A_EVENT_NAMES
    )
    return ErgonomicsRuleSnapshot(
        source_id=source_id,
        captured_at_ns=timestamp_ns,
        config_schema_version="1.0",
        config_status="development_defaults_not_acceptance_thresholds",
        evaluations=evaluations,
        blink_rate=None,
    )


def _warning(
    event_name: str,
    timestamp_ns: int,
    *,
    duration_ms: float,
    phase: TemporalPhase = TemporalPhase.ACTIVE,
) -> RuleEvaluation:
    return _evaluation(
        event_name,
        timestamp_ns,
        condition=(
            ConditionState.TRUE
            if phase is TemporalPhase.ACTIVE
            else ConditionState.FALSE
        ),
        semantic_state=SemanticState.WARNING,
        phase=phase,
        active_duration_ms=duration_ms,
    )


def _unknown(
    event_name: str,
    timestamp_ns: int,
    *,
    duration_ms: float = 0.0,
    reason: str = "evidence_unavailable",
    phase: TemporalPhase = TemporalPhase.IDLE,
) -> RuleEvaluation:
    return _evaluation(
        event_name,
        timestamp_ns,
        condition=ConditionState.UNKNOWN,
        semantic_state=SemanticState.UNKNOWN,
        phase=phase,
        active_duration_ms=duration_ms,
        reason=reason,
    )


def _clear(
    event_name: str,
    timestamp_ns: int,
    *,
    duration_ms: float,
) -> RuleEvaluation:
    return _evaluation(
        event_name,
        timestamp_ns,
        condition=ConditionState.FALSE,
        semantic_state=SemanticState.NORMAL,
        phase=TemporalPhase.COOLDOWN,
        active_duration_ms=duration_ms,
    )


def test_candidate_vocabulary_matches_rule_engine_lanes() -> None:
    assert PART_A_EVENT_NAMES == ErgonomicsRuleEngine.EVENT_NAMES


def test_start_mapping_is_deterministic_traceable_and_not_a_unified_event() -> None:
    first = PartACandidateEmitter()
    second = PartACandidateEmitter()
    timestamp_ns = _ms(100)
    snapshot = _snapshot(
        timestamp_ns,
        overrides={
            "static_too_long": _warning(
                "static_too_long", timestamp_ns, duration_ms=0.0
            )
        },
    )

    candidate = first.emit(snapshot, sequence_id=7, context=_context())[0]
    replayed = second.emit(snapshot, sequence_id=7, context=_context())[0]
    mapping = candidate.to_mapping()

    assert candidate.schema_version == CANDIDATE_SCHEMA_VERSION
    assert candidate.transition is CandidateTransition.START
    assert candidate.confirmed_at_ns == timestamp_ns
    assert candidate.duration_ms == 0
    assert candidate.episode_id == replayed.episode_id
    assert candidate.to_json() == replayed.to_json()
    assert json.loads(candidate.to_json()) == mapping
    assert mapping["confidence"] == {
        "value": None,
        "status": "uncalibrated",
        "method_id": None,
    }
    assert mapping["context"]["rule_config_sha256"] == _CONFIG_HASH
    assert mapping["context"]["components"][0]["role"] == "pose"
    assert list(mapping["supporting_evidence"]) == sorted(
        mapping["supporting_evidence"]
    )
    serialized = candidate.to_json()
    for forbidden in (
        "suggested_action",
        "motor_speed",
        "servo_angle",
        "arduino_command",
        "model_level",
    ):
        assert forbidden not in serialized


def test_start_bounded_update_unknown_recovery_exit_and_terminal_clear() -> None:
    emitter = PartACandidateEmitter(update_interval_ms=1_000)
    event_name = "screen_too_close"

    assert emitter.emit(_snapshot(0), sequence_id=0, context=_context("t0")) == ()

    started_ns = _ms(100)
    started = emitter.emit(
        _snapshot(
            started_ns,
            overrides={event_name: _warning(event_name, started_ns, duration_ms=0)},
        ),
        sequence_id=1,
        context=_context("t1"),
    )
    assert [item.transition for item in started] == [CandidateTransition.START]
    episode_id = started[0].episode_id

    quiet = emitter.emit(
        _snapshot(
            _ms(200),
            overrides={event_name: _warning(event_name, _ms(200), duration_ms=100)},
        ),
        sequence_id=2,
        context=_context("t2"),
    )
    assert quiet == ()

    heartbeat = emitter.emit(
        _snapshot(
            _ms(1_100),
            overrides={event_name: _warning(event_name, _ms(1_100), duration_ms=1_000)},
        ),
        sequence_id=3,
        context=_context("t3"),
    )
    assert [item.transition for item in heartbeat] == [CandidateTransition.UPDATE]

    unavailable = emitter.emit(
        _snapshot(
            _ms(1_200),
            overrides={
                event_name: _unknown(
                    event_name,
                    _ms(1_200),
                    duration_ms=1_000,
                    reason="face_geometry_stale",
                    phase=TemporalPhase.ACTIVE,
                )
            },
        ),
        sequence_id=4,
        context=_context("t4"),
    )
    assert [item.transition for item in unavailable] == [CandidateTransition.UNKNOWN]
    assert unavailable[0].episode_id == episode_id
    assert unavailable[0].duration_ms == pytest.approx(1_000)
    assert unavailable[0].confidence.status is ConfidenceStatus.UNAVAILABLE
    assert unavailable[0].confidence.value is None

    repeated_unknown = emitter.emit(
        _snapshot(
            _ms(1_300),
            overrides={
                event_name: _unknown(
                    event_name,
                    _ms(1_300),
                    duration_ms=1_000,
                    reason="face_geometry_stale",
                    phase=TemporalPhase.ACTIVE,
                )
            },
        ),
        sequence_id=5,
        context=_context("t5"),
    )
    assert repeated_unknown == ()

    recovered = emitter.emit(
        _snapshot(
            _ms(1_400),
            overrides={event_name: _warning(event_name, _ms(1_400), duration_ms=1_000)},
        ),
        sequence_id=6,
        context=_context("t6"),
    )
    assert [item.transition for item in recovered] == [CandidateTransition.UPDATE]
    assert recovered[0].episode_id == episode_id

    exiting = emitter.emit(
        _snapshot(
            _ms(1_500),
            overrides={
                event_name: _warning(
                    event_name,
                    _ms(1_500),
                    duration_ms=1_100,
                    phase=TemporalPhase.EXITING,
                )
            },
        ),
        sequence_id=7,
        context=_context("t7"),
    )
    assert [item.transition for item in exiting] == [CandidateTransition.UPDATE]
    assert exiting[0].condition is ConditionState.FALSE

    cleared = emitter.emit(
        _snapshot(
            _ms(2_000),
            overrides={event_name: _clear(event_name, _ms(2_000), duration_ms=1_600)},
        ),
        sequence_id=8,
        context=_context("t8"),
    )
    assert [item.transition for item in cleared] == [CandidateTransition.CLEAR]
    assert cleared[0].episode_id == episode_id
    assert cleared[0].confirmed_at_ns == started_ns
    assert cleared[0].observed_at_ns == _ms(2_000)
    assert cleared[0].duration_ms == pytest.approx(1_600)
    assert cleared[0].reason_code == "condition_exit_confirmed"
    assert cleared[0].confidence.status is ConfidenceStatus.UNCALIBRATED

    assert (
        emitter.emit(
            _snapshot(_ms(2_100)), sequence_id=9, context=_context("t9")
        )
        == ()
    )

    restarted = emitter.emit(
        _snapshot(
            _ms(3_000),
            overrides={event_name: _warning(event_name, _ms(3_000), duration_ms=0)},
        ),
        sequence_id=10,
        context=_context("t10"),
    )
    assert restarted[0].transition is CandidateTransition.START
    assert restarted[0].episode_id != episode_id


def test_parallel_lanes_start_and_clear_independently() -> None:
    emitter = PartACandidateEmitter(update_interval_ms=10_000)
    first_event = "static_too_long"
    second_event = "environment_too_dark"
    timestamp_ns = _ms(100)

    started = emitter.emit(
        _snapshot(
            timestamp_ns,
            overrides={
                first_event: _warning(first_event, timestamp_ns, duration_ms=0),
                second_event: _warning(second_event, timestamp_ns, duration_ms=0),
            },
        ),
        sequence_id=1,
        context=_context(),
    )
    assert [item.event_name for item in started] == [first_event, second_event]
    assert all(item.transition is CandidateTransition.START for item in started)
    first_id, second_id = (item.episode_id for item in started)

    cleared = emitter.emit(
        _snapshot(
            _ms(200),
            overrides={
                first_event: _clear(first_event, _ms(200), duration_ms=100),
                second_event: _warning(second_event, _ms(200), duration_ms=100),
            },
        ),
        sequence_id=2,
        context=_context("parallel-2"),
    )
    assert len(cleared) == 1
    assert cleared[0].event_name == first_event
    assert cleared[0].episode_id == first_id

    still_active = emitter.emit(
        _snapshot(
            _ms(10_100),
            overrides={
                second_event: _warning(second_event, _ms(10_100), duration_ms=10_000)
            },
        ),
        sequence_id=3,
        context=_context("parallel-3"),
    )
    assert len(still_active) == 1
    assert still_active[0].event_name == second_event
    assert still_active[0].episode_id == second_id
    assert still_active[0].transition is CandidateTransition.UPDATE


def test_explicit_unavailable_is_bounded_and_never_synthesizes_clear() -> None:
    emitter = PartACandidateEmitter(update_interval_ms=1_000)
    started_ns = _ms(100)
    event_name = "noise_too_high"
    started = emitter.emit(
        _snapshot(
            started_ns,
            overrides={event_name: _warning(event_name, started_ns, duration_ms=0)},
        ),
        sequence_id=1,
        context=_context("available"),
    )
    episode_id = started[0].episode_id

    unavailable = emitter.mark_unavailable(
        source_id="camera-0",
        sequence_id=2,
        observed_at_ns=_ms(200),
        reason_code="stream_ended",
        context=_context("ended"),
    )
    assert len(unavailable) == len(PART_A_EVENT_NAMES)
    active_unknown = next(item for item in unavailable if item.event_name == event_name)
    assert active_unknown.transition is CandidateTransition.UNKNOWN
    assert active_unknown.episode_id == episode_id
    assert all(item.transition is not CandidateTransition.CLEAR for item in unavailable)

    repeated = emitter.mark_unavailable(
        source_id="camera-0",
        sequence_id=3,
        observed_at_ns=_ms(300),
        reason_code="stream_ended",
        context=_context("ended-again"),
    )
    assert repeated == ()

    changed_reason = emitter.mark_unavailable(
        source_id="camera-0",
        sequence_id=4,
        observed_at_ns=_ms(400),
        reason_code="camera_unavailable",
        context=_context("camera-down"),
    )
    assert len(changed_reason) == len(PART_A_EVENT_NAMES)
    assert all(item.transition is CandidateTransition.UNKNOWN for item in changed_reason)


def test_idle_unknown_recovery_emits_available_not_clear() -> None:
    emitter = PartACandidateEmitter(update_interval_ms=10_000)
    unknown = emitter.emit(
        _snapshot(
            0,
            overrides={
                event_name: _unknown(event_name, 0, reason="pose_missing")
                for event_name in PART_A_EVENT_NAMES
            },
        ),
        sequence_id=0,
        context=_context("unknown"),
    )
    assert len(unknown) == len(PART_A_EVENT_NAMES)

    recovered = emitter.emit(
        _snapshot(_ms(100)),
        sequence_id=1,
        context=_context("recovered"),
    )
    assert len(recovered) == len(PART_A_EVENT_NAMES)
    assert all(item.transition is CandidateTransition.AVAILABLE for item in recovered)
    assert all(item.episode_id is None for item in recovered)
    assert all(item.reason_code == "evidence_available" for item in recovered)


def test_strict_validation_rejects_bad_confidence_context_and_control_evidence() -> None:
    with pytest.raises(ValueError, match="non-calibrated"):
        ConfidenceEstimate(0.5, ConfidenceStatus.UNCALIBRATED)
    with pytest.raises(ValueError, match="calibrated confidence"):
        ConfidenceEstimate(float("nan"), ConfidenceStatus.CALIBRATED, "iso-v1")
    with pytest.raises(ValueError, match="unsupported by candidate schema v1"):
        ConfidenceEstimate(0.8, ConfidenceStatus.CALIBRATED)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_context(), rule_config_sha256="BAD")

    emitter = PartACandidateEmitter()
    timestamp_ns = _ms(100)
    candidate = emitter.emit(
        _snapshot(
            timestamp_ns,
            overrides={
                "static_too_long": _warning(
                    "static_too_long", timestamp_ns, duration_ms=0
                )
            },
        ),
        sequence_id=1,
        context=_context(),
    )[0]
    with pytest.raises(ValueError, match="control fields"):
        replace(candidate, supporting_evidence=(("suggested_action", "move"),))
    for control_key in (
        "motor_pwm",
        "servoVelocity",
        "arduino-output",
        "controller_command",
        "actuation_target",
    ):
        with pytest.raises(ValueError, match="control fields"):
            replace(candidate, supporting_evidence=((control_key, 1),))
    with pytest.raises(ValueError, match="UNKNOWN confidence"):
        replace(
            candidate,
            transition=CandidateTransition.UNKNOWN,
            semantic_state=SemanticState.UNKNOWN,
            condition=ConditionState.UNKNOWN,
            reason_code="missing",
        )


def test_snapshot_validation_is_strict_and_failed_input_does_not_advance_guard() -> None:
    emitter = PartACandidateEmitter()
    bad_context = replace(_context(), rule_config_status="release")
    with pytest.raises(ValueError, match="statuses differ"):
        emitter.emit(_snapshot(0), sequence_id=0, context=bad_context)

    assert emitter.emit(_snapshot(0), sequence_id=0, context=_context()) == ()
    with pytest.raises(ValueError, match="increase strictly"):
        emitter.emit(_snapshot(_ms(1)), sequence_id=0, context=_context("dup"))
    with pytest.raises(ValueError, match="increase strictly"):
        emitter.emit(_snapshot(0), sequence_id=1, context=_context("old-time"))

    missing = replace(
        _snapshot(_ms(1)),
        evaluations=_snapshot(_ms(1)).evaluations[:-1],
    )
    with pytest.raises(ValueError, match="every Part A lane"):
        emitter.emit(missing, sequence_id=1, context=_context("missing"))

    fresh = PartACandidateEmitter()
    event_name = "static_too_long"
    with pytest.raises(ValueError, match="zero duration"):
        fresh.emit(
            _snapshot(
                _ms(10),
                overrides={
                    event_name: _warning(event_name, _ms(10), duration_ms=1)
                },
            ),
            sequence_id=1,
            context=_context("mid-episode"),
        )
    accepted = fresh.emit(
        _snapshot(
            _ms(10),
            overrides={event_name: _warning(event_name, _ms(10), duration_ms=0)},
        ),
        sequence_id=1,
        context=_context("proper-start"),
    )
    assert accepted[0].transition is CandidateTransition.START


def test_session_context_is_frozen_but_trace_id_may_change() -> None:
    emitter = PartACandidateEmitter()
    assert emitter.emit(_snapshot(0), sequence_id=0, context=_context("frame-0")) == ()

    changed = replace(_context("frame-1"), producer_bundle_sha256="9" * 64)
    with pytest.raises(ValueError, match="session context changed"):
        emitter.emit(_snapshot(_ms(1)), sequence_id=1, context=changed)

    assert (
        emitter.emit(
            _snapshot(_ms(1)),
            sequence_id=1,
            context=_context("frame-1"),
        )
        == ()
    )


def test_clear_requires_terminal_duration_and_rejected_snapshot_is_atomic() -> None:
    emitter = PartACandidateEmitter(update_interval_ms=10_000)
    event_name = "bad_posture"
    start_ns = _ms(100)
    emitter.emit(
        _snapshot(
            start_ns,
            overrides={event_name: _warning(event_name, start_ns, duration_ms=0)},
        ),
        sequence_id=1,
        context=_context("start"),
    )
    emitter.emit(
        _snapshot(
            _ms(200),
            overrides={event_name: _warning(event_name, _ms(200), duration_ms=100)},
        ),
        sequence_id=2,
        context=_context("active"),
    )

    with pytest.raises(ValueError, match="terminal active duration"):
        emitter.emit(
            _snapshot(
                _ms(300),
                overrides={event_name: _clear(event_name, _ms(300), duration_ms=0)},
            ),
            sequence_id=3,
            context=_context("bad-clear"),
        )

    accepted = emitter.emit(
        _snapshot(
            _ms(300),
            overrides={event_name: _clear(event_name, _ms(300), duration_ms=200)},
        ),
        sequence_id=3,
        context=_context("good-clear"),
    )
    assert accepted[0].transition is CandidateTransition.CLEAR
    assert accepted[0].duration_ms == pytest.approx(200)


def _candidate_stream() -> tuple[object, ...]:
    emitter = PartACandidateEmitter(update_interval_ms=1)
    first_timestamp = _ms(100)
    second_timestamp = _ms(200)
    events = ("static_too_long", "bad_posture")
    first = emitter.emit(
        _snapshot(
            first_timestamp,
            overrides={
                event_name: _warning(
                    event_name, first_timestamp, duration_ms=0
                )
                for event_name in events
            },
        ),
        sequence_id=10,
        context=_context("frame-10"),
    )
    second = emitter.emit(
        _snapshot(
            second_timestamp,
            overrides={
                event_name: _warning(
                    event_name, second_timestamp, duration_ms=100
                )
                for event_name in events
            },
        ),
        sequence_id=11,
        context=_context("frame-11"),
    )
    return first + second


def _candidate_jsonl_bytes(candidates: tuple[object, ...]) -> bytes:
    return "".join(
        candidate.to_json() + "\n"  # type: ignore[attr-defined]
        for candidate in candidates
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_candidate_mapping_and_jsonl_roundtrip_use_one_immutable_snapshot(
    tmp_path: Path,
) -> None:
    candidates = _candidate_stream()
    first_mapping = json.loads(candidates[0].to_json())  # type: ignore[attr-defined]
    reconstructed = candidate_from_mapping(first_mapping)
    assert reconstructed.to_json() == candidates[0].to_json()  # type: ignore[attr-defined]

    path = tmp_path / "candidates.jsonl"
    payload = _candidate_jsonl_bytes(candidates)
    expected_sha256 = _write_bytes(path, payload)
    with CandidateJsonlFile(path, expected_sha256=expected_sha256) as artifact:
        # Replacing the source after construction cannot change staged bytes.
        path.write_text("not the staged artifact\n", encoding="utf-8")
        parsed = list(artifact.iter_candidates())
        summary = artifact.validate()

    assert [item.to_json() for item in parsed] == [
        item.to_json() for item in candidates  # type: ignore[attr-defined]
    ]
    assert summary.artifact_sha256 == expected_sha256
    assert summary.artifact_bytes == len(payload)
    assert summary.records == len(candidates)
    assert summary.source_id == "camera-0"
    assert summary.context_sha256 is not None
    assert summary.data_status == "synthetic_contract_test"
    assert summary.input_artifact_sha256 == "2" * 64
    assert summary.first_sequence_id == 10
    assert summary.last_sequence_id == 11
    # Per-record trace IDs may vary; immutable producer/model context may not.
    assert parsed[0].context.trace_id != parsed[-1].context.trace_id


def test_candidate_mapping_reader_rejects_schema_drift_and_bad_nested_types() -> None:
    candidate = _candidate_stream()[0]
    canonical = json.loads(candidate.to_json())  # type: ignore[attr-defined]

    invalid_mappings: list[dict[str, object]] = []

    extra_top = json.loads(json.dumps(canonical))
    extra_top["unexpected"] = True
    invalid_mappings.append(extra_top)

    missing_top = json.loads(json.dumps(canonical))
    missing_top.pop("reason_code")
    invalid_mappings.append(missing_top)

    extra_confidence = json.loads(json.dumps(canonical))
    extra_confidence["confidence"]["calibrator"] = "unversioned"
    invalid_mappings.append(extra_confidence)

    extra_context = json.loads(json.dumps(canonical))
    extra_context["context"]["device_handle"] = 1
    invalid_mappings.append(extra_context)

    missing_component = json.loads(json.dumps(canonical))
    missing_component["context"]["components"][0].pop("model_version")
    invalid_mappings.append(missing_component)

    illegal_enum = json.loads(json.dumps(canonical))
    illegal_enum["transition"] = "acknowledged"
    invalid_mappings.append(illegal_enum)

    nested_evidence = json.loads(json.dumps(canonical))
    nested_evidence["supporting_evidence"]["nested"] = {"not": "scalar"}
    invalid_mappings.append(nested_evidence)

    unversioned_evidence = json.loads(json.dumps(canonical))
    unversioned_evidence["supporting_evidence"]["novel_metric"] = 1.0
    invalid_mappings.append(unversioned_evidence)

    boolean_sequence = json.loads(json.dumps(canonical))
    boolean_sequence["sequence_id"] = True
    invalid_mappings.append(boolean_sequence)

    calibrated = json.loads(json.dumps(canonical))
    calibrated["confidence"] = {
        "value": 0.9,
        "status": "calibrated",
        "method_id": "unversioned-calibrator",
    }
    invalid_mappings.append(calibrated)

    impossible_duration = json.loads(
        _candidate_stream()[2].to_json()  # type: ignore[attr-defined]
    )
    impossible_duration["duration_ms"] = 101.0
    invalid_mappings.append(impossible_duration)

    for mapping in invalid_mappings:
        with pytest.raises(CandidateValidationError):
            candidate_from_mapping(mapping)


@pytest.mark.parametrize(
    "payload_builder",
    [
        lambda line: line.replace(
            b'"schema_version":',
            b'"schema_version":"duplicate","schema_version":',
            1,
        ),
        lambda line: line.replace(b'"duration_ms":0.0', b'"duration_ms":NaN', 1),
        lambda line: line.replace(
            b'"duration_ms":0.0', b'"duration_ms":Infinity', 1
        ),
    ],
)
def test_candidate_jsonl_rejects_duplicate_keys_and_non_finite_constants(
    tmp_path: Path,
    payload_builder: object,
) -> None:
    line = _candidate_stream()[0].to_json().encode("utf-8") + b"\n"  # type: ignore[attr-defined]
    payload = payload_builder(line)  # type: ignore[operator]
    path = tmp_path / "invalid.jsonl"
    _write_bytes(path, payload)
    with CandidateJsonlFile(path) as artifact:
        with pytest.raises(CandidateValidationError):
            list(artifact.iter_candidates())


def test_candidate_jsonl_rejects_invalid_utf8_and_excessive_json_depth(
    tmp_path: Path,
) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    _write_bytes(invalid_utf8, b"\xff\n")
    with CandidateJsonlFile(invalid_utf8) as artifact:
        with pytest.raises(CandidateValidationError, match="UTF-8"):
            list(artifact.iter_candidates())

    deeply_nested = tmp_path / "deep.jsonl"
    payload = b'{"unexpected":' + (b"[" * 9) + b"0" + (b"]" * 9) + b"}\n"
    _write_bytes(deeply_nested, payload)
    limits = CandidateJsonlLimits(max_json_depth=8)
    with CandidateJsonlFile(deeply_nested, limits=limits) as artifact:
        with pytest.raises(CandidateValidationError, match="nesting limit"):
            list(artifact.iter_candidates())


def test_candidate_jsonl_limits_and_optional_sha_are_enforced(tmp_path: Path) -> None:
    candidates = _candidate_stream()
    payload = _candidate_jsonl_bytes(candidates[:2])
    path = tmp_path / "bounded.jsonl"
    expected_sha256 = _write_bytes(path, payload)
    first_line_bytes = len(candidates[0].to_json().encode("utf-8")) + 1  # type: ignore[attr-defined]

    with pytest.raises(CandidateValidationError, match="SHA-256 mismatch"):
        CandidateJsonlFile(path, expected_sha256="0" * 64)
    with pytest.raises(CandidateValidationError, match="lowercase SHA-256"):
        CandidateJsonlFile(path, expected_sha256="BAD")
    with pytest.raises(CandidateValidationError, match="total-byte limit"):
        CandidateJsonlFile(
            path,
            limits=CandidateJsonlLimits(max_file_bytes=len(payload) - 1),
        )

    with CandidateJsonlFile(
        path,
        expected_sha256=expected_sha256,
        limits=CandidateJsonlLimits(max_line_bytes=first_line_bytes - 1),
    ) as artifact:
        with pytest.raises(CandidateValidationError, match="line 1"):
            list(artifact.iter_candidates())

    with CandidateJsonlFile(
        path,
        limits=CandidateJsonlLimits(max_records=1),
    ) as artifact:
        with pytest.raises(CandidateValidationError, match="record limit"):
            list(artifact.iter_candidates())

    for keyword, value in (
        ("max_line_bytes", 64 * 1024 + 1),
        ("max_records", 200_001),
        ("max_file_bytes", 512 * 1024 * 1024 + 1),
        ("max_json_depth", 33),
    ):
        with pytest.raises(ValueError, match="immutable safety ceiling"):
            CandidateJsonlLimits(**{keyword: value})


def _write_candidate_mappings(path: Path, mappings: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(
            mapping,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for mapping in mappings
    ).encode("utf-8")
    _write_bytes(path, payload)


def test_candidate_jsonl_rejects_stream_context_source_and_order_drift(
    tmp_path: Path,
) -> None:
    canonical = [
        json.loads(candidate.to_json())  # type: ignore[attr-defined]
        for candidate in _candidate_stream()
    ]

    variants: list[tuple[str, list[dict[str, object]], str]] = []

    source_drift = json.loads(json.dumps(canonical))
    source_drift[1]["source_id"] = "camera-1"
    variants.append(("source", source_drift, "source_id"))

    context_drift = json.loads(json.dumps(canonical))
    context_drift[1]["context"]["producer_bundle_sha256"] = "9" * 64
    variants.append(("context", context_drift, "context"))

    sequence_regression = json.loads(json.dumps(canonical))
    sequence_regression[2]["sequence_id"] = 9
    variants.append(("sequence", sequence_regression, "sequence_id regressed"))

    timestamp_regression = json.loads(json.dumps(canonical))
    timestamp_regression[2]["observed_at_ns"] = _ms(99)
    timestamp_regression[2]["confirmed_at_ns"] = _ms(99)
    timestamp_regression[2]["duration_ms"] = 0.0
    variants.append(
        ("timestamp", timestamp_regression, "observed_at_ns regressed")
    )

    timestamp_drift = json.loads(json.dumps(canonical))
    timestamp_drift[1]["observed_at_ns"] += 1
    timestamp_drift[1]["confirmed_at_ns"] += 1
    variants.append(("same-sequence-time", timestamp_drift, "within a sequence"))

    frozen_timestamp = json.loads(json.dumps(canonical))
    frozen_timestamp[2]["observed_at_ns"] = frozen_timestamp[1]["observed_at_ns"]
    frozen_timestamp[2]["duration_ms"] = 0.0
    variants.append(
        ("new-sequence-frozen-time", frozen_timestamp, "did not increase")
    )

    duplicate_event = json.loads(json.dumps(canonical))
    duplicate_event[1]["event_name"] = duplicate_event[0]["event_name"]
    variants.append(("duplicate-event", duplicate_event, "duplicates an event"))

    for filename, mappings, message in variants:
        path = tmp_path / f"{filename}.jsonl"
        _write_candidate_mappings(path, mappings)
        with CandidateJsonlFile(path) as artifact:
            with pytest.raises(CandidateValidationError, match=message):
                list(artifact.iter_candidates())


def test_candidate_jsonl_accepts_parallel_lane_clear_and_availability_lifecycles(
    tmp_path: Path,
) -> None:
    emitter = PartACandidateEmitter(update_interval_ms=10_000)
    timestamp = _ms(100)
    starts = emitter.emit(
        _snapshot(
            timestamp,
            overrides={
                event_name: _warning(event_name, timestamp, duration_ms=0)
                for event_name in ("static_too_long", "bad_posture")
            },
        ),
        sequence_id=1,
        context=_context("parallel-start"),
    )
    clears = emitter.emit(
        _snapshot(
            _ms(200),
            overrides={
                event_name: _clear(event_name, _ms(200), duration_ms=100)
                for event_name in ("static_too_long", "bad_posture")
            },
        ),
        sequence_id=2,
        context=_context("parallel-clear"),
    )
    path = tmp_path / "parallel-clear.jsonl"
    _write_bytes(path, _candidate_jsonl_bytes(starts + clears))
    with CandidateJsonlFile(path) as artifact:
        assert len(list(artifact.iter_candidates())) == 4

    health_emitter = PartACandidateEmitter(update_interval_ms=10_000)
    unknown = health_emitter.mark_unavailable(
        source_id="camera-0",
        sequence_id=1,
        observed_at_ns=_ms(100),
        reason_code="camera_unavailable",
        context=_context("idle-unknown"),
    )
    available = health_emitter.emit(
        _snapshot(_ms(200)),
        sequence_id=2,
        context=_context("idle-available"),
    )
    health_path = tmp_path / "idle-availability.jsonl"
    _write_bytes(health_path, _candidate_jsonl_bytes(unknown + available))
    with CandidateJsonlFile(health_path) as artifact:
        assert len(list(artifact.iter_candidates())) == 16


def test_candidate_jsonl_rejects_invalid_lane_lifecycles(tmp_path: Path) -> None:
    canonical = [
        json.loads(candidate.to_json())  # type: ignore[attr-defined]
        for candidate in _candidate_stream()
    ]
    start = canonical[0]
    update = canonical[2]
    variants: list[tuple[str, list[dict[str, object]], str]] = []

    clear_before_start = json.loads(json.dumps(start))
    clear_before_start.update(
        {
            "transition": "clear",
            "semantic_state": "normal",
            "condition": "false",
            "reason_code": "condition_exit_confirmed",
        }
    )
    variants.append(
        ("clear-before-start", [clear_before_start], "CLEAR before START")
    )

    wrong_episode = json.loads(json.dumps(update))
    wrong_episode["episode_id"] = "parta-" + ("f" * 32)
    variants.append(
        ("wrong-episode", [start, wrong_episode], "changed the active episode_id")
    )

    duplicate_start = json.loads(json.dumps(start))
    duplicate_start.update(
        {
            "sequence_id": 11,
            "observed_at_ns": _ms(200),
            "confirmed_at_ns": _ms(200),
            "episode_id": "parta-" + ("e" * 32),
        }
    )
    variants.append(
        ("duplicate-start", [start, duplicate_start], "duplicate START")
    )

    regressed_duration = json.loads(json.dumps(update))
    regressed_duration.update(
        {
            "sequence_id": 12,
            "observed_at_ns": _ms(300),
            "duration_ms": 50.0,
        }
    )
    variants.append(
        (
            "duration-regression",
            [start, update, regressed_duration],
            "duration regressed",
        )
    )

    available_without_unknown = json.loads(json.dumps(start))
    available_without_unknown.update(
        {
            "transition": "available",
            "semantic_state": "normal",
            "condition": "false",
            "episode_id": None,
            "confirmed_at_ns": None,
            "reason_code": "evidence_available",
        }
    )
    variants.append(
        (
            "available-without-unknown",
            [available_without_unknown],
            "without a preceding idle UNKNOWN",
        )
    )

    active_unknown_before_start = json.loads(json.dumps(start))
    active_unknown_before_start.update(
        {
            "transition": "unknown",
            "semantic_state": "unknown",
            "condition": "unknown",
            "reason_code": "evidence_unavailable",
            "confidence": {
                "value": None,
                "status": "unavailable",
                "method_id": None,
            },
        }
    )
    variants.append(
        (
            "active-unknown-before-start",
            [active_unknown_before_start],
            "UNKNOWN before START",
        )
    )

    first_active_unknown = json.loads(json.dumps(start))
    first_active_unknown.update(
        {
            "transition": "unknown",
            "semantic_state": "unknown",
            "condition": "unknown",
            "sequence_id": 11,
            "observed_at_ns": _ms(200),
            "duration_ms": 100.0,
            "reason_code": "evidence_unavailable",
            "confidence": {
                "value": None,
                "status": "unavailable",
                "method_id": None,
            },
        }
    )
    growing_active_unknown = json.loads(json.dumps(first_active_unknown))
    growing_active_unknown.update(
        {
            "sequence_id": 12,
            "observed_at_ns": _ms(300),
            "duration_ms": 150.0,
        }
    )
    variants.append(
        (
            "growing-active-unknown",
            [start, first_active_unknown, growing_active_unknown],
            "active UNKNOWN",
        )
    )

    growing_recovery = json.loads(
        _candidate_stream()[2].to_json()  # type: ignore[attr-defined]
    )
    growing_recovery.update(
        {
            "sequence_id": 12,
            "observed_at_ns": _ms(300),
            "duration_ms": 150.0,
        }
    )
    variants.append(
        (
            "growing-active-unknown-recovery",
            [start, first_active_unknown, growing_recovery],
            "after active UNKNOWN",
        )
    )

    for filename, mappings, message in variants:
        path = tmp_path / f"{filename}.jsonl"
        _write_candidate_mappings(path, mappings)
        with CandidateJsonlFile(path) as artifact:
            with pytest.raises(CandidateValidationError, match=message):
                list(artifact.iter_candidates())


def test_empty_candidate_jsonl_is_a_valid_zero_transition_stream(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"
    expected_sha256 = _write_bytes(path, b"")
    with CandidateJsonlFile(path, expected_sha256=expected_sha256) as artifact:
        assert list(artifact.iter_candidates()) == []
        summary = artifact.validate()
    assert summary.records == 0
    assert summary.source_id is None
    assert summary.context_sha256 is None
    assert summary.data_status is None
    assert summary.input_artifact_sha256 is None
