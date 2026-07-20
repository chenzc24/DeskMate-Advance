from __future__ import annotations

from dataclasses import replace

import pytest

from deskmate_advance.temporal.ergonomics.core import (
    ConditionState,
    SemanticState,
    TemporalPhase,
    TemporalStateConfig,
    TemporalStateMachine,
)
from deskmate_advance.temporal.ergonomics.evaluation import (
    Annotation,
    AnnotationLabel,
    ContinuousRuleEvaluator,
    EvaluationDataStatus,
    evaluate_rule_snapshots,
)
from deskmate_advance.temporal.ergonomics.rules import (
    ErgonomicsRuleEngine,
    ErgonomicsRuleSnapshot,
    RuleEvaluation,
)


EVENT_NAMES = ErgonomicsRuleEngine.EVENT_NAMES


def _ms(value: int) -> int:
    return value * 1_000_000


def _snapshots(
    rows: list[tuple[int, dict[str, ConditionState]]],
    *,
    enter_ms: int = 200,
    exit_ms: int = 200,
    cooldown_ms: int = 300,
) -> list[ErgonomicsRuleSnapshot]:
    machines = {
        event_name: TemporalStateMachine(
            TemporalStateConfig(enter_ms, exit_ms, cooldown_ms)
        )
        for event_name in EVENT_NAMES
    }
    snapshots: list[ErgonomicsRuleSnapshot] = []
    for timestamp_ms, overrides in rows:
        evaluations: list[RuleEvaluation] = []
        for event_name in EVENT_NAMES:
            condition = overrides.get(event_name, ConditionState.FALSE)
            state = machines[event_name].update(condition, _ms(timestamp_ms))
            evaluations.append(
                RuleEvaluation(
                    event_name=event_name,
                    condition=state.condition,
                    semantic_state=state.semantic_state,
                    phase=state.phase,
                    observed_at_ns=state.timestamp_ns,
                    evidence_elapsed_ms=state.evidence_elapsed_ms,
                    active_duration_ms=state.active_duration_ms,
                    cooldown_remaining_ms=state.cooldown_remaining_ms,
                    reason="synthetic" if condition is ConditionState.UNKNOWN else None,
                    evidence=(),
                )
            )
        snapshots.append(
            ErgonomicsRuleSnapshot(
                source_id="synthetic-camera",
                captured_at_ns=_ms(timestamp_ms),
                config_schema_version="1.0",
                config_status="synthetic_contract_defaults",
                evaluations=tuple(evaluations),
                blink_rate=None,
            )
        )
    return snapshots


def test_entry_clear_unknown_and_active_duration_are_conservative() -> None:
    event = "bad_posture"
    stream = _snapshots(
        [
            (0, {}),
            (100, {event: ConditionState.TRUE}),
            (200, {event: ConditionState.TRUE}),
            (300, {event: ConditionState.TRUE}),
            (400, {event: ConditionState.TRUE}),
            (500, {event: ConditionState.UNKNOWN}),
            (600, {event: ConditionState.TRUE}),
            (700, {event: ConditionState.FALSE}),
            (800, {event: ConditionState.FALSE}),
            (900, {event: ConditionState.FALSE}),
        ]
    )

    summary = evaluate_rule_snapshots(
        stream,
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
    )
    lane = summary.lane(event)

    assert summary.contract_only is True
    assert summary.formal_effect_metric_eligible is False
    assert lane.warning_entry_count == 1
    assert lane.warning_clear_count == 1
    assert lane.known_duration_ms == pytest.approx(700)
    assert lane.unknown_duration_ms == pytest.approx(200)
    assert lane.unknown_fraction == pytest.approx(2 / 9)
    assert lane.active_valid_duration_ms == pytest.approx(400)
    assert lane.active_unknown_pause_ms == pytest.approx(200)
    assert len(lane.episodes) == 1
    assert lane.episodes[0].wall_duration_ms == pytest.approx(600)
    assert lane.episodes[0].active_valid_duration_ms == pytest.approx(400)
    assert lane.episodes[0].unknown_pause_ms == pytest.approx(200)
    assert summary.parallel_max_active == 1
    assert summary.parallel_known_duration_ms == pytest.approx(700)
    assert summary.parallel_unknown_duration_ms == pytest.approx(200)
    assert summary.parallel_active_histogram_ms[0] == pytest.approx(300)
    assert summary.parallel_active_histogram_ms[1] == pytest.approx(400)


def test_parallel_histogram_tracks_independent_simultaneous_lanes() -> None:
    first, second, third = EVENT_NAMES[:3]
    stream = _snapshots(
        [
            (0, {}),
            (100, {first: ConditionState.TRUE, second: ConditionState.TRUE}),
            (
                200,
                {
                    first: ConditionState.TRUE,
                    second: ConditionState.TRUE,
                    third: ConditionState.TRUE,
                },
            ),
            (300, {second: ConditionState.TRUE, third: ConditionState.TRUE}),
            (400, {}),
        ],
        enter_ms=0,
        exit_ms=0,
        cooldown_ms=0,
    )

    summary = evaluate_rule_snapshots(
        stream,
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
    )

    assert summary.parallel_max_active == 3
    assert summary.parallel_active_histogram_ms[0] == pytest.approx(100)
    assert summary.parallel_active_histogram_ms[2] == pytest.approx(200)
    assert summary.parallel_active_histogram_ms[3] == pytest.approx(100)
    assert summary.lane(first).warning_entry_count == 1
    assert summary.lane(second).warning_entry_count == 1
    assert summary.lane(third).warning_entry_count == 1


def test_labeled_negative_rate_and_positive_latency_keep_contract_status() -> None:
    event = "static_too_long"
    stream = _snapshots(
        [
            (0, {}),
            (100, {event: ConditionState.TRUE}),
            (200, {}),
            (300, {}),
            (400, {event: ConditionState.TRUE}),
            (500, {event: ConditionState.TRUE}),
            (600, {}),
        ],
        enter_ms=0,
        exit_ms=0,
        cooldown_ms=0,
    )
    annotations = (
        Annotation(
            event_id="negative-1",
            event_name=event,
            label=AnnotationLabel.NEGATIVE,
            onset_ns=_ms(0),
            offset_ns=_ms(300),
        ),
        Annotation(
            event_id="positive-1",
            event_name=event,
            label=AnnotationLabel.POSITIVE,
            onset_ns=_ms(300),
            eligible_at_ns=_ms(350),
            offset_ns=_ms(600),
        ),
    )

    summary = evaluate_rule_snapshots(
        stream,
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
        annotations=annotations,
    )
    lane = summary.lane(event)

    assert summary.contract_only is True
    assert lane.false_trigger_count == 1
    assert lane.known_negative_duration_ms == pytest.approx(300)
    assert lane.false_triggers_per_hour == pytest.approx(12_000)
    assert lane.positive_annotation_count == 1
    assert lane.positive_detection_count == 1
    assert lane.positive_miss_count == 0
    assert lane.raw_detection_latency_ms == pytest.approx((100,))
    assert lane.excess_detection_latency_ms == pytest.approx((50,))


def test_unlabeled_stream_has_screening_rate_but_no_false_trigger_rate() -> None:
    event = "static_too_long"
    stream = _snapshots(
        [(0, {}), (100, {event: ConditionState.TRUE}), (200, {})],
        enter_ms=0,
        exit_ms=0,
        cooldown_ms=0,
    )

    lane = evaluate_rule_snapshots(
        stream,
        maximum_evidence_gap_ms=500,
    ).lane(event)

    assert lane.false_trigger_count == 0
    assert lane.known_negative_duration_ms == 0
    assert lane.false_triggers_per_hour is None
    assert lane.screening_warning_entries_per_hour == pytest.approx(18_000)


def test_partial_truth_annotations_do_not_mark_all_eight_lanes_formal() -> None:
    event = EVENT_NAMES[0]
    summary = evaluate_rule_snapshots(
        _snapshots([(0, {}), (100, {})]),
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.LABELED_EVIDENCE,
        annotations=(
            Annotation(
                event_id="only-one-lane",
                event_name=event,
                label=AnnotationLabel.NEGATIVE,
                onset_ns=_ms(0),
                offset_ns=_ms(100),
            ),
        ),
    )

    assert summary.formal_effect_metric_eligible is False
    assert summary.annotated_lane_count == 1
    assert summary.formal_metric_eligible_lane_count == 0
    assert summary.lane(event).formal_false_trigger_metric_eligible is True
    assert summary.lane(event).formal_detection_latency_metric_eligible is False
    assert summary.lane(EVENT_NAMES[1]).annotation_count == 0
    assert summary.lane(EVENT_NAMES[1]).formal_false_trigger_metric_eligible is False


def test_aggregate_formal_effect_requires_both_metric_families_for_every_lane() -> None:
    all_negative = tuple(
        Annotation(
            event_id=f"negative-{event_name}",
            event_name=event_name,
            label=AnnotationLabel.NEGATIVE,
            onset_ns=_ms(0),
            offset_ns=_ms(100),
        )
        for event_name in EVENT_NAMES
    )
    negative_only = evaluate_rule_snapshots(
        _snapshots([(0, {}), (100, {}), (200, {})]),
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.LABELED_EVIDENCE,
        annotations=all_negative,
    )

    assert negative_only.annotated_lane_count == len(EVENT_NAMES)
    assert negative_only.formal_metric_eligible_lane_count == 0
    assert negative_only.formal_effect_metric_eligible is False
    assert all(
        lane.formal_false_trigger_metric_eligible
        and not lane.formal_detection_latency_metric_eligible
        for lane in negative_only.lanes
    )

    both_metric_families = all_negative + tuple(
        Annotation(
            event_id=f"positive-{event_name}",
            event_name=event_name,
            label=AnnotationLabel.POSITIVE,
            onset_ns=_ms(100),
            eligible_at_ns=_ms(100),
            offset_ns=_ms(200),
        )
        for event_name in EVENT_NAMES
    )
    fully_covered = evaluate_rule_snapshots(
        _snapshots([(0, {}), (100, {}), (200, {})]),
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.LABELED_EVIDENCE,
        annotations=both_metric_families,
    )

    assert fully_covered.formal_metric_eligible_lane_count == len(EVENT_NAMES)
    assert fully_covered.formal_effect_metric_eligible is True


def test_cooldown_suppression_and_censoring_are_not_new_entries() -> None:
    event = "screen_too_close"
    stream = _snapshots(
        [
            (0, {event: ConditionState.TRUE}),
            (100, {}),
            (200, {event: ConditionState.TRUE}),
            (300, {event: ConditionState.TRUE}),
            (400, {event: ConditionState.TRUE}),
            (500, {event: ConditionState.TRUE}),
        ],
        enter_ms=0,
        exit_ms=0,
        cooldown_ms=300,
    )

    lane = evaluate_rule_snapshots(
        stream,
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
    ).lane(event)

    assert lane.left_censored_episode_count == 1
    assert lane.warning_clear_count == 1
    assert lane.cooldown_started_count == 1
    assert lane.cooldown_completed_count == 1
    assert lane.cooldown_suppressed_true_bout_count == 1
    assert lane.cooldown_violation_count == 0
    assert lane.warning_entry_count == 1
    assert lane.right_censored_episode_count == 1
    assert lane.episodes[0].left_censored is True
    assert lane.episodes[1].right_censored is True


def test_large_gap_is_unknown_even_when_endpoint_conditions_are_false() -> None:
    summary = evaluate_rule_snapshots(
        _snapshots([(0, {}), (1000, {})]),
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
    )

    assert summary.parallel_known_duration_ms == 0
    assert summary.parallel_unknown_duration_ms == pytest.approx(1000)
    assert all(lane.known_duration_ms == 0 for lane in summary.lanes)
    assert all(lane.unknown_duration_ms == pytest.approx(1000) for lane in summary.lanes)


def test_retained_diagnostics_are_bounded() -> None:
    event = "noise_too_high"
    rows: list[tuple[int, dict[str, ConditionState]]] = [(0, {})]
    annotations: list[Annotation] = []
    for index in range(5):
        entry_ms = 100 + index * 200
        rows.extend(
            [
                (entry_ms, {event: ConditionState.TRUE}),
                (entry_ms + 100, {}),
            ]
        )
        annotations.append(
            Annotation(
                event_id=f"positive-{index}",
                event_name=event,
                label=AnnotationLabel.POSITIVE,
                onset_ns=_ms(entry_ms),
                eligible_at_ns=_ms(entry_ms),
                offset_ns=_ms(entry_ms + 100),
            )
        )

    lane = evaluate_rule_snapshots(
        _snapshots(rows, enter_ms=0, exit_ms=0, cooldown_ms=0),
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
        annotations=annotations,
        max_retained_episodes_per_lane=2,
        max_retained_latencies_per_lane=2,
    ).lane(event)

    assert lane.warning_entry_count == 5
    assert len(lane.episodes) == 2
    assert lane.episode_records_dropped == 3
    assert lane.positive_detection_count == 5
    assert len(lane.raw_detection_latency_ms) == 2
    assert lane.latency_records_dropped == 3


def test_strict_stream_identity_timestamp_and_lane_validation() -> None:
    first, second = _snapshots([(0, {}), (100, {})])

    evaluator = ContinuousRuleEvaluator(maximum_evidence_gap_ms=500)
    evaluator.add(first)
    with pytest.raises(ValueError, match="source_id"):
        evaluator.add(replace(second, source_id="other-camera"))

    evaluator = ContinuousRuleEvaluator(maximum_evidence_gap_ms=500)
    evaluator.add(first)
    with pytest.raises(ValueError, match="strictly"):
        evaluator.add(first)

    duplicate = replace(
        first,
        evaluations=first.evaluations[:-1] + (first.evaluations[0],),
    )
    with pytest.raises(ValueError, match="duplicate lane"):
        ContinuousRuleEvaluator(maximum_evidence_gap_ms=500).add(duplicate)

    mismatched_observation = replace(
        first,
        evaluations=(
            replace(first.evaluations[0], observed_at_ns=1),
            *first.evaluations[1:],
        ),
    )
    with pytest.raises(ValueError, match="observed_at_ns"):
        ContinuousRuleEvaluator(maximum_evidence_gap_ms=500).add(
            mismatched_observation
        )

    unknown_with_known_condition = replace(
        first,
        evaluations=(
            replace(
                first.evaluations[0],
                semantic_state=SemanticState.UNKNOWN,
            ),
            *first.evaluations[1:],
        ),
    )
    with pytest.raises(ValueError, match="must agree"):
        ContinuousRuleEvaluator(maximum_evidence_gap_ms=500).add(
            unknown_with_known_condition
        )

    normal_active = replace(
        first,
        evaluations=(
            replace(
                first.evaluations[0],
                phase=TemporalPhase.ACTIVE,
            ),
            *first.evaluations[1:],
        ),
    )
    with pytest.raises(ValueError, match="normal has an active phase"):
        ContinuousRuleEvaluator(maximum_evidence_gap_ms=500).add(normal_active)


def test_annotations_are_non_overlapping_bounded_and_status_compatible() -> None:
    event = EVENT_NAMES[0]
    positive = Annotation(
        event_id="positive",
        event_name=event,
        label=AnnotationLabel.POSITIVE,
        onset_ns=0,
        eligible_at_ns=10,
        offset_ns=100,
    )
    overlapping = Annotation(
        event_id="negative",
        event_name=event,
        label=AnnotationLabel.NEGATIVE,
        onset_ns=50,
        offset_ns=150,
    )

    with pytest.raises(ValueError, match="unlabeled_screening"):
        ContinuousRuleEvaluator(
            maximum_evidence_gap_ms=500,
            annotations=(positive,),
        )
    with pytest.raises(ValueError, match="overlapping"):
        ContinuousRuleEvaluator(
            maximum_evidence_gap_ms=500,
            data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
            annotations=(positive, overlapping),
        )
    evaluator = ContinuousRuleEvaluator(
        maximum_evidence_gap_ms=500,
        data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
        annotations=(replace(positive, onset_ns=200, eligible_at_ns=210, offset_ns=300),),
    )
    evaluator.add(_snapshots([(0, {}), (100, {})])[0])
    with pytest.raises(ValueError, match="outside"):
        evaluator.finish()


def test_empty_stream_and_repeated_finish_fail_explicitly() -> None:
    evaluator = ContinuousRuleEvaluator(maximum_evidence_gap_ms=500)
    with pytest.raises(ValueError, match="empty"):
        evaluator.finish()

    evaluator = ContinuousRuleEvaluator(maximum_evidence_gap_ms=500)
    evaluator.add(_snapshots([(0, {})])[0])
    evaluator.finish()
    with pytest.raises(RuntimeError, match="only be called once"):
        evaluator.finish()


def test_annotation_and_retained_record_limits_have_hard_ceilings() -> None:
    annotation = Annotation(
        event_id="bounded",
        event_name=EVENT_NAMES[0],
        label=AnnotationLabel.NEGATIVE,
        onset_ns=0,
        offset_ns=1,
    )
    with pytest.raises(ValueError, match="annotations exceed"):
        ContinuousRuleEvaluator(
            maximum_evidence_gap_ms=500,
            data_status=EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST,
            annotations=[annotation] * 100_001,
        )
    for argument in (
        {"max_retained_episodes_per_lane": 4_097},
        {"max_retained_latencies_per_lane": 4_097},
    ):
        with pytest.raises(ValueError, match="immutable safety ceiling"):
            ContinuousRuleEvaluator(
                maximum_evidence_gap_ms=500,
                **argument,
            )
    with pytest.raises(ValueError, match="event_id is too long"):
        Annotation(
            event_id="x" * 129,
            event_name=EVENT_NAMES[0],
            label=AnnotationLabel.NEGATIVE,
            onset_ns=0,
            offset_ns=1,
        )
