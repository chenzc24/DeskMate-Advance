"""Streaming evaluation for the eight Part A semantic-state lanes.

This module evaluates an already produced :class:`ErgonomicsRuleSnapshot`
stream.  It deliberately does not infer product quality from unlabelled or
synthetic data.  A synthetic summary is marked ``contract_only`` and an
unlabelled summary exposes an observed alert rate, never a false-trigger rate.

Durations are conservative.  An interval is known only when both endpoint
conditions are known and the timestamp gap is within the configured maximum.
This prevents a stale warning from being integrated across missing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Iterable, Sequence

from .core import ConditionState, SemanticState, TemporalPhase
from .rules import ErgonomicsRuleEngine, ErgonomicsRuleSnapshot, RuleEvaluation


_NS_PER_MS = 1_000_000
_MS_PER_HOUR = 3_600_000.0
_LATCHED_PHASES = frozenset({TemporalPhase.ACTIVE, TemporalPhase.EXITING})
_HARD_MAX_ANNOTATIONS = 100_000
_HARD_MAX_RETAINED_RECORDS_PER_LANE = 4_096
_MAX_ANNOTATION_ID_CHARS = 128
EVENT_NAMES = ErgonomicsRuleEngine.EVENT_NAMES


class EvaluationDataStatus(StrEnum):
    """Permitted evidence interpretations for a stream summary."""

    LABELED_EVIDENCE = "labeled_evidence"
    UNLABELED_SCREENING = "unlabeled_screening"
    SYNTHETIC_CONTRACT_TEST = "synthetic_contract_test"


class AnnotationLabel(StrEnum):
    """Half-open truth interval label for one semantic lane."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    TRANSITION = "transition"
    DO_NOT_SCORE = "do_not_score"


@dataclass(frozen=True, slots=True)
class Annotation:
    """One non-overlapping truth interval, expressed in source timestamps.

    ``eligible_at_ns`` is required for positives.  It is the earliest time at
    which the configured temporal/window behaviour is expected to be able to
    warn.  This keeps intentional dwell time visible: raw latency is measured
    from ``onset_ns`` and excess latency from ``eligible_at_ns``.
    """

    event_id: str
    event_name: str
    label: AnnotationLabel
    onset_ns: int
    offset_ns: int
    eligible_at_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("annotation event_id must not be empty")
        if len(self.event_id) > _MAX_ANNOTATION_ID_CHARS:
            raise ValueError("annotation event_id is too long")
        if not isinstance(self.event_name, str) or self.event_name not in EVENT_NAMES:
            raise ValueError(f"unsupported annotation event_name: {self.event_name!r}")
        if not isinstance(self.label, AnnotationLabel):
            raise TypeError("annotation label must be an AnnotationLabel")
        for name, value in (("onset_ns", self.onset_ns), ("offset_ns", self.offset_ns)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"annotation {name} must be an integer")
            if value < 0:
                raise ValueError(f"annotation {name} must be non-negative")
        if self.offset_ns <= self.onset_ns:
            raise ValueError("annotation offset_ns must be later than onset_ns")
        if self.label is AnnotationLabel.POSITIVE:
            if isinstance(self.eligible_at_ns, bool) or not isinstance(
                self.eligible_at_ns, int
            ):
                raise TypeError("positive annotation eligible_at_ns must be an integer")
            if not self.onset_ns <= self.eligible_at_ns < self.offset_ns:
                raise ValueError(
                    "positive eligible_at_ns must be inside its half-open interval"
                )
        elif self.eligible_at_ns is not None:
            raise ValueError("eligible_at_ns is only valid for positive annotations")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Bounded diagnostic record for one observed warning episode."""

    entry_at_ns: int | None
    observed_started_at_ns: int
    clear_at_ns: int | None
    active_valid_duration_ms: float
    unknown_pause_ms: float
    wall_duration_ms: float
    left_censored: bool
    right_censored: bool


@dataclass(frozen=True, slots=True)
class LaneEvaluationSummary:
    """Continuous metrics for one semantic lane."""

    event_name: str
    warning_entry_count: int
    warning_clear_count: int
    known_duration_ms: float
    unknown_duration_ms: float
    unknown_fraction: float | None
    active_valid_duration_ms: float
    active_unknown_pause_ms: float
    left_censored_episode_count: int
    right_censored_episode_count: int
    cooldown_started_count: int
    cooldown_completed_count: int
    cooldown_left_censored_count: int
    cooldown_right_censored_count: int
    cooldown_suppressed_true_bout_count: int
    cooldown_violation_count: int
    annotation_count: int
    formal_false_trigger_metric_eligible: bool
    formal_detection_latency_metric_eligible: bool
    false_trigger_count: int
    known_negative_duration_ms: float
    false_triggers_per_hour: float | None
    screening_warning_entries_per_hour: float | None
    positive_annotation_count: int
    positive_detection_count: int
    positive_miss_count: int
    raw_detection_latency_ms: tuple[float, ...]
    excess_detection_latency_ms: tuple[float, ...]
    latency_records_dropped: int
    episodes: tuple[EpisodeRecord, ...]
    episode_records_dropped: int


@dataclass(frozen=True, slots=True)
class ContinuousEvaluationSummary:
    """Immutable result for exactly one source/config session."""

    data_status: EvaluationDataStatus
    contract_only: bool
    formal_effect_metric_eligible: bool
    annotated_lane_count: int
    formal_metric_eligible_lane_count: int
    source_id: str
    config_schema_version: str
    config_status: str
    snapshot_count: int
    started_at_ns: int
    ended_at_ns: int
    timeline_duration_ms: float
    parallel_max_active: int
    parallel_known_duration_ms: float
    parallel_unknown_duration_ms: float
    parallel_active_histogram_ms: tuple[float, ...]
    lanes: tuple[LaneEvaluationSummary, ...]

    def lane(self, event_name: str) -> LaneEvaluationSummary:
        for lane in self.lanes:
            if lane.event_name == event_name:
                return lane
        raise KeyError(event_name)


@dataclass(slots=True)
class _OpenEpisode:
    entry_at_ns: int | None
    observed_started_at_ns: int
    active_valid_ns: int = 0
    unknown_pause_ns: int = 0
    left_censored: bool = False


class _LaneAccumulator:
    def __init__(
        self,
        event_name: str,
        annotations: tuple[Annotation, ...],
        *,
        formal_metrics_allowed: bool,
        max_retained_episodes: int,
        max_retained_latencies: int,
    ) -> None:
        self.event_name = event_name
        self.annotations = annotations
        self.formal_metrics_allowed = formal_metrics_allowed
        self.max_retained_episodes = max_retained_episodes
        self.max_retained_latencies = max_retained_latencies
        self.previous: RuleEvaluation | None = None
        self.known_ns = 0
        self.unknown_ns = 0
        self.active_valid_ns = 0
        self.active_unknown_pause_ns = 0
        self.warning_entries = 0
        self.warning_clears = 0
        self.left_censored = 0
        self.right_censored = 0
        self.open_episode: _OpenEpisode | None = None
        self.episodes: list[EpisodeRecord] = []
        self.episode_records_dropped = 0
        self.cooldown_started = 0
        self.cooldown_completed = 0
        self.cooldown_left_censored = 0
        self.cooldown_right_censored = 0
        self.cooldown_suppressed_bouts = 0
        self.cooldown_violation_count = 0
        self._cooldown_expected_end_ns: int | None = None
        self._cooldown_true_bout_open = False
        self.known_negative_ns = 0
        self.false_triggers = 0
        self._annotation_cursor = 0
        self._negative_cursor = 0
        self._positive_annotation_count = sum(
            annotation.label is AnnotationLabel.POSITIVE
            for annotation in annotations
        )
        self._matched_positive_indices: set[int] = set()
        self.raw_latencies_ms: list[float] = []
        self.excess_latencies_ms: list[float] = []
        self.latency_records_dropped = 0

    def begin(self, evaluation: RuleEvaluation, timestamp_ns: int) -> None:
        self.previous = evaluation
        if evaluation.phase in _LATCHED_PHASES:
            self.left_censored += 1
            self.open_episode = _OpenEpisode(
                entry_at_ns=None,
                observed_started_at_ns=timestamp_ns,
                left_censored=True,
            )
        if evaluation.phase is TemporalPhase.COOLDOWN:
            self.cooldown_left_censored += 1
            self._cooldown_expected_end_ns = _cooldown_end_ns(
                timestamp_ns, evaluation.cooldown_remaining_ms
            )
            self._update_suppressed_bout(evaluation)

    def add(
        self,
        evaluation: RuleEvaluation,
        *,
        previous_timestamp_ns: int,
        timestamp_ns: int,
        interval_known: bool,
    ) -> None:
        previous = self.previous
        if previous is None:
            raise RuntimeError("lane accumulator has not been initialized")
        delta_ns = timestamp_ns - previous_timestamp_ns
        if interval_known:
            self.known_ns += delta_ns
            self.known_negative_ns += self._negative_overlap_ns(
                previous_timestamp_ns, timestamp_ns
            )
        else:
            self.unknown_ns += delta_ns

        if previous.phase in _LATCHED_PHASES:
            if self.open_episode is None:
                raise ValueError(f"{self.event_name} has an untracked active phase")
            if interval_known:
                self.active_valid_ns += delta_ns
                self.open_episode.active_valid_ns += delta_ns
            else:
                self.active_unknown_pause_ns += delta_ns
                self.open_episode.unknown_pause_ns += delta_ns

        was_latched = previous.phase in _LATCHED_PHASES
        is_latched = evaluation.phase in _LATCHED_PHASES
        if not was_latched and is_latched:
            self._open_warning(timestamp_ns)
        elif was_latched and not is_latched:
            self._clear_warning(timestamp_ns)

        self._advance_cooldown(previous, evaluation, timestamp_ns)
        self.previous = evaluation

    def _open_warning(self, timestamp_ns: int) -> None:
        if self.open_episode is not None:
            raise ValueError(f"{self.event_name} opened a duplicate warning episode")
        if (
            self._cooldown_expected_end_ns is not None
            and timestamp_ns < self._cooldown_expected_end_ns
        ):
            self.cooldown_violation_count += 1
        self.warning_entries += 1
        self.open_episode = _OpenEpisode(
            entry_at_ns=timestamp_ns,
            observed_started_at_ns=timestamp_ns,
        )
        annotation_index = self._annotation_at(timestamp_ns)
        if annotation_index is None:
            return
        annotation = self.annotations[annotation_index]
        if annotation.label is AnnotationLabel.NEGATIVE:
            self.false_triggers += 1
        elif (
            annotation.label is AnnotationLabel.POSITIVE
            and annotation_index not in self._matched_positive_indices
        ):
            self._matched_positive_indices.add(annotation_index)
            raw_ms = (timestamp_ns - annotation.onset_ns) / _NS_PER_MS
            eligible_at_ns = annotation.eligible_at_ns
            if eligible_at_ns is None:
                raise RuntimeError("validated positive annotation lost eligible timestamp")
            excess_ms = (timestamp_ns - eligible_at_ns) / _NS_PER_MS
            if len(self.raw_latencies_ms) < self.max_retained_latencies:
                self.raw_latencies_ms.append(raw_ms)
                self.excess_latencies_ms.append(excess_ms)
            else:
                self.latency_records_dropped += 1

    def _clear_warning(self, timestamp_ns: int) -> None:
        episode = self.open_episode
        if episode is None:
            raise ValueError(f"{self.event_name} cleared a warning that was not open")
        self.warning_clears += 1
        record = EpisodeRecord(
            entry_at_ns=episode.entry_at_ns,
            observed_started_at_ns=episode.observed_started_at_ns,
            clear_at_ns=timestamp_ns,
            active_valid_duration_ms=episode.active_valid_ns / _NS_PER_MS,
            unknown_pause_ms=episode.unknown_pause_ns / _NS_PER_MS,
            wall_duration_ms=(timestamp_ns - episode.observed_started_at_ns)
            / _NS_PER_MS,
            left_censored=episode.left_censored,
            right_censored=False,
        )
        self._retain_episode(record)
        self.open_episode = None

    def _advance_cooldown(
        self,
        previous: RuleEvaluation,
        current: RuleEvaluation,
        timestamp_ns: int,
    ) -> None:
        was_cooldown = previous.phase is TemporalPhase.COOLDOWN
        is_cooldown = current.phase is TemporalPhase.COOLDOWN
        if not was_cooldown and is_cooldown:
            self.cooldown_started += 1
            self._cooldown_expected_end_ns = _cooldown_end_ns(
                timestamp_ns, current.cooldown_remaining_ms
            )
        elif was_cooldown and is_cooldown:
            expected = _cooldown_end_ns(timestamp_ns, current.cooldown_remaining_ms)
            if (
                self._cooldown_expected_end_ns is None
                or abs(expected - self._cooldown_expected_end_ns) > 1
            ):
                raise ValueError(f"{self.event_name} cooldown remaining time changed deadline")
        elif was_cooldown and not is_cooldown:
            self.cooldown_completed += 1

        self._update_suppressed_bout(current)

    def _update_suppressed_bout(self, evaluation: RuleEvaluation) -> None:
        suppressed = (
            evaluation.phase is TemporalPhase.COOLDOWN
            and evaluation.condition is ConditionState.TRUE
        )
        if suppressed and not self._cooldown_true_bout_open:
            self.cooldown_suppressed_bouts += 1
        self._cooldown_true_bout_open = suppressed

    def _negative_overlap_ns(self, start_ns: int, end_ns: int) -> int:
        while (
            self._negative_cursor < len(self.annotations)
            and self.annotations[self._negative_cursor].offset_ns <= start_ns
        ):
            self._negative_cursor += 1
        overlap_ns = 0
        index = self._negative_cursor
        while index < len(self.annotations):
            annotation = self.annotations[index]
            if annotation.onset_ns >= end_ns:
                break
            if annotation.label is AnnotationLabel.NEGATIVE:
                overlap_ns += max(
                    0,
                    min(end_ns, annotation.offset_ns)
                    - max(start_ns, annotation.onset_ns),
                )
            index += 1
        return overlap_ns

    def _annotation_at(self, timestamp_ns: int) -> int | None:
        while (
            self._annotation_cursor < len(self.annotations)
            and self.annotations[self._annotation_cursor].offset_ns <= timestamp_ns
        ):
            self._annotation_cursor += 1
        if self._annotation_cursor >= len(self.annotations):
            return None
        annotation = self.annotations[self._annotation_cursor]
        if annotation.onset_ns <= timestamp_ns < annotation.offset_ns:
            return self._annotation_cursor
        return None

    def _retain_episode(self, record: EpisodeRecord) -> None:
        if len(self.episodes) < self.max_retained_episodes:
            self.episodes.append(record)
        else:
            self.episode_records_dropped += 1

    def finish(self, ended_at_ns: int) -> LaneEvaluationSummary:
        if self.open_episode is not None:
            self.right_censored += 1
            episode = self.open_episode
            self._retain_episode(
                EpisodeRecord(
                    entry_at_ns=episode.entry_at_ns,
                    observed_started_at_ns=episode.observed_started_at_ns,
                    clear_at_ns=None,
                    active_valid_duration_ms=episode.active_valid_ns / _NS_PER_MS,
                    unknown_pause_ms=episode.unknown_pause_ns / _NS_PER_MS,
                    wall_duration_ms=(ended_at_ns - episode.observed_started_at_ns)
                    / _NS_PER_MS,
                    left_censored=episode.left_censored,
                    right_censored=True,
                )
            )
        if self.previous is not None and self.previous.phase is TemporalPhase.COOLDOWN:
            expected_end = self._cooldown_expected_end_ns
            if expected_end is None or ended_at_ns < expected_end:
                self.cooldown_right_censored += 1

        known_ms = self.known_ns / _NS_PER_MS
        unknown_ms = self.unknown_ns / _NS_PER_MS
        timeline_ms = known_ms + unknown_ms
        screening_rate = (
            self.warning_entries * _MS_PER_HOUR / known_ms if known_ms > 0 else None
        )
        known_negative_ms = self.known_negative_ns / _NS_PER_MS
        false_trigger_rate = (
            self.false_triggers * _MS_PER_HOUR / known_negative_ms
            if known_negative_ms > 0
            else None
        )
        positive_detections = len(self._matched_positive_indices)
        return LaneEvaluationSummary(
            event_name=self.event_name,
            warning_entry_count=self.warning_entries,
            warning_clear_count=self.warning_clears,
            known_duration_ms=known_ms,
            unknown_duration_ms=unknown_ms,
            unknown_fraction=unknown_ms / timeline_ms if timeline_ms > 0 else None,
            active_valid_duration_ms=self.active_valid_ns / _NS_PER_MS,
            active_unknown_pause_ms=self.active_unknown_pause_ns / _NS_PER_MS,
            left_censored_episode_count=self.left_censored,
            right_censored_episode_count=self.right_censored,
            cooldown_started_count=self.cooldown_started,
            cooldown_completed_count=self.cooldown_completed,
            cooldown_left_censored_count=self.cooldown_left_censored,
            cooldown_right_censored_count=self.cooldown_right_censored,
            cooldown_suppressed_true_bout_count=self.cooldown_suppressed_bouts,
            cooldown_violation_count=self.cooldown_violation_count,
            annotation_count=len(self.annotations),
            formal_false_trigger_metric_eligible=(
                self.formal_metrics_allowed and known_negative_ms > 0
            ),
            formal_detection_latency_metric_eligible=(
                self.formal_metrics_allowed and self._positive_annotation_count > 0
            ),
            false_trigger_count=self.false_triggers,
            known_negative_duration_ms=known_negative_ms,
            false_triggers_per_hour=false_trigger_rate,
            screening_warning_entries_per_hour=screening_rate,
            positive_annotation_count=self._positive_annotation_count,
            positive_detection_count=positive_detections,
            positive_miss_count=self._positive_annotation_count - positive_detections,
            raw_detection_latency_ms=tuple(self.raw_latencies_ms),
            excess_detection_latency_ms=tuple(self.excess_latencies_ms),
            latency_records_dropped=self.latency_records_dropped,
            episodes=tuple(self.episodes),
            episode_records_dropped=self.episode_records_dropped,
        )


class ContinuousRuleEvaluator:
    """Single-pass, bounded-memory evaluator for one recorded session."""

    def __init__(
        self,
        *,
        maximum_evidence_gap_ms: int,
        data_status: EvaluationDataStatus = EvaluationDataStatus.UNLABELED_SCREENING,
        annotations: Sequence[Annotation] = (),
        max_retained_episodes_per_lane: int = 128,
        max_retained_latencies_per_lane: int = 128,
    ) -> None:
        if (
            isinstance(maximum_evidence_gap_ms, bool)
            or not isinstance(maximum_evidence_gap_ms, int)
        ):
            raise TypeError("maximum_evidence_gap_ms must be an integer")
        if maximum_evidence_gap_ms <= 0:
            raise ValueError("maximum_evidence_gap_ms must be positive")
        if not isinstance(data_status, EvaluationDataStatus):
            raise TypeError("data_status must be an EvaluationDataStatus")
        for name, value in (
            ("max_retained_episodes_per_lane", max_retained_episodes_per_lane),
            ("max_retained_latencies_per_lane", max_retained_latencies_per_lane),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if value > _HARD_MAX_RETAINED_RECORDS_PER_LANE:
                raise ValueError(f"{name} exceeds the immutable safety ceiling")
        if len(annotations) > _HARD_MAX_ANNOTATIONS:
            raise ValueError("annotations exceed the immutable safety ceiling")
        annotation_tuple = tuple(annotations)
        if not all(isinstance(annotation, Annotation) for annotation in annotation_tuple):
            raise TypeError("annotations must contain only Annotation records")
        if data_status is EvaluationDataStatus.UNLABELED_SCREENING and annotation_tuple:
            raise ValueError("unlabeled_screening cannot carry truth annotations")
        if data_status is EvaluationDataStatus.LABELED_EVIDENCE and not annotation_tuple:
            raise ValueError("labeled_evidence requires at least one annotation")
        _validate_annotation_set(annotation_tuple)
        by_lane = {
            event_name: tuple(
                sorted(
                    (
                        annotation
                        for annotation in annotation_tuple
                        if annotation.event_name == event_name
                    ),
                    key=lambda annotation: annotation.onset_ns,
                )
            )
            for event_name in EVENT_NAMES
        }
        self.maximum_evidence_gap_ns = maximum_evidence_gap_ms * _NS_PER_MS
        self.data_status = data_status
        self.annotations = annotation_tuple
        self._lanes = {
            event_name: _LaneAccumulator(
                event_name,
                by_lane[event_name],
                formal_metrics_allowed=(
                    data_status is EvaluationDataStatus.LABELED_EVIDENCE
                ),
                max_retained_episodes=max_retained_episodes_per_lane,
                max_retained_latencies=max_retained_latencies_per_lane,
            )
            for event_name in EVENT_NAMES
        }
        self._source_id: str | None = None
        self._config_schema_version: str | None = None
        self._config_status: str | None = None
        self._started_at_ns: int | None = None
        self._previous_timestamp_ns: int | None = None
        self._previous_evaluations: dict[str, RuleEvaluation] | None = None
        self._snapshot_count = 0
        self._parallel_histogram_ns = [0] * (len(EVENT_NAMES) + 1)
        self._parallel_known_ns = 0
        self._parallel_unknown_ns = 0
        self._parallel_max_active = 0
        self._finished = False

    def add(self, snapshot: ErgonomicsRuleSnapshot) -> None:
        if self._finished:
            raise RuntimeError("cannot add snapshots after finish")
        evaluations = _validate_snapshot(snapshot)
        timestamp_ns = snapshot.captured_at_ns
        if self._snapshot_count == 0:
            self._source_id = snapshot.source_id
            self._config_schema_version = snapshot.config_schema_version
            self._config_status = snapshot.config_status
            self._started_at_ns = timestamp_ns
            for event_name in EVENT_NAMES:
                self._lanes[event_name].begin(evaluations[event_name], timestamp_ns)
            self._update_point_parallel(evaluations)
        else:
            previous_timestamp_ns = self._previous_timestamp_ns
            previous_evaluations = self._previous_evaluations
            if previous_timestamp_ns is None or previous_evaluations is None:
                raise RuntimeError("evaluator lost its previous snapshot")
            if timestamp_ns <= previous_timestamp_ns:
                raise ValueError("snapshot timestamps must increase strictly")
            if snapshot.source_id != self._source_id:
                raise ValueError("all snapshots must have the same source_id")
            if snapshot.config_schema_version != self._config_schema_version:
                raise ValueError("config_schema_version changed within a session")
            if snapshot.config_status != self._config_status:
                raise ValueError("config_status changed within a session")
            delta_ns = timestamp_ns - previous_timestamp_ns
            gap_within_limit = delta_ns <= self.maximum_evidence_gap_ns
            lane_known: dict[str, bool] = {}
            for event_name in EVENT_NAMES:
                previous = previous_evaluations[event_name]
                current = evaluations[event_name]
                known = (
                    gap_within_limit
                    and previous.condition is not ConditionState.UNKNOWN
                    and current.condition is not ConditionState.UNKNOWN
                )
                lane_known[event_name] = known
                self._lanes[event_name].add(
                    current,
                    previous_timestamp_ns=previous_timestamp_ns,
                    timestamp_ns=timestamp_ns,
                    interval_known=known,
                )
            if all(lane_known.values()):
                active_count = sum(
                    previous_evaluations[event_name].phase in _LATCHED_PHASES
                    for event_name in EVENT_NAMES
                )
                self._parallel_histogram_ns[active_count] += delta_ns
                self._parallel_known_ns += delta_ns
                self._parallel_max_active = max(
                    self._parallel_max_active, active_count
                )
            else:
                self._parallel_unknown_ns += delta_ns
            self._update_point_parallel(evaluations)

        self._snapshot_count += 1
        self._previous_timestamp_ns = timestamp_ns
        self._previous_evaluations = evaluations

    def finish(self) -> ContinuousEvaluationSummary:
        if self._finished:
            raise RuntimeError("finish may only be called once")
        if self._snapshot_count == 0:
            raise ValueError("cannot finish an empty snapshot stream")
        started_at_ns = self._started_at_ns
        ended_at_ns = self._previous_timestamp_ns
        if started_at_ns is None or ended_at_ns is None:
            raise RuntimeError("evaluator timestamps are incomplete")
        for annotation in self.annotations:
            if annotation.onset_ns < started_at_ns or annotation.offset_ns > ended_at_ns:
                raise ValueError(
                    f"annotation {annotation.event_id!r} is outside the snapshot span"
                )
        self._finished = True
        lanes = tuple(
            self._lanes[event_name].finish(ended_at_ns)
            for event_name in EVENT_NAMES
        )
        annotated_lane_count = sum(lane.annotation_count > 0 for lane in lanes)
        formal_metric_eligible_lane_count = sum(
            lane.formal_false_trigger_metric_eligible
            and lane.formal_detection_latency_metric_eligible
            for lane in lanes
        )
        source_id = self._source_id
        schema_version = self._config_schema_version
        config_status = self._config_status
        if source_id is None or schema_version is None or config_status is None:
            raise RuntimeError("evaluator session identity is incomplete")
        return ContinuousEvaluationSummary(
            data_status=self.data_status,
            contract_only=(
                self.data_status is EvaluationDataStatus.SYNTHETIC_CONTRACT_TEST
            ),
            formal_effect_metric_eligible=(
                self.data_status is EvaluationDataStatus.LABELED_EVIDENCE
                and formal_metric_eligible_lane_count == len(EVENT_NAMES)
            ),
            annotated_lane_count=annotated_lane_count,
            formal_metric_eligible_lane_count=formal_metric_eligible_lane_count,
            source_id=source_id,
            config_schema_version=schema_version,
            config_status=config_status,
            snapshot_count=self._snapshot_count,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            timeline_duration_ms=(ended_at_ns - started_at_ns) / _NS_PER_MS,
            parallel_max_active=self._parallel_max_active,
            parallel_known_duration_ms=self._parallel_known_ns / _NS_PER_MS,
            parallel_unknown_duration_ms=self._parallel_unknown_ns / _NS_PER_MS,
            parallel_active_histogram_ms=tuple(
                value / _NS_PER_MS for value in self._parallel_histogram_ns
            ),
            lanes=lanes,
        )

    def _update_point_parallel(
        self, evaluations: dict[str, RuleEvaluation]
    ) -> None:
        if all(
            evaluation.condition is not ConditionState.UNKNOWN
            for evaluation in evaluations.values()
        ):
            self._parallel_max_active = max(
                self._parallel_max_active,
                sum(
                    evaluation.phase in _LATCHED_PHASES
                    for evaluation in evaluations.values()
                ),
            )


def evaluate_rule_snapshots(
    snapshots: Iterable[ErgonomicsRuleSnapshot],
    *,
    maximum_evidence_gap_ms: int,
    data_status: EvaluationDataStatus = EvaluationDataStatus.UNLABELED_SCREENING,
    annotations: Sequence[Annotation] = (),
    max_retained_episodes_per_lane: int = 128,
    max_retained_latencies_per_lane: int = 128,
) -> ContinuousEvaluationSummary:
    """Evaluate one snapshot iterable without retaining the stream."""

    evaluator = ContinuousRuleEvaluator(
        maximum_evidence_gap_ms=maximum_evidence_gap_ms,
        data_status=data_status,
        annotations=annotations,
        max_retained_episodes_per_lane=max_retained_episodes_per_lane,
        max_retained_latencies_per_lane=max_retained_latencies_per_lane,
    )
    for snapshot in snapshots:
        evaluator.add(snapshot)
    return evaluator.finish()


def _validate_annotation_set(annotations: tuple[Annotation, ...]) -> None:
    event_ids: set[str] = set()
    by_lane: dict[str, list[Annotation]] = {name: [] for name in EVENT_NAMES}
    for annotation in annotations:
        if annotation.event_id in event_ids:
            raise ValueError(f"duplicate annotation event_id: {annotation.event_id}")
        event_ids.add(annotation.event_id)
        by_lane[annotation.event_name].append(annotation)
    for event_name, lane_annotations in by_lane.items():
        ordered = sorted(lane_annotations, key=lambda annotation: annotation.onset_ns)
        for previous, current in zip(ordered, ordered[1:]):
            if current.onset_ns < previous.offset_ns:
                raise ValueError(f"overlapping annotations for {event_name}")


def _validate_snapshot(
    snapshot: ErgonomicsRuleSnapshot,
) -> dict[str, RuleEvaluation]:
    if not isinstance(snapshot, ErgonomicsRuleSnapshot):
        raise TypeError("snapshot must be an ErgonomicsRuleSnapshot")
    if not isinstance(snapshot.source_id, str) or not snapshot.source_id.strip():
        raise ValueError("snapshot source_id must not be empty")
    if (
        isinstance(snapshot.captured_at_ns, bool)
        or not isinstance(snapshot.captured_at_ns, int)
    ):
        raise TypeError("snapshot captured_at_ns must be an integer")
    if snapshot.captured_at_ns < 0:
        raise ValueError("snapshot captured_at_ns must be non-negative")
    if (
        not isinstance(snapshot.config_schema_version, str)
        or not snapshot.config_schema_version.strip()
    ):
        raise ValueError("snapshot config_schema_version must not be empty")
    if not isinstance(snapshot.config_status, str) or not snapshot.config_status.strip():
        raise ValueError("snapshot config_status must not be empty")
    if len(snapshot.evaluations) != len(EVENT_NAMES):
        raise ValueError(f"snapshot must contain exactly {len(EVENT_NAMES)} lanes")
    evaluations: dict[str, RuleEvaluation] = {}
    for evaluation in snapshot.evaluations:
        if not isinstance(evaluation, RuleEvaluation):
            raise TypeError("snapshot lanes must be RuleEvaluation records")
        if evaluation.event_name not in EVENT_NAMES:
            raise ValueError(f"snapshot contains unsupported lane: {evaluation.event_name}")
        if evaluation.event_name in evaluations:
            raise ValueError(f"snapshot contains duplicate lane: {evaluation.event_name}")
        if evaluation.observed_at_ns != snapshot.captured_at_ns:
            raise ValueError(
                f"{evaluation.event_name} observed_at_ns does not match snapshot"
            )
        if not isinstance(evaluation.condition, ConditionState):
            raise TypeError(f"{evaluation.event_name} condition has invalid type")
        if not isinstance(evaluation.semantic_state, SemanticState):
            raise TypeError(f"{evaluation.event_name} semantic_state has invalid type")
        if not isinstance(evaluation.phase, TemporalPhase):
            raise TypeError(f"{evaluation.event_name} phase has invalid type")
        for field_name, value in (
            ("evidence_elapsed_ms", evaluation.evidence_elapsed_ms),
            ("active_duration_ms", evaluation.active_duration_ms),
            ("cooldown_remaining_ms", evaluation.cooldown_remaining_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{evaluation.event_name} {field_name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{evaluation.event_name} {field_name} must be finite and non-negative"
                )
        if (
            evaluation.condition is ConditionState.UNKNOWN
        ) != (evaluation.semantic_state is SemanticState.UNKNOWN):
            raise ValueError(
                f"{evaluation.event_name} unknown condition and semantic state must agree"
            )
        if (
            evaluation.semantic_state is SemanticState.WARNING
            and evaluation.phase not in _LATCHED_PHASES
        ):
            raise ValueError(f"{evaluation.event_name} warning has an inactive phase")
        if (
            evaluation.semantic_state is SemanticState.NORMAL
            and evaluation.phase
            not in {
                TemporalPhase.IDLE,
                TemporalPhase.ENTERING,
                TemporalPhase.COOLDOWN,
            }
        ):
            raise ValueError(f"{evaluation.event_name} normal has an active phase")
        evaluations[evaluation.event_name] = evaluation
    missing = set(EVENT_NAMES) - evaluations.keys()
    if missing:
        raise ValueError(f"snapshot is missing lanes: {', '.join(sorted(missing))}")
    return evaluations


def _cooldown_end_ns(timestamp_ns: int, remaining_ms: float) -> int:
    return timestamp_ns + round(remaining_ms * _NS_PER_MS)
