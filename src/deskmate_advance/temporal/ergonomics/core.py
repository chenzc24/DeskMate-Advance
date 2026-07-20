"""Framework-independent temporal confirmation for ergonomic conditions.

The machine consumes an already evaluated three-valued condition.  It does
not own feature thresholds, scheduling, or event emission.  In particular,
``UNKNOWN`` is missing evidence: it can neither confirm a warning nor clear a
previously confirmed warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


_NANOSECONDS_PER_MILLISECOND = 1_000_000


class ConditionState(StrEnum):
    """Truth value of the feature-level condition at a timestamp."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class SemanticState(StrEnum):
    """Public, controller-independent interpretation of a condition."""

    NORMAL = "normal"
    WARNING = "warning"
    UNKNOWN = "unknown"


class TemporalPhase(StrEnum):
    """Internal lifecycle used to explain temporal confirmation."""

    IDLE = "idle"
    ENTERING = "entering"
    ACTIVE = "active"
    EXITING = "exiting"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class TemporalStateConfig:
    """Timestamp-based confirmation and refractory durations."""

    enter_duration_ms: int
    exit_duration_ms: int
    cooldown_ms: int

    def __post_init__(self) -> None:
        for name, value in (
            ("enter_duration_ms", self.enter_duration_ms),
            ("exit_duration_ms", self.exit_duration_ms),
            ("cooldown_ms", self.cooldown_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer number of milliseconds")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class TemporalStateSnapshot:
    """One immutable explanation of the machine state after an update."""

    timestamp_ns: int
    condition: ConditionState
    semantic_state: SemanticState
    phase: TemporalPhase
    evidence_elapsed_ms: float
    active_duration_ms: float
    cooldown_remaining_ms: float


class TemporalStateMachine:
    """Confirm a three-valued condition using timestamps, never frame counts.

    ``UNKNOWN`` interrupts unconfirmed entry evidence.  If a warning was
    already confirmed, unknown evidence exposes ``SemanticState.UNKNOWN`` but
    retains the internal active phase; it therefore cannot silently clear the
    warning.  Unknown evidence while exiting cancels that exit attempt.
    """

    def __init__(self, config: TemporalStateConfig) -> None:
        if not isinstance(config, TemporalStateConfig):
            raise TypeError("config must be a TemporalStateConfig")
        self.config = config
        self._enter_duration_ns = (
            config.enter_duration_ms * _NANOSECONDS_PER_MILLISECOND
        )
        self._exit_duration_ns = (
            config.exit_duration_ms * _NANOSECONDS_PER_MILLISECOND
        )
        self._cooldown_ns = config.cooldown_ms * _NANOSECONDS_PER_MILLISECOND
        self.reset()

    @property
    def phase(self) -> TemporalPhase:
        return self._phase

    def reset(self) -> None:
        """Return to an unused state, including the timestamp guard."""

        self._phase = TemporalPhase.IDLE
        self._last_timestamp_ns: int | None = None
        self._previous_condition: ConditionState | None = None
        self._evidence_started_at_ns: int | None = None
        self._active_started_at_ns: int | None = None
        self._active_duration_ns = 0
        self._cooldown_started_at_ns: int | None = None

    def update(
        self,
        condition: ConditionState,
        timestamp_ns: int,
    ) -> TemporalStateSnapshot:
        """Advance the machine with one strictly newer condition sample."""

        self._validate_update(condition, timestamp_ns)
        self._accumulate_active_interval(condition, timestamp_ns)
        self._last_timestamp_ns = timestamp_ns
        self._previous_condition = condition

        if self._phase is TemporalPhase.COOLDOWN:
            if self._cooldown_complete(timestamp_ns):
                self._set_phase(TemporalPhase.IDLE)
            else:
                semantic_state = (
                    SemanticState.UNKNOWN
                    if condition is ConditionState.UNKNOWN
                    else SemanticState.NORMAL
                )
                return self._snapshot(timestamp_ns, condition, semantic_state)

        if condition is ConditionState.UNKNOWN:
            return self._handle_unknown(timestamp_ns, condition)
        if self._phase is TemporalPhase.IDLE:
            return self._handle_idle(timestamp_ns, condition)
        if self._phase is TemporalPhase.ENTERING:
            return self._handle_entering(timestamp_ns, condition)
        if self._phase is TemporalPhase.ACTIVE:
            return self._handle_active(timestamp_ns, condition)
        if self._phase is TemporalPhase.EXITING:
            return self._handle_exiting(timestamp_ns, condition)
        raise RuntimeError(f"unsupported temporal phase: {self._phase}")

    def _handle_unknown(
        self,
        timestamp_ns: int,
        condition: ConditionState,
    ) -> TemporalStateSnapshot:
        if self._phase is TemporalPhase.ENTERING:
            self._set_phase(TemporalPhase.IDLE)
        elif self._phase is TemporalPhase.EXITING:
            self._set_phase(TemporalPhase.ACTIVE)
        return self._snapshot(
            timestamp_ns,
            condition,
            SemanticState.UNKNOWN,
        )

    def _handle_idle(
        self,
        timestamp_ns: int,
        condition: ConditionState,
    ) -> TemporalStateSnapshot:
        if condition is ConditionState.TRUE:
            self._start_evidence(TemporalPhase.ENTERING, timestamp_ns)
            if self._enter_duration_ns == 0:
                self._activate(timestamp_ns)
                return self._snapshot(
                    timestamp_ns,
                    condition,
                    SemanticState.WARNING,
                )
        return self._snapshot(timestamp_ns, condition, SemanticState.NORMAL)

    def _handle_entering(
        self,
        timestamp_ns: int,
        condition: ConditionState,
    ) -> TemporalStateSnapshot:
        if condition is ConditionState.FALSE:
            self._set_phase(TemporalPhase.IDLE)
        elif self._evidence_complete(timestamp_ns, self._enter_duration_ns):
            self._activate(timestamp_ns)
            return self._snapshot(
                timestamp_ns,
                condition,
                SemanticState.WARNING,
            )
        return self._snapshot(timestamp_ns, condition, SemanticState.NORMAL)

    def _handle_active(
        self,
        timestamp_ns: int,
        condition: ConditionState,
    ) -> TemporalStateSnapshot:
        if condition is ConditionState.FALSE:
            self._start_evidence(TemporalPhase.EXITING, timestamp_ns)
            if self._exit_duration_ns == 0:
                terminal_duration_ns = self._clear_warning(timestamp_ns)
                return self._snapshot(
                    timestamp_ns,
                    condition,
                    SemanticState.NORMAL,
                    active_duration_override_ns=terminal_duration_ns,
                )
        return self._snapshot(timestamp_ns, condition, SemanticState.WARNING)

    def _handle_exiting(
        self,
        timestamp_ns: int,
        condition: ConditionState,
    ) -> TemporalStateSnapshot:
        if condition is ConditionState.TRUE:
            self._set_phase(TemporalPhase.ACTIVE)
        elif self._evidence_complete(timestamp_ns, self._exit_duration_ns):
            terminal_duration_ns = self._clear_warning(timestamp_ns)
            return self._snapshot(
                timestamp_ns,
                condition,
                SemanticState.NORMAL,
                active_duration_override_ns=terminal_duration_ns,
            )
        return self._snapshot(timestamp_ns, condition, SemanticState.WARNING)

    def _clear_warning(self, timestamp_ns: int) -> int:
        """Clear the active phase and return its final timestamp duration.

        The machine itself must reset its duration before cooldown or a fresh
        episode.  The snapshot produced for the clear edge still needs the
        terminal value so an event-candidate adapter does not have to infer it
        from neighbouring samples.
        """

        terminal_duration_ns = self._active_duration_ns
        self._active_started_at_ns = None
        self._active_duration_ns = 0
        if self._cooldown_ns == 0:
            self._set_phase(TemporalPhase.IDLE)
            return terminal_duration_ns
        self._phase = TemporalPhase.COOLDOWN
        self._evidence_started_at_ns = None
        self._cooldown_started_at_ns = timestamp_ns
        return terminal_duration_ns

    def _start_evidence(self, phase: TemporalPhase, timestamp_ns: int) -> None:
        self._phase = phase
        self._evidence_started_at_ns = timestamp_ns
        self._cooldown_started_at_ns = None
        if phase is not TemporalPhase.EXITING:
            self._active_started_at_ns = None
            self._active_duration_ns = 0

    def _activate(self, timestamp_ns: int) -> None:
        self._phase = TemporalPhase.ACTIVE
        self._evidence_started_at_ns = None
        self._cooldown_started_at_ns = None
        self._active_started_at_ns = timestamp_ns
        self._active_duration_ns = 0

    def _set_phase(self, phase: TemporalPhase) -> None:
        self._phase = phase
        self._evidence_started_at_ns = None
        self._cooldown_started_at_ns = None
        if phase not in {TemporalPhase.ACTIVE, TemporalPhase.EXITING}:
            self._active_started_at_ns = None
            self._active_duration_ns = 0

    def _accumulate_active_interval(
        self,
        condition: ConditionState,
        timestamp_ns: int,
    ) -> None:
        if (
            self._phase in {TemporalPhase.ACTIVE, TemporalPhase.EXITING}
            and self._last_timestamp_ns is not None
            and self._previous_condition is not None
            and self._previous_condition is not ConditionState.UNKNOWN
            and condition is not ConditionState.UNKNOWN
        ):
            self._active_duration_ns += timestamp_ns - self._last_timestamp_ns

    def _evidence_complete(self, timestamp_ns: int, duration_ns: int) -> bool:
        if self._evidence_started_at_ns is None:
            raise RuntimeError("temporal evidence has no start timestamp")
        return timestamp_ns - self._evidence_started_at_ns >= duration_ns

    def _cooldown_complete(self, timestamp_ns: int) -> bool:
        if self._cooldown_started_at_ns is None:
            raise RuntimeError("cooldown has no start timestamp")
        return timestamp_ns - self._cooldown_started_at_ns >= self._cooldown_ns

    def _snapshot(
        self,
        timestamp_ns: int,
        condition: ConditionState,
        semantic_state: SemanticState,
        *,
        active_duration_override_ns: int | None = None,
    ) -> TemporalStateSnapshot:
        evidence_elapsed_ms = 0.0
        if self._evidence_started_at_ns is not None:
            evidence_elapsed_ms = (
                timestamp_ns - self._evidence_started_at_ns
            ) / _NANOSECONDS_PER_MILLISECOND

        active_duration_ns = (
            self._active_duration_ns
            if active_duration_override_ns is None
            else active_duration_override_ns
        )
        active_duration_ms = active_duration_ns / _NANOSECONDS_PER_MILLISECOND

        cooldown_remaining_ms = 0.0
        if self._cooldown_started_at_ns is not None:
            elapsed_ns = timestamp_ns - self._cooldown_started_at_ns
            cooldown_remaining_ms = max(
                0.0,
                (self._cooldown_ns - elapsed_ns) / _NANOSECONDS_PER_MILLISECOND,
            )

        return TemporalStateSnapshot(
            timestamp_ns=timestamp_ns,
            condition=condition,
            semantic_state=semantic_state,
            phase=self._phase,
            evidence_elapsed_ms=evidence_elapsed_ms,
            active_duration_ms=active_duration_ms,
            cooldown_remaining_ms=cooldown_remaining_ms,
        )

    def _validate_update(
        self,
        condition: ConditionState,
        timestamp_ns: int,
    ) -> None:
        if not isinstance(condition, ConditionState):
            raise TypeError("condition must be a ConditionState")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("timestamps must increase strictly")
