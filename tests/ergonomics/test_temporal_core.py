import pytest

from deskmate_advance.temporal.ergonomics.core import (
    ConditionState,
    SemanticState,
    TemporalPhase,
    TemporalStateConfig,
    TemporalStateMachine,
)


def _ms(value: int) -> int:
    return value * 1_000_000


def _machine(
    *,
    enter_ms: int = 1000,
    exit_ms: int = 500,
    cooldown_ms: int = 1000,
) -> TemporalStateMachine:
    return TemporalStateMachine(
        TemporalStateConfig(
            enter_duration_ms=enter_ms,
            exit_duration_ms=exit_ms,
            cooldown_ms=cooldown_ms,
        )
    )


def test_config_rejects_negative_non_integer_and_boolean_durations() -> None:
    with pytest.raises(ValueError, match="enter_duration_ms"):
        TemporalStateConfig(-1, 0, 0)
    with pytest.raises(TypeError, match="exit_duration_ms"):
        TemporalStateConfig(0, 1.5, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cooldown_ms"):
        TemporalStateConfig(0, 0, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        TemporalStateMachine(None)  # type: ignore[arg-type]


def test_update_requires_a_condition_and_strictly_increasing_timestamp() -> None:
    machine = _machine()

    with pytest.raises(TypeError, match="ConditionState"):
        machine.update("true", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        machine.update(ConditionState.TRUE, -1)

    machine.update(ConditionState.TRUE, 1)
    with pytest.raises(ValueError, match="strictly"):
        machine.update(ConditionState.TRUE, 1)
    with pytest.raises(ValueError, match="strictly"):
        machine.update(ConditionState.TRUE, 0)
    with pytest.raises(TypeError, match="integer"):
        machine.update(ConditionState.TRUE, 1.0)  # type: ignore[arg-type]


def test_entry_confirmation_uses_elapsed_timestamps_not_frame_count() -> None:
    machine = _machine(enter_ms=1000)

    first = machine.update(ConditionState.TRUE, _ms(100))
    irregular = machine.update(ConditionState.TRUE, _ms(877))
    confirmed = machine.update(ConditionState.TRUE, _ms(1100))

    assert first.semantic_state is SemanticState.NORMAL
    assert first.phase is TemporalPhase.ENTERING
    assert first.evidence_elapsed_ms == 0
    assert irregular.semantic_state is SemanticState.NORMAL
    assert irregular.evidence_elapsed_ms == pytest.approx(777)
    assert confirmed.semantic_state is SemanticState.WARNING
    assert confirmed.phase is TemporalPhase.ACTIVE
    assert confirmed.active_duration_ms == 0


def test_false_or_unknown_cannot_bridge_entry_evidence() -> None:
    machine = _machine(enter_ms=1000)

    machine.update(ConditionState.TRUE, _ms(0))
    machine.update(ConditionState.TRUE, _ms(700))
    unknown = machine.update(ConditionState.UNKNOWN, _ms(900))
    restarted = machine.update(ConditionState.TRUE, _ms(1500))
    not_yet = machine.update(ConditionState.TRUE, _ms(2400))
    confirmed = machine.update(ConditionState.TRUE, _ms(2500))

    assert unknown.semantic_state is SemanticState.UNKNOWN
    assert unknown.phase is TemporalPhase.IDLE
    assert unknown.evidence_elapsed_ms == 0
    assert restarted.phase is TemporalPhase.ENTERING
    assert restarted.evidence_elapsed_ms == 0
    assert not_yet.semantic_state is SemanticState.NORMAL
    assert confirmed.semantic_state is SemanticState.WARNING

    second = _machine(enter_ms=1000)
    second.update(ConditionState.TRUE, _ms(0))
    reset = second.update(ConditionState.FALSE, _ms(500))
    assert reset.phase is TemporalPhase.IDLE
    assert reset.semantic_state is SemanticState.NORMAL


def test_unknown_never_clears_a_confirmed_warning() -> None:
    machine = _machine(enter_ms=100, exit_ms=500)
    machine.update(ConditionState.TRUE, _ms(0))
    machine.update(ConditionState.TRUE, _ms(100))

    exiting = machine.update(ConditionState.FALSE, _ms(200))
    unknown = machine.update(ConditionState.UNKNOWN, _ms(600))
    restored = machine.update(ConditionState.TRUE, _ms(800))
    continued = machine.update(ConditionState.TRUE, _ms(900))

    assert exiting.phase is TemporalPhase.EXITING
    assert exiting.semantic_state is SemanticState.WARNING
    assert unknown.phase is TemporalPhase.ACTIVE
    assert unknown.semantic_state is SemanticState.UNKNOWN
    assert unknown.evidence_elapsed_ms == 0
    assert unknown.active_duration_ms == pytest.approx(100)
    assert restored.phase is TemporalPhase.ACTIVE
    assert restored.semantic_state is SemanticState.WARNING
    assert restored.active_duration_ms == pytest.approx(100)
    assert continued.active_duration_ms == pytest.approx(200)


def test_exit_confirmation_and_cooldown_require_fresh_entry_evidence() -> None:
    machine = _machine(enter_ms=1000, exit_ms=500, cooldown_ms=1000)
    machine.update(ConditionState.TRUE, _ms(0))
    machine.update(ConditionState.TRUE, _ms(1000))
    machine.update(ConditionState.FALSE, _ms(1100))

    still_warning = machine.update(ConditionState.FALSE, _ms(1599))
    cleared = machine.update(ConditionState.FALSE, _ms(1600))
    suppressed = machine.update(ConditionState.TRUE, _ms(2100))
    fresh_entry = machine.update(ConditionState.TRUE, _ms(2600))
    reconfirmed = machine.update(ConditionState.TRUE, _ms(3600))

    assert still_warning.phase is TemporalPhase.EXITING
    assert still_warning.semantic_state is SemanticState.WARNING
    assert still_warning.active_duration_ms == pytest.approx(599)
    assert cleared.phase is TemporalPhase.COOLDOWN
    assert cleared.semantic_state is SemanticState.NORMAL
    assert cleared.active_duration_ms == pytest.approx(600)
    assert cleared.cooldown_remaining_ms == pytest.approx(1000)
    assert suppressed.phase is TemporalPhase.COOLDOWN
    assert suppressed.semantic_state is SemanticState.NORMAL
    assert suppressed.active_duration_ms == 0
    assert suppressed.cooldown_remaining_ms == pytest.approx(500)
    assert fresh_entry.phase is TemporalPhase.ENTERING
    assert fresh_entry.evidence_elapsed_ms == 0
    assert reconfirmed.semantic_state is SemanticState.WARNING


def test_unknown_is_public_during_cooldown_but_does_not_extend_it() -> None:
    machine = _machine(enter_ms=0, exit_ms=0, cooldown_ms=1000)
    machine.update(ConditionState.TRUE, _ms(0))
    machine.update(ConditionState.FALSE, _ms(100))

    unknown = machine.update(ConditionState.UNKNOWN, _ms(600))
    after_deadline = machine.update(ConditionState.UNKNOWN, _ms(1100))
    fresh = machine.update(ConditionState.TRUE, _ms(1200))

    assert unknown.phase is TemporalPhase.COOLDOWN
    assert unknown.semantic_state is SemanticState.UNKNOWN
    assert unknown.cooldown_remaining_ms == pytest.approx(500)
    assert after_deadline.phase is TemporalPhase.IDLE
    assert after_deadline.semantic_state is SemanticState.UNKNOWN
    assert fresh.phase is TemporalPhase.ACTIVE
    assert fresh.semantic_state is SemanticState.WARNING


def test_zero_durations_transition_immediately_and_reset_clears_guard() -> None:
    machine = _machine(enter_ms=0, exit_ms=0, cooldown_ms=0)

    warning = machine.update(ConditionState.TRUE, _ms(100))
    normal = machine.update(ConditionState.FALSE, _ms(101))

    assert warning.phase is TemporalPhase.ACTIVE
    assert warning.semantic_state is SemanticState.WARNING
    assert normal.phase is TemporalPhase.IDLE
    assert normal.semantic_state is SemanticState.NORMAL
    assert normal.active_duration_ms == pytest.approx(1)

    machine.reset()
    restarted = machine.update(ConditionState.FALSE, 0)
    assert restarted.phase is TemporalPhase.IDLE
    assert restarted.semantic_state is SemanticState.NORMAL
