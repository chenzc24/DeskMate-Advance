import pytest

from deskmate_advance.temporal.ergonomics.blink import (
    BlinkRateTracker,
    BlinkTrackerConfig,
    EyeState,
)


def _config(**overrides: object) -> BlinkTrackerConfig:
    values = {
        "closed_score": 0.55,
        "open_score": 0.35,
        "window_ms": 1_000,
        "minimum_valid_ms": 150,
        "minimum_blinks_per_minute": 8.0,
        "minimum_closed_ms": 50,
        "maximum_closed_ms": 800,
        "maximum_valid_gap_ms": 500,
    }
    values.update(overrides)
    return BlinkTrackerConfig(**values)


def test_tracker_counts_only_complete_plausible_blinks() -> None:
    tracker = BlinkRateTracker(_config())

    tracker.update(timestamp_ns=0, left_score=0.1, right_score=0.1, valid=True)
    closed = tracker.update(
        timestamp_ns=100_000_000,
        left_score=0.8,
        right_score=0.9,
        valid=True,
    )
    opened = tracker.update(
        timestamp_ns=200_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )

    assert closed.eye_state is EyeState.CLOSED
    assert opened.eye_state is EyeState.OPEN
    assert opened.blink_count == 1
    assert opened.valid_duration_ms == pytest.approx(200)
    assert opened.blinks_per_minute == pytest.approx(300)
    assert opened.low_rate is False


def test_missing_evidence_does_not_bridge_time_or_complete_blink() -> None:
    tracker = BlinkRateTracker(_config(minimum_valid_ms=100))
    tracker.update(timestamp_ns=0, left_score=0.1, right_score=0.1, valid=True)
    tracker.update(timestamp_ns=50_000_000, left_score=0.8, right_score=0.8, valid=True)
    missing = tracker.update(
        timestamp_ns=100_000_000,
        left_score=None,
        right_score=None,
        valid=False,
    )
    result = tracker.update(
        timestamp_ns=200_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )

    assert missing.eye_state is EyeState.UNKNOWN
    assert missing.low_rate is None
    assert result.blink_count == 0
    assert result.valid_duration_ms == pytest.approx(50)
    assert result.low_rate is None


def test_low_rate_requires_minimum_valid_observation_time() -> None:
    tracker = BlinkRateTracker(_config(minimum_valid_ms=300))
    tracker.update(timestamp_ns=0, left_score=0.1, right_score=0.1, valid=True)
    early = tracker.update(
        timestamp_ns=200_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )
    ready = tracker.update(
        timestamp_ns=300_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )

    assert early.low_rate is None
    assert early.reason == "insufficient_valid_eye_time"
    assert ready.low_rate is True
    assert ready.blinks_per_minute == 0


def test_ambiguous_scores_are_not_treated_as_open() -> None:
    tracker = BlinkRateTracker(_config(minimum_valid_ms=100))
    tracker.update(timestamp_ns=0, left_score=0.8, right_score=0.8, valid=True)
    ambiguous = tracker.update(
        timestamp_ns=100_000_000,
        left_score=0.45,
        right_score=0.45,
        valid=True,
    )

    assert ambiguous.eye_state is EyeState.AMBIGUOUS
    assert ambiguous.blink_count == 0


def test_tracker_requires_open_before_closed_to_count_a_blink() -> None:
    tracker = BlinkRateTracker(_config(minimum_valid_ms=100))
    tracker.update(timestamp_ns=0, left_score=0.8, right_score=0.8, valid=True)
    opened = tracker.update(
        timestamp_ns=100_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )

    assert opened.blink_count == 0


def test_long_gap_disarms_incomplete_blink() -> None:
    tracker = BlinkRateTracker(_config(maximum_valid_gap_ms=100))
    tracker.update(timestamp_ns=0, left_score=0.1, right_score=0.1, valid=True)
    tracker.update(timestamp_ns=50_000_000, left_score=0.8, right_score=0.8, valid=True)
    result = tracker.update(
        timestamp_ns=200_000_000,
        left_score=0.1,
        right_score=0.1,
        valid=True,
    )

    assert result.blink_count == 0


def test_tracker_rejects_non_increasing_time_and_bad_config() -> None:
    tracker = BlinkRateTracker(_config())
    tracker.update(timestamp_ns=1, left_score=0.1, right_score=0.1, valid=True)

    with pytest.raises(ValueError, match="increase strictly"):
        tracker.update(timestamp_ns=1, left_score=0.1, right_score=0.1, valid=True)
    with pytest.raises(ValueError, match="open_score"):
        _config(open_score=0.6)
    with pytest.raises(TypeError, match="window_ms"):
        _config(window_ms=True)
    with pytest.raises(TypeError, match="closed_score"):
        _config(closed_score=False)
