from chip_value_tracker import ChipValueTracker


def _detection(
    value,
    *,
    source_frame,
    bbox=(10, 10, 70, 70),
    score=0.90,
    rejection=None,
):
    return {
        "bbox_xyxy": list(bbox),
        "denomination": value,
        "value_score": score,
        "value_source_frame": source_frame,
        "value_rejection_reason": rejection,
    }


def test_requires_five_distinct_evidence_batches_before_confirmation():
    tracker = ChipValueTracker()
    latest = None
    for index in range(1, 6):
        latest = _detection(5, source_frame=index)
        tracker.update(index, [latest])
    assert latest["track_id"] == 1
    assert latest["stable_denomination"] == 5
    assert latest["value_state"] == "confirmed"


def test_reused_cache_frame_counts_only_once():
    tracker = ChipValueTracker()
    latest = None
    for frame in range(1, 10):
        latest = _detection(1, source_frame=1)
        tracker.update(frame, [latest])
    assert latest["stable_denomination"] is None
    assert latest["value_votes"] == 1


def test_single_alternative_does_not_flip_confirmed_value():
    tracker = ChipValueTracker()
    for frame in range(1, 6):
        detection = _detection(5, source_frame=frame)
        tracker.update(frame, [detection])
    alternative = _detection(1, source_frame=6, score=0.95)
    tracker.update(6, [alternative])
    assert alternative["stable_denomination"] == 5


def test_five_of_seven_jittering_observations_confirm_majority():
    tracker = ChipValueTracker()
    for frame, value in enumerate((5, 5, 1, 5, 1, 5, 5), start=1):
        detection = _detection(value, source_frame=frame)
        tracker.update(frame, [detection])
    assert detection["stable_denomination"] == 5
    assert detection["value_votes"] == 5


def test_three_strong_alternatives_switch_confirmed_value():
    tracker = ChipValueTracker()
    for frame in range(1, 6):
        detection = _detection(5, source_frame=frame)
        tracker.update(frame, [detection])
    for frame in range(6, 9):
        alternative = _detection(1, source_frame=frame, score=0.95)
        tracker.update(frame, [alternative])
    assert alternative["stable_denomination"] == 1


def test_two_quality_failures_hide_but_do_not_forget_confirmed_value():
    tracker = ChipValueTracker()
    for frame in range(1, 6):
        detection = _detection(10, source_frame=frame)
        tracker.update(frame, [detection])
    for frame in range(6, 8):
        rejected = _detection(
            None,
            source_frame=frame,
            rejection="too_flat",
        )
        tracker.update(frame, [rejected])
    assert rejected["stable_denomination"] is None
    assert rejected["value_state"] == "too_flat"

    recovered = _detection(10, source_frame=8)
    tracker.update(8, [recovered])
    assert recovered["stable_denomination"] == 10


def test_track_id_is_available_before_value_evidence_is_ingested():
    tracker = ChipValueTracker()
    detection = _detection(None, source_frame=None)

    tracker.associate(1, [detection])

    assert detection["track_id"] == 1
    assert detection["stable_denomination"] is None
    assert detection["value_state"] == "collecting_0_of_5"

    detection.update(
        {
            "denomination": 5,
            "value_score": 0.9,
            "value_source_frame": 1,
        }
    )
    tracker.ingest([detection])
    assert detection["value_votes"] == 1
