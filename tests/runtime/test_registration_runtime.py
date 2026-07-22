from __future__ import annotations

import pytest

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    ControlSource,
    Seat,
    TableRole,
)
from poker_dealer.runtime import RegistrationPhase, RegistrationRuntime


def control(sequence: int, intent: ControlIntent) -> ControlObservation:
    return ControlObservation(
        f"control:{sequence}",
        intent,
        ControlSource.SIMULATOR,
        sequence,
        "test",
        sequence,
    )


def test_registration_uses_roles_and_freezes_four_participant_roster() -> None:
    runtime = RegistrationRuntime("session", Seat.C)
    expected = (
        (TableRole.BUTTON, Seat.C),
        (TableRole.SMALL_BLIND, Seat.D),
        (TableRole.BIG_BLIND, Seat.A),
        (TableRole.UNDER_THE_GUN, Seat.B),
    )
    for sequence, (role, seat) in enumerate(expected, start=1):
        runtime.select_role(role)
        assert runtime.focus_seat is seat
        assert runtime.accept_control(control(sequence, ControlIntent.CONFIRM)).accepted
        participant = runtime.complete_face_enrollment(5)
        assert participant.initial_role is role
        assert participant.seat is seat
    assert runtime.phase is RegistrationPhase.READY_TO_START
    outcome = runtime.accept_control(control(10, ControlIntent.START))
    assert outcome.accepted
    assert outcome.roster is not None
    assert outcome.roster.button is Seat.C
    assert [item.participant_id for item in outcome.roster.participants] == [
        "participant_1",
        "participant_2",
        "participant_3",
        "participant_4",
    ]
    assert runtime.phase is RegistrationPhase.STARTED


def test_start_is_blocked_until_all_roles_are_registered() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    outcome = runtime.accept_control(control(1, ControlIntent.START))
    assert not outcome.accepted
    assert outcome.reason == "four_roles_required"


def test_duplicate_controls_and_clear_are_idempotent_safe() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    observation = control(1, ControlIntent.CONFIRM)
    assert runtime.accept_control(observation).accepted
    assert runtime.accept_control(observation).reason == "duplicate_control"
    runtime.complete_face_enrollment(5)
    cleared = runtime.accept_control(control(2, ControlIntent.CLEAR))
    assert cleared.accepted
    assert not runtime.registered_seats
    assert runtime.focus_role is TableRole.BUTTON


def test_voice_is_optional_but_can_be_recorded_in_roster() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    assert runtime.accept_control(control(1, ControlIntent.CONFIRM)).accepted
    participant = runtime.complete_face_enrollment(5)
    runtime.mark_voice_enrolled(participant.seat)
    assert runtime.participants[0].voice_enrolled


def test_next_previous_controls_navigate_roles_without_crashing() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    next_outcome = runtime.accept_control(control(1, ControlIntent.NEXT_OPTION))
    assert next_outcome.accepted
    assert runtime.focus_role is TableRole.SMALL_BLIND
    previous = runtime.accept_control(control(2, ControlIntent.PREVIOUS_OPTION))
    assert previous.accepted
    assert runtime.focus_role is TableRole.BUTTON


def test_frozen_roster_cannot_be_mutated_by_late_voice_enrollment() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    for sequence, role in enumerate(
        (
            TableRole.BUTTON,
            TableRole.SMALL_BLIND,
            TableRole.BIG_BLIND,
            TableRole.UNDER_THE_GUN,
        ),
        start=1,
    ):
        runtime.select_role(role)
        assert runtime.accept_control(control(sequence, ControlIntent.CONFIRM)).accepted
        runtime.complete_face_enrollment(5)
    assert runtime.accept_control(control(10, ControlIntent.START)).accepted
    with pytest.raises(ValueError, match="roster already frozen"):
        runtime.mark_voice_enrolled(Seat.A)
