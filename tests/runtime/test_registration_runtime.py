from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import pytest

from poker_dealer.domain import (
    ColorSpace,
    ControlIntent,
    ControlObservation,
    ControlSource,
    FramePacket,
    Seat,
    TableRole,
)
from poker_dealer.perception.identity import (
    DetectedFaceFeature,
    DuplicateFaceEnrollmentError,
    FaceFrameEvidence,
)
from poker_dealer.runtime import RegistrationPhase, RegistrationRuntime
from poker_dealer.runtime.live_perception import LivePerceptionSession
from poker_dealer.runtime.ports import FrameRead, FrameReadState


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


def test_simulated_participant_is_explicit_and_does_not_claim_enrollment() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    participant = runtime.add_simulated_participant(
        seat=Seat.B,
        participant_id="development-simulator-seat-b",
    )

    assert participant.seat is Seat.B
    assert participant.initial_role is TableRole.SMALL_BLIND
    assert participant.simulated
    assert participant.face_sample_count == 0
    assert not participant.voice_enrolled

    with pytest.raises(ValueError, match="seat is already registered"):
        runtime.add_simulated_participant(
            seat=Seat.B,
            participant_id="another-simulator",
        )


def test_two_humans_and_two_simulators_freeze_a_four_seat_roster() -> None:
    runtime = RegistrationRuntime("session", Seat.A)
    runtime.add_simulated_participant(
        seat=Seat.B,
        participant_id="development-simulator-seat-b",
    )
    runtime.add_simulated_participant(
        seat=Seat.C,
        participant_id="development-simulator-seat-c",
    )
    assert runtime.accept_control(control(1, ControlIntent.CONFIRM)).accepted
    runtime.complete_face_enrollment(5)
    runtime.select_role(TableRole.UNDER_THE_GUN)
    assert runtime.accept_control(control(2, ControlIntent.CONFIRM)).accepted
    runtime.complete_face_enrollment(5)

    outcome = runtime.accept_control(control(3, ControlIntent.START))

    assert outcome.accepted
    assert outcome.roster is not None
    assert {participant.seat for participant in outcome.roster.participants} == set(Seat)
    assert {
        participant.seat
        for participant in outcome.roster.participants
        if participant.simulated
    } == {Seat.B, Seat.C}


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


class _StopAfterDuplicate(Exception):
    pass


class _DuplicateRecoveryFrameSource:
    camera_epoch = 0

    def __init__(self) -> None:
        self.sequence = 0
        self.statuses: list[dict[str, object]] = []
        self.face_statuses: list[str | None] = []

    def read(self) -> FrameRead:
        self.sequence += 1
        observed_at_ns = self.sequence * 200_000_000
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image.setflags(write=False)
        frame = FramePacket(
            self.sequence,
            observed_at_ns,
            "test-camera",
            0,
            320,
            240,
            ColorSpace.BGR,
            5.0,
            0,
            image,
        )
        return FrameRead(
            FrameReadState.OK,
            observed_at_ns,
            frame,
            self.camera_epoch,
        )

    def set_registration_status(self, **status: object) -> None:
        self.statuses.append(status)

    def set_face_detections(
        self,
        boxes: tuple[tuple[int, int, int, int], ...],
        *,
        status: str | None,
    ) -> None:
        del boxes
        self.face_statuses.append(status)


class _TwoPlayerControls:
    def __init__(self) -> None:
        self.polls = 0

    def poll_controls(
        self, observed_at_ns: int
    ) -> tuple[ControlObservation, ...]:
        self.polls += 1
        if self.polls not in {1, 3}:
            return ()
        return (
            ControlObservation(
                f"confirm:{self.polls}",
                ControlIntent.CONFIRM,
                ControlSource.SIMULATOR,
                observed_at_ns,
                "test",
                self.polls,
            ),
        )


class _FaceModel:
    def analyze(self, frame: FramePacket) -> FaceFrameEvidence:
        embedding = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        embedding.setflags(write=False)
        feature = DetectedFaceFeature(
            frame.captured_at_ns,
            (80, 40, 120, 150),
            0.99,
            embedding,
        )
        return FaceFrameEvidence(
            frame.captured_at_ns,
            1,
            0,
            (feature,),
            1.0,
        )


class _AllowingGallery:
    def enroll(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _FourPlayerControls:
    def __init__(self) -> None:
        self.polls = 0

    def poll_controls(
        self, observed_at_ns: int
    ) -> tuple[ControlObservation, ...]:
        self.polls += 1
        intent = (
            ControlIntent.CONFIRM
            if self.polls <= 4
            else ControlIntent.START
        )
        return (
            ControlObservation(
                f"face-only:{self.polls}",
                intent,
                ControlSource.SIMULATOR,
                observed_at_ns,
                "test",
                self.polls,
            ),
        )


class _AudioHealth:
    @staticmethod
    def snapshot() -> SimpleNamespace:
        return SimpleNamespace(
            peak_level=0.0,
            callback_blocks=0,
            is_stale=lambda *_args: False,
        )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: dict[str, object],
    ) -> None:
        del observed_at_ns
        self.events.append((kind, payload))


def test_live_registration_skips_speaker_enrollment_when_speech_is_enabled() -> None:
    frame_source = _DuplicateRecoveryFrameSource()
    sink = _RecordingSink()
    session = object.__new__(LivePerceptionSession)
    session.gallery = _AllowingGallery()
    session.face_model = _FaceModel()
    session.speaker_gallery = object()
    session.identity_config = SimpleNamespace(minimum_samples=1)
    session.config = SimpleNamespace(
        speech_enabled=True,
        consent_confirmed=True,
    )
    session.frame_source = frame_source
    session._audio_health = _AudioHealth()
    session._audio_stream = None
    session._audio_stale_after_ms = 2000

    roster = session.acquire_roster(
        frame_source=frame_source,
        control_source=_FourPlayerControls(),
        event_sink=sink,
        session_id="face-only-registration",
        button=Seat.A,
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
    )

    assert len(roster.participants) == 4
    assert all(not participant.voice_enrolled for participant in roster.participants)
    skipped = [payload for kind, payload in sink.events if kind == "speaker_enrollment_skipped"]
    assert len(skipped) == 4
    assert all(payload["reason"] == "temporarily_disabled" for payload in skipped)
    assert frame_source.statuses[-1]["voice_target"] == 0
    assert frame_source.statuses[-1]["voice_active"] is False


class _GalleryRejectingSecondPlayer:
    def __init__(self) -> None:
        self.calls = 0

    def enroll(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.calls += 1
        if self.calls == 2:
            raise DuplicateFaceEnrollmentError(
                existing_player_id="participant_1",
                existing_seat=Seat.A,
                similarity=0.98,
                threshold=0.45,
            )


class _StopSink:
    def __init__(self) -> None:
        self.duplicate_payload: dict[str, object] | None = None

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: dict[str, object],
    ) -> None:
        del observed_at_ns
        if kind == "registration_face_rejected":
            self.duplicate_payload = payload
            raise _StopAfterDuplicate


def test_duplicate_second_face_is_retryable_without_erasing_button() -> None:
    frame_source = _DuplicateRecoveryFrameSource()
    gallery = _GalleryRejectingSecondPlayer()
    sink = _StopSink()
    session = object.__new__(LivePerceptionSession)
    session.gallery = gallery
    session.face_model = _FaceModel()
    session.speaker_gallery = object()
    session.identity_config = SimpleNamespace(minimum_samples=2)
    session.speaker_config = SimpleNamespace(minimum_samples=3)
    session.config = SimpleNamespace(
        speech_enabled=False,
        consent_confirmed=True,
    )
    session.frame_source = frame_source
    session._speech_playback_gate = None

    with pytest.raises(_StopAfterDuplicate):
        session.acquire_roster(
            frame_source=frame_source,
            control_source=_TwoPlayerControls(),
            event_sink=sink,
            session_id="duplicate-recovery",
            button=Seat.A,
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
        )

    assert gallery.calls == 2
    assert sink.duplicate_payload is not None
    assert sink.duplicate_payload["role"] == "small_blind"
    assert sink.duplicate_payload["existing_role"] == "button"
    assert sink.duplicate_payload["retryable"] is True
    assert frame_source.face_statuses[-1] == "ALREADY REGISTERED: BUTTON"
