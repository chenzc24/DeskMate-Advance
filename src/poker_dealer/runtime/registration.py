"""Registration-only orchestration before Part A is loaded."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    SEAT_ORDER,
    Seat,
    TableRole,
    role_seats,
)


ROLE_ORDER = (
    TableRole.BUTTON,
    TableRole.SMALL_BLIND,
    TableRole.BIG_BLIND,
    TableRole.UNDER_THE_GUN,
)


class RegistrationPhase(StrEnum):
    READY_FOR_FACE = "ready_for_face"
    CAPTURING_FACE = "capturing_face"
    READY_TO_START = "ready_to_start"
    STARTED = "started"


@dataclass(frozen=True, slots=True)
class RegisteredParticipant:
    participant_id: str
    seat: Seat
    initial_role: TableRole
    face_sample_count: int
    voice_enrolled: bool = False
    simulated: bool = False


@dataclass(frozen=True, slots=True)
class FrozenSessionRoster:
    session_id: str
    button: Seat
    roster_version: int
    participants: tuple[RegisteredParticipant, ...]


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    accepted: bool
    reason: str
    roster: FrozenSessionRoster | None = None


class RegistrationRuntime:
    """Own role-oriented enrollment state without camera/model dependencies."""

    def __init__(self, session_id: str, button: Seat) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.button = button
        self.phase = RegistrationPhase.READY_FOR_FACE
        self.focus_role = TableRole.BUTTON
        self._role_seats = role_seats(button)
        self._participants: dict[Seat, RegisteredParticipant] = {}
        self._seen_controls: set[str] = set()
        self._roster_version = 0

    @property
    def focus_seat(self) -> Seat:
        return self._role_seats[self.focus_role]

    @property
    def participant_id(self) -> str:
        index = ROLE_ORDER.index(self.focus_role) + 1
        return f"participant_{index}"

    @property
    def registered_seats(self) -> frozenset[Seat]:
        return frozenset(self._participants)

    @property
    def participants(self) -> tuple[RegisteredParticipant, ...]:
        return tuple(
            self._participants[self._role_seats[role]]
            for role in ROLE_ORDER
            if self._role_seats[role] in self._participants
        )

    def select_role(self, role: TableRole) -> None:
        if self.phase in {RegistrationPhase.CAPTURING_FACE, RegistrationPhase.STARTED}:
            raise ValueError("cannot change registration role in the current phase")
        self.focus_role = role
        self.phase = (
            RegistrationPhase.READY_TO_START
            if len(self._participants) == len(SEAT_ORDER)
            else RegistrationPhase.READY_FOR_FACE
        )

    def accept_control(self, observation: ControlObservation) -> RegistrationOutcome:
        if observation.observation_id in self._seen_controls:
            return RegistrationOutcome(False, "duplicate_control")
        self._seen_controls.add(observation.observation_id)
        if observation.intent is ControlIntent.CONFIRM:
            if self.phase is not RegistrationPhase.READY_FOR_FACE:
                return RegistrationOutcome(False, "face_capture_not_available")
            if self.focus_seat in self._participants:
                return RegistrationOutcome(False, "role_already_registered")
            self.phase = RegistrationPhase.CAPTURING_FACE
            return RegistrationOutcome(True, "face_capture_started")
        if observation.intent is ControlIntent.CANCEL:
            if self.phase is not RegistrationPhase.CAPTURING_FACE:
                return RegistrationOutcome(False, "nothing_to_cancel")
            self.phase = RegistrationPhase.READY_FOR_FACE
            return RegistrationOutcome(True, "face_capture_cancelled")
        if observation.intent is ControlIntent.CLEAR:
            if self.phase is RegistrationPhase.STARTED:
                return RegistrationOutcome(False, "roster_already_frozen")
            self._participants.clear()
            self.focus_role = TableRole.BUTTON
            self.phase = RegistrationPhase.READY_FOR_FACE
            self._roster_version += 1
            return RegistrationOutcome(True, "registration_cleared")
        if observation.intent in {
            ControlIntent.NEXT_OPTION,
            ControlIntent.PREVIOUS_OPTION,
        }:
            if self.phase in {
                RegistrationPhase.CAPTURING_FACE,
                RegistrationPhase.STARTED,
            }:
                return RegistrationOutcome(False, "role_navigation_not_available")
            offset = 1 if observation.intent is ControlIntent.NEXT_OPTION else -1
            current = ROLE_ORDER.index(self.focus_role)
            self.select_role(ROLE_ORDER[(current + offset) % len(ROLE_ORDER)])
            return RegistrationOutcome(True, "registration_role_changed")
        if observation.intent is ControlIntent.START:
            if self.phase is not RegistrationPhase.READY_TO_START:
                return RegistrationOutcome(False, "four_roles_required")
            roster = FrozenSessionRoster(
                self.session_id,
                self.button,
                self._roster_version,
                self.participants,
            )
            self.phase = RegistrationPhase.STARTED
            return RegistrationOutcome(True, "roster_frozen", roster)
        return RegistrationOutcome(False, "unsupported_control_intent")

    def complete_face_enrollment(self, sample_count: int) -> RegisteredParticipant:
        if self.phase is not RegistrationPhase.CAPTURING_FACE:
            raise ValueError("face enrollment is not active")
        if sample_count <= 0:
            raise ValueError("face sample count must be positive")
        participant = RegisteredParticipant(
            participant_id=self.participant_id,
            seat=self.focus_seat,
            initial_role=self.focus_role,
            face_sample_count=sample_count,
        )
        self._participants[participant.seat] = participant
        self.phase = (
            RegistrationPhase.READY_TO_START
            if len(self._participants) == len(SEAT_ORDER)
            else RegistrationPhase.READY_FOR_FACE
        )
        return participant

    def add_simulated_participant(
        self,
        *,
        seat: Seat,
        participant_id: str,
    ) -> RegisteredParticipant:
        """Add one explicit development simulator without claiming enrollment."""

        if self.phase in {RegistrationPhase.CAPTURING_FACE, RegistrationPhase.STARTED}:
            raise ValueError("cannot add a simulator in the current phase")
        if not participant_id.strip():
            raise ValueError("simulated participant ID is required")
        if seat in self._participants:
            raise ValueError("seat is already registered")
        if any(
            participant.participant_id == participant_id
            for participant in self._participants.values()
        ):
            raise ValueError("participant ID is already registered")
        role = next(
            role
            for role, mapped_seat in self._role_seats.items()
            if mapped_seat is seat
        )
        participant = RegisteredParticipant(
            participant_id=participant_id,
            seat=seat,
            initial_role=role,
            face_sample_count=0,
            voice_enrolled=False,
            simulated=True,
        )
        self._participants[seat] = participant
        self.phase = (
            RegistrationPhase.READY_TO_START
            if len(self._participants) == len(SEAT_ORDER)
            else RegistrationPhase.READY_FOR_FACE
        )
        return participant

    def reject_face_enrollment(self) -> None:
        if self.phase is not RegistrationPhase.CAPTURING_FACE:
            raise ValueError("face enrollment is not active")
        self.phase = RegistrationPhase.READY_FOR_FACE

    def mark_voice_enrolled(self, seat: Seat) -> None:
        if self.phase is RegistrationPhase.STARTED:
            raise ValueError("roster already frozen")
        participant = self._participants.get(seat)
        if participant is None:
            raise ValueError("face enrollment must precede voice enrollment")
        self._participants[seat] = RegisteredParticipant(
            participant.participant_id,
            participant.seat,
            participant.initial_role,
            participant.face_sample_count,
            True,
            participant.simulated,
        )
