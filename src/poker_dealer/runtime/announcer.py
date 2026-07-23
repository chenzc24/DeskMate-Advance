"""Event-driven announcements that never participate in game decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from queue import Empty, Full, Queue
import subprocess
import sys
from threading import Lock, Thread
import time
from typing import Callable, Mapping, Protocol

from poker_dealer.domain import HandPhase, Seat, role_for_seat
from poker_dealer.game import EventLog, HandEvent

from .event_log import RuntimeEventWriter


class AnnouncementPriority(IntEnum):
    INFORMATION = 10
    TURN = 20
    RECOVERY = 30
    SAFETY = 40


@dataclass(frozen=True, slots=True)
class Announcement:
    text: str
    priority: AnnouncementPriority = AnnouncementPriority.INFORMATION

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("announcement text is required")


class AnnouncerPort(Protocol):
    def announce(self, announcement: Announcement) -> None: ...


class ConsoleAnnouncer:
    """Laptop-safe test adapter; callers can replace it with audible output."""

    def announce(self, announcement: Announcement) -> None:
        print(f"[ANNOUNCER:{announcement.priority.name}] {announcement.text}", flush=True)


class SpeechPlaybackGate:
    """Suppress microphone processing while local announcements may be audible."""

    def __init__(
        self,
        *,
        tail_guard_ms: int = 350,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if tail_guard_ms < 0:
            raise ValueError("tail_guard_ms must be non-negative")
        self._tail_guard_ns = tail_guard_ms * 1_000_000
        self._clock_ns = clock_ns
        self._lock = Lock()
        self._reservations = 0
        self._suppress_until_ns = 0

    def reserve_playback(self) -> None:
        with self._lock:
            self._reservations += 1

    def complete_playback(self) -> None:
        with self._lock:
            if self._reservations <= 0:
                raise RuntimeError("speech playback reservation underflow")
            self._reservations -= 1
            if self._reservations == 0:
                self._suppress_until_ns = max(
                    self._suppress_until_ns,
                    self._clock_ns() + self._tail_guard_ns,
                )

    def is_suppressed(self, observed_at_ns: int | None = None) -> bool:
        now_ns = self._clock_ns() if observed_at_ns is None else observed_at_ns
        with self._lock:
            return self._reservations > 0 or now_ns < self._suppress_until_ns


class WindowsSpeechAnnouncer:
    """Non-blocking Windows laptop TTS adapter backed by System.Speech."""

    def __init__(self, playback_gate: SpeechPlaybackGate | None = None) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows speech announcer requires Windows")
        self._queue: Queue[Announcement | None] = Queue(maxsize=32)
        self._playback_gate = playback_gate
        self._worker = Thread(target=self._run, name="poker-announcer", daemon=True)
        self._worker.start()

    def announce(self, announcement: Announcement) -> None:
        if announcement.priority >= AnnouncementPriority.RECOVERY:
            while True:
                try:
                    dropped = self._queue.get_nowait()
                except Empty:
                    break
                if dropped is None:
                    self._queue.put_nowait(None)
                    return
                self._complete_gate_reservation()
        if self._playback_gate is not None:
            self._playback_gate.reserve_playback()
        try:
            self._queue.put_nowait(announcement)
        except Full:
            self._complete_gate_reservation()
            # Announcements are feedback, never a reason to block game safety.
            return

    def _complete_gate_reservation(self) -> None:
        if self._playback_gate is not None:
            self._playback_gate.complete_playback()

    def _run(self) -> None:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            announcement = self._queue.get()
            if announcement is None:
                return
            try:
                escaped = announcement.text.replace("'", "''")
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{escaped}')"
                )
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        script,
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
            finally:
                self._complete_gate_reservation()

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            return

    def __enter__(self) -> WindowsSpeechAnnouncer:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class AnnouncementPolicy:
    """Render committed domain/runtime events into concise English prompts."""

    _ROLE_LABELS = {
        "button": "Button",
        "small_blind": "Small Blind",
        "big_blind": "Big Blind",
        "under_the_gun": "Under the Gun",
    }
    _STREET_LABELS = {
        "preflop": "Pre-flop",
        "flop": "Flop",
        "turn": "Turn",
        "river": "River",
        "showdown": "Showdown",
    }
    _ACTION_LABELS = {
        "fold": "folds",
        "check": "checks",
        "call": "calls",
        "bet": "bets",
        "raise": "raises",
    }

    def render(
        self, event_type: str, payload: Mapping[str, object]
    ) -> Announcement | None:
        role = self._ROLE_LABELS.get(str(payload.get("role", "")), "player")
        if event_type == "registration_focus_changed":
            return Announcement(f"{role}, please look at me and press confirm.")
        if event_type == "enrollment_completed":
            return Announcement(f"{role} registration complete.")
        if event_type == "roster_ready":
            return Announcement("All four roles are registered. Ready to start.")
        if event_type == "voice_enrollment_started":
            return Announcement(
                f"{role} voice enrollment. Read phrase one in one breath after the prompt."
            )
        if event_type == "voice_enrollment_sample_accepted":
            sample_number = int(payload["sample_number"])
            total_samples = int(payload["total_samples"])
            if sample_number < total_samples:
                return Announcement(
                    f"Phrase {sample_number} accepted. Phrase {sample_number + 1}."
                )
            return None
        if event_type == "voice_enrollment_retry":
            phrase_number = int(payload["phrase_number"])
            return Announcement(
                f"Voice sample too short. Repeat phrase {phrase_number} in one breath."
            )
        if event_type == "voice_enrollment_completed":
            return Announcement(f"{role} voice enrollment complete.")
        if event_type == "voice_enrollment_cancelled":
            return Announcement(f"{role} voice enrollment cancelled.")
        if event_type == "blind_posted":
            amount = int(payload["amount_units"])
            return Announcement(f"{role} posts {amount}.")
        if event_type == "turn_started":
            return Announcement(f"{role} to act.", AnnouncementPriority.TURN)
        if event_type == "action_committed":
            action_value = str(payload["action"])
            action = self._ACTION_LABELS.get(
                action_value, action_value.replace("_", " ")
            )
            amount = payload.get("amount_units")
            text = f"{role} {action}." if amount is None else f"{role} {action} {amount}."
            return Announcement(text)
        if event_type == "street_started":
            street = self._STREET_LABELS.get(
                str(payload.get("street", "")), str(payload.get("street", "street"))
            )
            return Announcement(f"{street}.")
        if event_type == "hand_paused":
            return Announcement(
                "The hand is paused. Operator assistance is required.",
                AnnouncementPriority.RECOVERY,
            )
        if event_type == "safety_stop":
            return Announcement(
                "Safety stop. Keep clear of the robot.",
                AnnouncementPriority.SAFETY,
            )
        return None


class EventAnnouncer:
    def __init__(
        self, port: AnnouncerPort, policy: AnnouncementPolicy | None = None
    ) -> None:
        self.port = port
        self.policy = policy or AnnouncementPolicy()

    def publish(self, event_type: str, **payload: object) -> Announcement | None:
        announcement = self.policy.render(event_type, payload)
        if announcement is not None:
            self.port.announce(announcement)
        return announcement


class AnnouncingRuntimeEventWriter(RuntimeEventWriter):
    """Copy audit evidence and announce only committed runtime/engine events."""

    def __init__(
        self,
        path,
        announcer: EventAnnouncer,
    ) -> None:
        super().__init__(path)
        self.announcer = announcer
        self._roles_by_seat: dict[str, str] = {}

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        super().emit(kind, observed_at_ns=observed_at_ns, payload=payload)
        if kind == "registration_enrolled":
            role = str(payload.get("role", ""))
            seat = str(payload.get("seat", ""))
            if role and seat:
                self._roles_by_seat[seat] = role
            self.announcer.publish("enrollment_completed", role=role)
            self.announcer.publish("voice_enrollment_started", role=role)
        elif kind == "speaker_enrollment_completed":
            role = self._roles_by_seat.get(str(payload.get("seat", "")), "")
            self.announcer.publish("voice_enrollment_completed", role=role)

    def sync_engine(self, engine_log: EventLog) -> None:
        first_unwritten = self._engine_events_written
        super().sync_engine(engine_log)
        for event in engine_log.events[first_unwritten : self._engine_events_written]:
            self._announce_engine_event(event)

    def _announce_engine_event(self, event: HandEvent) -> None:
        if not event.accepted:
            return
        state = event.state_after
        button_value = state.get("button")
        if button_value is None:
            return
        button = Seat(str(button_value))

        def role_for(seat_value: object) -> str:
            return role_for_seat(button, Seat(str(seat_value))).value

        if event.kind == "hole_cards_confirmed":
            acting_seat = state.get("acting_seat")
            if acting_seat is not None:
                self.announcer.publish("turn_started", role=role_for(acting_seat))
            return
        if event.kind == "action_applied":
            self.announcer.publish(
                "action_committed",
                role=role_for(event.payload["seat"]),
                action=event.payload["action"],
            )
            if (
                state.get("phase") == HandPhase.AWAITING_ACTION.value
                and state.get("acting_seat") is not None
            ):
                self.announcer.publish(
                    "turn_started", role=role_for(state["acting_seat"])
                )
            return
        if event.kind == "board_confirmed":
            self.announcer.publish(
                "street_started", street=event.payload.get("street", "")
            )
            if (
                state.get("phase") == HandPhase.AWAITING_ACTION.value
                and state.get("acting_seat") is not None
            ):
                self.announcer.publish(
                    "turn_started", role=role_for(state["acting_seat"])
                )
            return
        if event.kind == "hand_paused":
            self.announcer.publish(
                "hand_paused", reason=event.payload.get("reason", "")
            )
