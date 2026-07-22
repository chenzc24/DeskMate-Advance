"""Event-driven announcements that never participate in game decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from queue import Queue
import subprocess
import sys
from threading import Thread
from typing import Mapping, Protocol


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


class WindowsSpeechAnnouncer:
    """Non-blocking Windows laptop TTS adapter backed by System.Speech."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows speech announcer requires Windows")
        self._queue: Queue[Announcement | None] = Queue(maxsize=32)
        self._worker = Thread(target=self._run, name="poker-announcer", daemon=True)
        self._worker.start()

    def announce(self, announcement: Announcement) -> None:
        if announcement.priority >= AnnouncementPriority.RECOVERY:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
        try:
            self._queue.put_nowait(announcement)
        except Exception:
            # Announcements are feedback, never a reason to block game safety.
            return

    def _run(self) -> None:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            announcement = self._queue.get()
            if announcement is None:
                return
            escaped = announcement.text.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{escaped}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )

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
