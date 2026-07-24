"""Event-driven announcements that never participate in game decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from queue import Empty, Full, Queue
from string import Formatter
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


@dataclass(frozen=True, slots=True)
class AnnouncementTemplate:
    text: str
    priority: AnnouncementPriority


@dataclass(frozen=True, slots=True)
class AnnouncementCatalog:
    schema_version: str
    catalog_id: str
    catalog_version: str
    language: str
    speech_rate: int
    speech_volume: int
    voice_preferences: tuple[str, ...]
    entries: Mapping[str, AnnouncementTemplate]

    _REQUIRED_ENTRY_IDS = frozenset(
        {
            "system_ready",
            "audio_link_lost",
            "audio_link_restored",
            "microphone_unavailable",
            "camera_disconnected",
            "registration_focus_changed",
            "duplicate_face_rejected",
            "enrollment_completed",
            "roster_ready",
            "voice_enrollment_started",
            "voice_enrollment_sample_accepted",
            "voice_enrollment_retry",
            "voice_enrollment_completed",
            "voice_enrollment_cancelled",
            "blind_posted",
            "dealing_hole_cards",
            "turn_started",
            "action_pending_confirmation",
            "action_committed",
            "action_committed_with_amount",
            "action_cancelled",
            "command_not_understood",
            "illegal_action",
            "action_timeout",
            "action_confirmation_timeout",
            "street_started",
            "dealing_flop",
            "dealing_turn",
            "dealing_river",
            "card_unknown",
            "duplicate_card",
            "card_delivery_failed",
            "showdown_started",
            "pot_awarded",
            "split_pot",
            "hand_completed",
            "hand_voided",
            "hand_paused",
            "dealer_timeout",
            "dealer_jam",
            "table_not_clear",
            "safety_stop",
            "session_completed",
        }
    )
    _ALLOWED_PLACEHOLDERS = frozenset(
        {
            "action",
            "amount_units",
            "next_phrase_number",
            "next_phrase",
            "phrase",
            "phrase_number",
            "reason",
            "role",
            "existing_role",
            "street",
            "winner",
        }
    )

    @classmethod
    def from_json(cls, path: Path) -> AnnouncementCatalog:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported announcement catalog schema version")
        entries_value = value.get("entries")
        if not isinstance(entries_value, dict):
            raise ValueError("announcement catalog entries must be an object")
        entries: dict[str, AnnouncementTemplate] = {}
        priorities = {
            priority.name.lower(): priority for priority in AnnouncementPriority
        }
        formatter = Formatter()
        for entry_id, raw in entries_value.items():
            if not isinstance(entry_id, str) or not entry_id.strip():
                raise ValueError("announcement entry IDs must be non-empty strings")
            if not isinstance(raw, dict):
                raise ValueError(f"announcement entry {entry_id} must be an object")
            text = raw.get("text")
            priority_value = raw.get("priority")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"announcement entry {entry_id} requires text")
            if priority_value not in priorities:
                raise ValueError(
                    f"announcement entry {entry_id} has invalid priority"
                )
            for _literal, field, format_spec, conversion in formatter.parse(text):
                if field is None:
                    continue
                if (
                    field not in cls._ALLOWED_PLACEHOLDERS
                    or format_spec
                    or conversion
                ):
                    raise ValueError(
                        f"announcement entry {entry_id} has invalid placeholder"
                    )
            entries[entry_id] = AnnouncementTemplate(
                text=text,
                priority=priorities[str(priority_value)],
            )
        missing = cls._REQUIRED_ENTRY_IDS - entries.keys()
        if missing:
            raise ValueError(
                "announcement catalog is missing required entries: "
                + ",".join(sorted(missing))
            )
        voice_preferences = value.get("voice_preferences")
        if (
            not isinstance(voice_preferences, list)
            or not voice_preferences
            or any(
                not isinstance(voice, str) or not voice.strip()
                for voice in voice_preferences
            )
        ):
            raise ValueError("announcement voice preferences must be non-empty")
        catalog = cls(
            schema_version="1.0",
            catalog_id=str(value.get("catalog_id", "")),
            catalog_version=str(value.get("catalog_version", "")),
            language=str(value.get("language", "")),
            speech_rate=int(value.get("speech_rate", 0)),
            speech_volume=int(value.get("speech_volume", 100)),
            voice_preferences=tuple(voice_preferences),
            entries=entries,
        )
        if not all(
            (
                catalog.catalog_id.strip(),
                catalog.catalog_version.strip(),
                catalog.language.strip(),
            )
        ):
            raise ValueError("announcement catalog identity and language are required")
        if not -10 <= catalog.speech_rate <= 10:
            raise ValueError("announcement speech rate must be in [-10, 10]")
        if not 0 <= catalog.speech_volume <= 100:
            raise ValueError("announcement speech volume must be in [0, 100]")
        return catalog

    def render(
        self, entry_id: str, payload: Mapping[str, object]
    ) -> Announcement | None:
        template = self.entries.get(entry_id)
        if template is None:
            return None
        try:
            text = template.text.format_map(dict(payload))
        except (KeyError, ValueError):
            return None
        return Announcement(text, template.priority)


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

    def __init__(
        self,
        playback_gate: SpeechPlaybackGate | None = None,
        *,
        language: str = "en-US",
        voice_preferences: tuple[str, ...] = (),
        speech_rate: int = 0,
        speech_volume: int = 100,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows speech announcer requires Windows")
        if not language.strip():
            raise ValueError("speech language is required")
        if not -10 <= speech_rate <= 10 or not 0 <= speech_volume <= 100:
            raise ValueError("speech rate or volume is outside the supported range")
        self._queue: Queue[Announcement | None] = Queue(maxsize=32)
        self._playback_gate = playback_gate
        self._language = language
        self._voice_preferences = voice_preferences
        self._speech_rate = speech_rate
        self._speech_volume = speech_volume
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
                language = self._language.replace("'", "''")
                preferred = ",".join(
                    "'" + voice.replace("'", "''") + "'"
                    for voice in self._voice_preferences
                )
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {self._speech_rate}; "
                    f"$s.Volume = {self._speech_volume}; "
                    "$installed = @($s.GetInstalledVoices() | "
                    "ForEach-Object { $_.VoiceInfo }); "
                    f"$preferred = @({preferred}); "
                    "$selected = $false; "
                    "foreach ($name in $preferred) { "
                    "$match = $installed | Where-Object { $_.Name -eq $name } | "
                    "Select-Object -First 1; "
                    "if ($null -ne $match) { $s.SelectVoice($match.Name); "
                    "$selected = $true; break } }; "
                    "if (-not $selected) { "
                    "$match = $installed | Where-Object { "
                    f"$_.Culture.Name -eq '{language}' "
                    "} | Select-Object -First 1; "
                    "if ($null -ne $match) { $s.SelectVoice($match.Name) } }; "
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
            except OSError:
                pass
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

    def __init__(self, catalog: AnnouncementCatalog | None = None) -> None:
        self.catalog = catalog

    def _announcement(
        self,
        entry_id: str,
        fallback: str,
        *,
        priority: AnnouncementPriority = AnnouncementPriority.INFORMATION,
        **payload: object,
    ) -> Announcement:
        if self.catalog is not None:
            rendered = self.catalog.render(entry_id, payload)
            if rendered is not None:
                return rendered
        return Announcement(fallback.format_map(payload), priority)

    def render(
        self, event_type: str, payload: Mapping[str, object]
    ) -> Announcement | None:
        role = self._ROLE_LABELS.get(str(payload.get("role", "")), "player")
        existing_role = self._ROLE_LABELS.get(
            str(payload.get("existing_role", "")), "another role"
        )
        if event_type == "registration_focus_changed":
            return self._announcement(
                event_type,
                "{role}, please look at the camera and press the E key.",
                role=role,
            )
        if event_type == "duplicate_face_rejected":
            return self._announcement(
                event_type,
                "This player is already registered as {existing_role}. "
                "A different player must register as {role}. "
                "Press the E key to try again.",
                priority=AnnouncementPriority.RECOVERY,
                role=role,
                existing_role=existing_role,
            )
        if event_type == "enrollment_completed":
            return self._announcement(
                event_type, "{role} registration complete.", role=role
            )
        if event_type == "roster_ready":
            return self._announcement(
                event_type, "All four roles are registered. Ready to start."
            )
        if event_type == "voice_enrollment_started":
            return self._announcement(
                event_type,
                "{role} voice enrollment. Say check, then pause.",
                role=role,
            )
        if event_type == "voice_enrollment_sample_accepted":
            sample_number = int(payload["sample_number"])
            total_samples = int(payload["total_samples"])
            if sample_number < total_samples:
                enrollment_phrases = ("check", "call", "raise")
                phrase = (
                    enrollment_phrases[sample_number - 1]
                    if sample_number <= len(enrollment_phrases)
                    else f"Phrase {sample_number}"
                )
                next_phrase = (
                    enrollment_phrases[sample_number]
                    if sample_number < len(enrollment_phrases)
                    else f"phrase {sample_number + 1}"
                )
                return self._announcement(
                    event_type,
                    "Phrase {phrase_number} accepted. "
                    "Phrase {next_phrase_number}.",
                    phrase_number=sample_number,
                    next_phrase_number=sample_number + 1,
                    phrase=phrase.capitalize(),
                    next_phrase=next_phrase,
                )
            return None
        if event_type == "voice_enrollment_retry":
            phrase_number = int(payload["phrase_number"])
            return self._announcement(
                event_type,
                "Voice sample too short. "
                "Repeat phrase {phrase_number} in one breath.",
                phrase_number=phrase_number,
            )
        if event_type == "voice_enrollment_completed":
            return self._announcement(
                event_type, "{role} voice enrollment complete.", role=role
            )
        if event_type == "voice_enrollment_cancelled":
            return self._announcement(
                event_type, "{role} voice enrollment cancelled.", role=role
            )
        if event_type == "blind_posted":
            amount = int(payload["amount_units"])
            return self._announcement(
                event_type,
                "{role} posts {amount_units}.",
                role=role,
                amount_units=amount,
            )
        if event_type == "turn_started":
            return self._announcement(
                event_type,
                "{role} to act.",
                priority=AnnouncementPriority.TURN,
                role=role,
            )
        if event_type == "action_committed":
            action_value = str(payload["action"])
            action = self._ACTION_LABELS.get(
                action_value, action_value.replace("_", " ")
            )
            amount = payload.get("amount_units")
            if amount is None:
                return self._announcement(
                    event_type,
                    "{role} {action}.",
                    role=role,
                    action=action,
                )
            return self._announcement(
                "action_committed_with_amount",
                "{role} {action} {amount_units}.",
                role=role,
                action=action,
                amount_units=amount,
            )
        if event_type == "street_started":
            street = self._STREET_LABELS.get(
                str(payload.get("street", "")), str(payload.get("street", "street"))
            )
            return self._announcement(
                event_type, "{street}.", street=street
            )
        if event_type == "hand_paused":
            return self._announcement(
                event_type,
                "The hand is paused. Operator assistance is required.",
                priority=AnnouncementPriority.RECOVERY,
            )
        if event_type == "safety_stop":
            return self._announcement(
                event_type,
                "Safety stop. Keep clear of the robot.",
                priority=AnnouncementPriority.SAFETY,
            )
        if self.catalog is not None:
            expanded = dict(payload)
            if "role" in expanded:
                expanded["role"] = role
            if "street" in expanded:
                street_value = str(expanded["street"])
                expanded["street"] = self._STREET_LABELS.get(
                    street_value, street_value.replace("_", " ").title()
                )
            if "action" in expanded:
                action_value = str(expanded["action"])
                expanded["action"] = action_value.replace("_", " ").title()
            return self.catalog.render(event_type, expanded)
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

    _REGISTRATION_ROLE_ORDER = (
        "button",
        "small_blind",
        "big_blind",
        "under_the_gun",
    )
    _RUNTIME_FEEDBACK_EVENTS = {
        "audio_link_lost": "audio_link_lost",
        "audio_link_restored": "audio_link_restored",
        "microphone_unavailable": "microphone_unavailable",
        "speech_action_pending": "action_pending_confirmation",
        "speech_action_cancelled": "action_cancelled",
        "speech_action_confirmation_expired": "action_confirmation_timeout",
        "speech_command_not_understood": "command_not_understood",
    }

    def __init__(
        self,
        path: Path | None,
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
        if kind == "registration_simulated_participant_added":
            role = str(payload.get("role", ""))
            seat = str(payload.get("seat", ""))
            if role and seat:
                self._roles_by_seat[seat] = role
        elif kind == "registration_enrolled":
            role = str(payload.get("role", ""))
            seat = str(payload.get("seat", ""))
            if role and seat:
                self._roles_by_seat[seat] = role
            self.announcer.publish("enrollment_completed", role=role)
            if bool(payload.get("speaker_enrollment_required", True)):
                self.announcer.publish("voice_enrollment_started", role=role)
        elif kind == "registration_control":
            if (
                bool(payload.get("accepted", False))
                and payload.get("intent") == "clear"
            ):
                self._roles_by_seat.clear()
                self.announcer.publish(
                    "registration_focus_changed", role="button"
                )
        elif kind == "registration_face_rejected":
            self.announcer.publish(
                "duplicate_face_rejected",
                role=payload.get("role", ""),
                existing_role=payload.get("existing_role", ""),
            )
        elif kind == "voice_enrollment_sample_accepted":
            role = self._roles_by_seat.get(str(payload.get("seat", "")), "")
            self.announcer.publish(
                "voice_enrollment_sample_accepted",
                role=role,
                sample_number=payload.get("sample_number", 0),
                total_samples=payload.get("total_samples", 0),
            )
        elif kind == "voice_enrollment_sample_rejected":
            self.announcer.publish(
                "voice_enrollment_retry",
                phrase_number=payload.get("sample_number", 1),
            )
        elif kind == "speaker_enrollment_completed":
            role = self._roles_by_seat.get(str(payload.get("seat", "")), "")
            self.announcer.publish("voice_enrollment_completed", role=role)
            self._announce_next_registration_role()
        elif kind == "speaker_enrollment_skipped":
            self._announce_next_registration_role()
        elif kind in self._RUNTIME_FEEDBACK_EVENTS:
            self.announcer.publish(
                self._RUNTIME_FEEDBACK_EVENTS[kind],
                **dict(payload),
            )

    def _announce_next_registration_role(self) -> None:
        registered_roles = set(self._roles_by_seat.values())
        next_role = next(
            (
                candidate
                for candidate in self._REGISTRATION_ROLE_ORDER
                if candidate not in registered_roles
            ),
            None,
        )
        if next_role is not None:
            self.announcer.publish("registration_focus_changed", role=next_role)

    def sync_engine(self, engine_log: EventLog) -> None:
        first_unwritten = self._engine_events_written
        super().sync_engine(engine_log)
        for event in engine_log.events[first_unwritten : self._engine_events_written]:
            self._announce_engine_event(event)

    def _announce_engine_event(self, event: HandEvent) -> None:
        if event.kind == "action_rejected":
            if event.payload.get("reason") == "illegal_action":
                self.announcer.publish("illegal_action")
            return
        if not event.accepted:
            return
        state = event.state_after
        button_value = state.get("button")
        if button_value is None:
            return
        button = Seat(str(button_value))

        def role_for(seat_value: object) -> str:
            return role_for_seat(button, Seat(str(seat_value))).value

        if event.kind == "hand_begun":
            players = state.get("players")
            if isinstance(players, Mapping):
                for seat_key in ("small_blind_seat", "big_blind_seat"):
                    seat_value = state.get(seat_key)
                    player = players.get(seat_value)
                    if seat_value is None or not isinstance(player, Mapping):
                        continue
                    self.announcer.publish(
                        "blind_posted",
                        role=role_for(seat_value),
                        amount_units=player.get("street_commit_units", 0),
                    )
            self.announcer.publish("dealing_hole_cards")
            return
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
            elif state.get("phase") == HandPhase.SETTLED.value:
                self._announce_settlement(state, role_for)
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
            reason = str(event.payload.get("reason", ""))
            if reason == "player_action_timeout":
                self.announcer.publish("action_timeout")
            elif "duplicate" in reason and "card" in reason:
                self.announcer.publish("duplicate_card")
            elif "unknown" in reason and "card" in reason:
                self.announcer.publish("card_unknown")
            elif "jam" in reason:
                self.announcer.publish("dealer_jam")
            elif "timeout" in reason and "dealer" in reason:
                self.announcer.publish("dealer_timeout")
            else:
                self.announcer.publish("hand_paused", reason=reason)
            return
        if event.kind == "hand_recovery_resumed":
            if (
                state.get("phase") == HandPhase.AWAITING_ACTION.value
                and state.get("acting_seat") is not None
            ):
                self.announcer.publish(
                    "turn_started", role=role_for(state["acting_seat"])
                )
            return
        if event.kind == "showdown_settled":
            self.announcer.publish("showdown_started")
            winners_by_pot = event.payload.get("winners_by_pot")
            if isinstance(winners_by_pot, Mapping) and any(
                isinstance(winners, list) and len(winners) > 1
                for winners in winners_by_pot.values()
            ):
                self.announcer.publish("split_pot")
            self._announce_settlement(state, role_for)
            return
        if event.kind == "operator_adjustment":
            self.announcer.publish("operator_adjustment")
            return
        if event.kind == "hand_voided":
            self.announcer.publish("hand_voided")

    def _announce_settlement(
        self,
        state: Mapping[str, object],
        role_for: Callable[[object], str],
    ) -> None:
        awards = state.get("awards")
        if isinstance(awards, Mapping):
            for seat, amount in awards.items():
                self.announcer.publish(
                    "pot_awarded",
                    winner=AnnouncementPolicy._ROLE_LABELS.get(
                        role_for(seat), "Player"
                    ),
                    amount_units=amount,
                )
        self.announcer.publish("hand_completed")
