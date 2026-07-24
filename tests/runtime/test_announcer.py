from __future__ import annotations

import json
from pathlib import Path
import queue
from threading import Event

import pytest

from poker_dealer.runtime import (
    Announcement,
    AnnouncementCatalog,
    AnnouncementPolicy,
    AnnouncingRuntimeEventWriter,
    AnnouncementPriority,
    EventAnnouncer,
    SpeechPlaybackGate,
    WindowsSpeechAnnouncer,
)
from poker_dealer.domain import PlayerActionType, Seat
from poker_dealer.game import ActionRequest, HandEngine
from poker_dealer.runtime.live_perception import LivePerceptionSession


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "configs" / "runtime" / "announcements_en.json"


class RecordingPort:
    def __init__(self) -> None:
        self.items: list[Announcement] = []

    def announce(self, announcement: Announcement) -> None:
        self.items.append(announcement)


class RecordingRecognizer:
    def __init__(self) -> None:
        self.resets = 0

    def reset_window(self) -> None:
        self.resets += 1


def test_registration_and_turn_announcements_use_roles_only() -> None:
    port = RecordingPort()
    announcer = EventAnnouncer(port)
    announcer.publish("registration_focus_changed", role="small_blind")
    announcer.publish("enrollment_completed", role="small_blind")
    announcer.publish("turn_started", role="under_the_gun")
    assert [item.text for item in port.items] == [
        "Small Blind, please look at the camera and press the E key.",
        "Small Blind registration complete.",
        "Under the Gun to act.",
    ]
    assert port.items[-1].priority is AnnouncementPriority.TURN


def test_duplicate_face_rejection_announces_recoverable_retry() -> None:
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    port = RecordingPort()
    writer = AnnouncingRuntimeEventWriter(
        None, EventAnnouncer(port, AnnouncementPolicy(catalog))
    )

    writer.emit(
        "registration_face_rejected",
        observed_at_ns=1,
        payload={
            "reason": "duplicate_face",
            "role": "small_blind",
            "existing_role": "button",
        },
    )

    assert [item.text for item in port.items] == [
        "This player is already registered as Button. "
        "A different player must register as Small Blind. "
        "Press the E key to try again."
    ]
    assert port.items[0].priority is AnnouncementPriority.RECOVERY


def test_committed_action_and_recovery_have_concise_templates() -> None:
    port = RecordingPort()
    announcer = EventAnnouncer(port)
    announcer.publish("action_committed", role="button", action="calls", amount_units=2)
    announcer.publish("hand_paused", reason="unknown_card")
    assert port.items[0].text == "Button calls 2."
    assert port.items[1].priority is AnnouncementPriority.RECOVERY


def test_unknown_event_does_not_announce() -> None:
    port = RecordingPort()
    assert EventAnnouncer(port).publish("model_candidate", action="raise") is None
    assert not port.items


def test_voice_enrollment_announces_progress_retry_completion_and_cancel() -> None:
    port = RecordingPort()
    announcer = EventAnnouncer(port)
    announcer.publish("voice_enrollment_started", role="button")
    announcer.publish(
        "voice_enrollment_sample_accepted",
        role="button",
        sample_number=1,
        total_samples=3,
    )
    announcer.publish("voice_enrollment_retry", role="button", phrase_number=2)
    announcer.publish("voice_enrollment_completed", role="button")
    announcer.publish("voice_enrollment_cancelled", role="button")
    assert [item.text for item in port.items] == [
        "Button voice enrollment. Say check, then pause.",
        "Phrase 1 accepted. Phrase 2.",
        "Voice sample too short. Repeat phrase 2 in one breath.",
        "Button voice enrollment complete.",
        "Button voice enrollment cancelled.",
    ]


def test_playback_gate_holds_through_tail_guard() -> None:
    now = [1_000_000_000]
    gate = SpeechPlaybackGate(tail_guard_ms=350, clock_ns=lambda: now[0])
    gate.reserve_playback()
    assert gate.is_suppressed()
    gate.complete_playback()
    assert gate.is_suppressed()
    now[0] += 349_000_000
    assert gate.is_suppressed()
    now[0] += 1_000_000
    assert not gate.is_suppressed()


def test_live_speech_drops_blocks_across_playback_boundary() -> None:
    now = [2_000_000_000]
    gate = SpeechPlaybackGate(tail_guard_ms=100, clock_ns=lambda: now[0])
    recognizer = RecordingRecognizer()
    session = object.__new__(LivePerceptionSession)
    session._audio_queue = queue.Queue()
    session._audio_queue.put_nowait(b"echo")
    session._speech_playback_gate = gate
    session._speech_was_suppressed = False
    session._speech_recognizer = recognizer

    gate.reserve_playback()
    assert session._speech_input_suppressed(now[0])
    assert session._audio_queue.empty()
    assert recognizer.resets == 1

    gate.complete_playback()
    now[0] += 100_000_000
    session._audio_queue.put_nowait(b"stale-tail")
    assert not session._speech_input_suppressed(now[0])
    assert session._audio_queue.empty()
    assert recognizer.resets == 2


def test_runtime_writer_announces_only_committed_engine_events() -> None:
    port = RecordingPort()
    engine = HandEngine.start("announced-hand", Seat.A)
    writer = AnnouncingRuntimeEventWriter(None, EventAnnouncer(port))
    writer.sync_engine(engine.log)
    assert not port.items

    state = engine.state
    engine.apply_action(
        ActionRequest(
            "action-1",
            state.hand_id,
            state.state_version,
            state.acting_seat,  # type: ignore[arg-type]
            PlayerActionType.CALL,
        )
    )
    writer.sync_engine(engine.log)

    assert [item.text for item in port.items] == [
        "Under the Gun calls.",
        "Button to act.",
    ]


def test_runtime_writer_announces_blinds_and_dealing_from_committed_begin() -> None:
    port = RecordingPort()
    engine = HandEngine.setup_session("live-announcement", Seat.A)
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    writer = AnnouncingRuntimeEventWriter(
        None, EventAnnouncer(port, AnnouncementPolicy(catalog))
    )
    writer.sync_engine(engine.log)
    engine.begin_hand("begin")
    writer.sync_engine(engine.log)

    assert [item.text for item in port.items] == [
        "Small Blind posts 1.",
        "Big Blind posts 2.",
        "Dealing hole cards.",
    ]


def test_uncontested_settlement_announces_award_and_completion() -> None:
    port = RecordingPort()
    engine = HandEngine.start("uncontested-announcement", Seat.A)
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    writer = AnnouncingRuntimeEventWriter(
        None, EventAnnouncer(port, AnnouncementPolicy(catalog))
    )
    writer.sync_engine(engine.log)
    for index in range(3):
        state = engine.state
        engine.apply_action(
            ActionRequest(
                f"fold-{index}",
                state.hand_id,
                state.state_version,
                state.acting_seat,  # type: ignore[arg-type]
                PlayerActionType.FOLD,
            )
        )
        writer.sync_engine(engine.log)

    assert [item.text for item in port.items][-2:] == [
        "Big Blind wins 3.",
        "Hand complete.",
    ]


def test_runtime_writer_advances_audible_registration_roles() -> None:
    port = RecordingPort()
    writer = AnnouncingRuntimeEventWriter(None, EventAnnouncer(port))
    writer.emit(
        "registration_enrolled",
        observed_at_ns=1,
        payload={"seat": "seat_a", "role": "button"},
    )
    writer.emit(
        "speaker_enrollment_completed",
        observed_at_ns=2,
        payload={"seat": "seat_a"},
    )
    assert [item.text for item in port.items] == [
        "Button registration complete.",
        "Button voice enrollment. Say check, then pause.",
        "Button voice enrollment complete.",
        "Small Blind, please look at the camera and press the E key.",
    ]


def test_catalog_registration_progress_uses_fixed_english_phrases() -> None:
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    port = RecordingPort()
    writer = AnnouncingRuntimeEventWriter(
        None, EventAnnouncer(port, AnnouncementPolicy(catalog))
    )
    writer.emit(
        "registration_enrolled",
        observed_at_ns=1,
        payload={"seat": "seat_a", "role": "button"},
    )
    for sample_number in (1, 2, 3):
        writer.emit(
            "voice_enrollment_sample_accepted",
            observed_at_ns=sample_number + 1,
            payload={
                "seat": "seat_a",
                "sample_number": sample_number,
                "total_samples": 3,
            },
        )

    assert [item.text for item in port.items] == [
        "Button registration complete.",
        "Button voice enrollment. Say check, then pause.",
        "Check accepted. Say call, then pause.",
        "Call accepted. Say raise, then pause.",
    ]


def test_english_catalog_covers_runtime_and_recovery_prompts() -> None:
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    assert catalog.language == "en-US"
    assert catalog.speech_rate == 0
    assert catalog.speech_volume == 100
    assert catalog.voice_preferences[0] == "Microsoft Zira Desktop"
    assert len(catalog.entries) >= 40

    port = RecordingPort()
    announcer = EventAnnouncer(port, AnnouncementPolicy(catalog))
    announcer.publish("registration_focus_changed", role="small_blind")
    announcer.publish("action_pending_confirmation", action="raise")
    announcer.publish("audio_link_lost")

    assert [item.text for item in port.items] == [
        "Small Blind, please look at the camera and press the E key.",
        "Raise detected. Say confirm or cancel.",
        "Audio link lost. Reconnecting.",
    ]
    assert port.items[-1].priority is AnnouncementPriority.RECOVERY


def test_catalog_rejects_unknown_placeholders(tmp_path: Path) -> None:
    value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    value["entries"]["turn_started"]["text"] = "{unsupported} to act."
    path = tmp_path / "invalid-announcements.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid placeholder"):
        AnnouncementCatalog.from_json(path)


def test_windows_speech_command_prefers_catalog_english_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spoken = Event()
    scripts: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        scripts.append(command[-1])
        spoken.set()

    monkeypatch.setattr(
        "poker_dealer.runtime.announcer.subprocess.run", fake_run
    )
    monkeypatch.setattr("poker_dealer.runtime.announcer.sys.platform", "win32")
    port = WindowsSpeechAnnouncer(
        language="en-US",
        voice_preferences=("Microsoft Zira Desktop",),
    )
    try:
        port.announce(Announcement("System ready."))
        assert spoken.wait(2)
    finally:
        port.close()

    assert "Microsoft Zira Desktop" in scripts[0]
    assert "$_.Culture.Name -eq 'en-US'" in scripts[0]
    assert "$s.Rate = 0" in scripts[0]
    assert "$s.Volume = 100" in scripts[0]
    assert "$s.Speak('System ready.')" in scripts[0]


def test_runtime_speech_feedback_events_use_catalog_prompts() -> None:
    catalog = AnnouncementCatalog.from_json(CATALOG_PATH)
    port = RecordingPort()
    writer = AnnouncingRuntimeEventWriter(
        None, EventAnnouncer(port, AnnouncementPolicy(catalog))
    )
    writer.emit(
        "speech_action_pending",
        observed_at_ns=1,
        payload={"action": "raise"},
    )
    writer.emit(
        "speech_action_cancelled",
        observed_at_ns=2,
        payload={},
    )
    writer.emit(
        "speech_action_confirmation_expired",
        observed_at_ns=3,
        payload={"action": "raise"},
    )

    assert [item.text for item in port.items] == [
        "Raise detected. Say confirm or cancel.",
        "Action cancelled.",
        "Confirmation timed out. Please state your action again.",
    ]
