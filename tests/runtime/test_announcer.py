from __future__ import annotations

import queue

from poker_dealer.runtime import (
    Announcement,
    AnnouncingRuntimeEventWriter,
    AnnouncementPriority,
    EventAnnouncer,
    SpeechPlaybackGate,
)
from poker_dealer.domain import PlayerActionType, Seat
from poker_dealer.game import ActionRequest, HandEngine
from poker_dealer.runtime.live_perception import LivePerceptionSession


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
        "Small Blind, please look at me and press confirm.",
        "Small Blind registration complete.",
        "Under the Gun to act.",
    ]
    assert port.items[-1].priority is AnnouncementPriority.TURN


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
        "Button voice enrollment. Read phrase one in one breath after the prompt.",
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
