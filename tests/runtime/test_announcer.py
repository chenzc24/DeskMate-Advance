from __future__ import annotations

from poker_dealer.runtime import (
    Announcement,
    AnnouncementPriority,
    EventAnnouncer,
)


class RecordingPort:
    def __init__(self) -> None:
        self.items: list[Announcement] = []

    def announce(self, announcement: Announcement) -> None:
        self.items.append(announcement)


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
