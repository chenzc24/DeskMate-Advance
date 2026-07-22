from __future__ import annotations

import numpy as np

from poker_dealer.perception.attribution import (
    SessionSpeakerGallery,
    SpeakerVerificationState,
)


def sample(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_session_speaker_gallery_matches_and_clears_in_memory() -> None:
    gallery = SessionSpeakerGallery(
        "session", minimum_samples=3, minimum_similarity=0.75, minimum_margin=0.1
    )
    gallery.enroll(
        "player_a",
        [sample((1.0, 0.0, 0.0)), sample((0.99, 0.01, 0.0)), sample((1.0, 0.02, 0.0))],
    )
    gallery.enroll(
        "player_b",
        [sample((0.0, 1.0, 0.0)), sample((0.01, 0.99, 0.0)), sample((0.0, 1.0, 0.02))],
    )
    result = gallery.match(sample((0.98, 0.04, 0.0)), speaker_frames=80)
    assert result.state is SpeakerVerificationState.MATCHED
    assert result.player_id == "player_a"
    gallery.clear()
    assert gallery.size == 0
    assert gallery.match(
        sample((1.0, 0.0, 0.0)), speaker_frames=80
    ).state is SpeakerVerificationState.ENROLLMENT_REQUIRED


def test_short_or_unknown_speaker_is_not_guessed() -> None:
    gallery = SessionSpeakerGallery("session", minimum_samples=1)
    gallery.enroll("player_a", [sample((1.0, 0.0, 0.0))])
    assert gallery.match(
        sample((1.0, 0.0, 0.0)), speaker_frames=0
    ).state is SpeakerVerificationState.INSUFFICIENT_AUDIO
    assert gallery.match(
        sample((0.0, 0.0, 1.0)), speaker_frames=100
    ).state is SpeakerVerificationState.UNKNOWN
