from __future__ import annotations

import numpy as np

from poker_dealer.runtime.live_perception import (
    InteractiveOpenCVFrameSource,
    RegistrationUiState,
)


def _source() -> InteractiveOpenCVFrameSource:
    return InteractiveOpenCVFrameSource(object(), display=False)  # type: ignore[arg-type]


def test_registration_dashboard_renders_video_inside_fixed_shell() -> None:
    source = _source()
    source.set_registration_status(
        phase="capturing_face",
        role="button",
        seat="seat_a",
        completed_roles=(),
        face_samples=3,
        face_target=5,
        voice_samples=0,
        voice_target=3,
        voice_active=False,
        prompt_playing=False,
        speech_enabled=True,
        alert_title=None,
        alert_detail=None,
    )
    source.set_face_detections(
        ((220, 120, 240, 280),),
        status="FACE DETECTED  3 / 5",
    )
    camera_frame = np.full((720, 1280, 3), (60, 80, 100), dtype=np.uint8)

    display = source._render_display(camera_frame)

    assert display.shape == (720, 1280, 3)
    assert tuple(display[0, 0]) == (22, 18, 14)
    assert np.any(display[86:638, 28:872] != (34, 29, 24))
    assert np.any(display[86:638, 896:1252] != (34, 29, 24))


def test_registration_copy_exposes_every_operator_stage() -> None:
    source = _source()
    common = {
        "role": "button",
        "seat": "seat_a",
        "completed_roles": (),
        "face_samples": 0,
        "face_target": 5,
        "voice_samples": 0,
        "voice_target": 3,
        "speech_enabled": True,
        "alert_title": None,
        "alert_detail": None,
    }
    ready = RegistrationUiState(
        phase="ready_for_face",
        voice_active=False,
        prompt_playing=False,
        **common,
    )
    capturing = RegistrationUiState(
        phase="capturing_face",
        voice_active=False,
        prompt_playing=False,
        **common,
    )
    prompt = RegistrationUiState(
        phase="ready_for_face",
        voice_active=True,
        prompt_playing=True,
        **common,
    )
    recording = RegistrationUiState(
        phase="ready_for_face",
        voice_active=True,
        prompt_playing=False,
        **common,
    )

    assert source._registration_copy(ready)[:2] == (
        "READY FOR FACE",
        "Press E to begin",
    )
    assert source._registration_copy(capturing)[0] == "CAPTURING FACE"
    assert source._registration_copy(prompt)[0] == "PLAYING PROMPT"
    assert source._registration_copy(recording)[:2] == (
        "RECORDING VOICE",
        "Say CHECK",
    )


def test_duplicate_player_alert_overrides_ready_state_and_keeps_retry() -> None:
    source = _source()
    state = RegistrationUiState(
        phase="ready_for_face",
        role="small_blind",
        seat="seat_b",
        completed_roles=("button",),
        face_samples=0,
        face_target=5,
        voice_samples=0,
        voice_target=3,
        voice_active=False,
        prompt_playing=False,
        speech_enabled=True,
        alert_title="Already registered as Button",
        alert_detail="Small Blind requires a different player",
    )

    assert source._registration_copy(state)[:3] == (
        "DUPLICATE PLAYER",
        "Already registered as Button",
        "Small Blind requires a different player",
    )
