import math

import numpy as np
import pytest

from deskmate_advance.domain.frame import ColorSpace, FramePacket
from deskmate_advance.perception.ergonomics import (
    AudioLevelCalculator,
    LuminanceCalculator,
    ObservationState,
)


def _frame(image: np.ndarray, color_space: ColorSpace) -> FramePacket:
    return FramePacket(
        sequence_id=7,
        captured_at_ns=123_000_000,
        source_id="fixture",
        device_index=0,
        width=image.shape[1],
        height=image.shape[0],
        color_space=color_space,
        nominal_fps=30.0,
        dropped_before=0,
        image=image,
    )


def test_luminance_uses_rgb_channel_semantics() -> None:
    bgr_red = np.array([[[0, 0, 255]]], dtype=np.uint8)

    observation = LuminanceCalculator().observe(
        _frame(bgr_red, ColorSpace.BGR)
    )

    expected = 0.2126 * 255
    assert observation.state is ObservationState.VALID
    assert observation.mean == pytest.approx(expected, abs=1e-4)
    assert observation.median == pytest.approx(expected, abs=1e-4)
    assert observation.p10 == pytest.approx(expected, abs=1e-4)
    assert observation.p90 == pytest.approx(expected, abs=1e-4)


def test_audio_level_for_float_window() -> None:
    samples = np.full((1600, 1), 0.5, dtype=np.float32)

    observation = AudioLevelCalculator().observe(
        samples,
        source_id="microphone",
        window_started_at_ns=1_000_000,
        window_ended_at_ns=101_000_000,
        sample_rate_hz=16_000,
    )

    assert observation.state is ObservationState.VALID
    assert observation.sample_count == 1600
    assert observation.rms == pytest.approx(0.5)
    assert observation.dbfs == pytest.approx(20 * math.log10(0.5))


def test_audio_level_normalizes_signed_pcm_and_floors_silence() -> None:
    calculator = AudioLevelCalculator(silence_floor_dbfs=-100)
    half_scale = np.full(100, 16384, dtype=np.int16)
    silence = np.zeros(100, dtype=np.int16)

    signal = calculator.observe(
        half_scale,
        source_id="microphone",
        window_started_at_ns=1,
        window_ended_at_ns=2,
        sample_rate_hz=16_000,
    )
    zero = calculator.observe(
        silence,
        source_id="microphone",
        window_started_at_ns=2,
        window_ended_at_ns=3,
        sample_rate_hz=16_000,
    )

    assert signal.rms == pytest.approx(0.5)
    assert zero.state is ObservationState.VALID
    assert zero.rms == 0
    assert zero.dbfs == -100


def test_audio_level_marks_empty_or_non_finite_windows() -> None:
    calculator = AudioLevelCalculator()

    empty = calculator.observe(
        np.array([], dtype=np.float32),
        source_id="microphone",
        window_started_at_ns=1,
        window_ended_at_ns=2,
        sample_rate_hz=16_000,
    )
    invalid = calculator.observe(
        np.array([np.nan], dtype=np.float32),
        source_id="microphone",
        window_started_at_ns=2,
        window_ended_at_ns=3,
        sample_rate_hz=16_000,
    )

    assert empty.state is ObservationState.MISSING
    assert empty.rms is None
    assert invalid.state is ObservationState.ERROR
    assert invalid.reason == "unsupported_or_non_finite_audio"
