from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from poker_dealer.domain import (
    ActionEvidenceState,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import HandEngine
from poker_dealer.perception.actions import (
    ActionObservationContext,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeechUtteranceEvidence,
    SpeakerVerificationConfig,
    VoskSpeechRecognizer,
    fuse_action_observations,
    observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/perception/actions_speech_pilot.json"
MODEL_PATH = ROOT / "models/assets/vosk-model-small-en-us-0.15"
SPEAKER_CONFIG_PATH = ROOT / "configs/perception/speaker_verification_session.json"
SPEAKER_MODEL_PATH = ROOT / "models/assets/vosk-model-spk-0.4"


def load_config() -> SpeechPilotConfig:
    return SpeechPilotConfig.from_json(CONFIG_PATH)


def speech_evidence(
    command: str,
    timestamp_ms: int = 500,
    *,
    confidence: float | None = 0.95,
    is_final: bool = True,
) -> SpeechUtteranceEvidence:
    return SpeechUtteranceEvidence(
        window_started_at_ns=max(0, timestamp_ms - 300) * 1_000_000,
        observed_at_ns=timestamp_ms * 1_000_000,
        transcript=command,
        confidence=confidence,
        is_final=is_final,
        supporting_blocks=3,
    )


def context() -> ActionObservationContext:
    return ActionObservationContext("speech-hand", 3, Seat.B)


def test_speech_config_covers_five_actions_and_has_no_audio_storage() -> None:
    config = load_config()
    assert set(config.command_to_action.values()) == set(PlayerActionType)
    assert config.control_commands == {"cancel", "confirm"}
    assert config.model.asset_path == MODEL_PATH
    assert not config.save_audio
    assert config.seat_attribution == "state_owned_listening_window_only"
    assert "fold" in config.grammar_json()


def test_speaker_config_is_memory_only_and_hash_addressed() -> None:
    config = SpeakerVerificationConfig.from_json(SPEAKER_CONFIG_PATH)
    assert config.model.asset_path == SPEAKER_MODEL_PATH
    assert config.minimum_samples == 3
    assert config.minimum_speaker_frames == 40
    assert config.embeddings_memory_only
    assert not config.audio_saved


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="ignored speech model unavailable")
def test_vosk_asset_hash_load_and_silence_decode() -> None:
    config = load_config()
    assert config.verify_model_asset() == config.model.tree_sha256
    recognizer = VoskSpeechRecognizer(config)
    assert recognizer.accept_audio(bytes(8000), 1_000_000_000) is None
    recognizer.reset_window()
    assert recognizer.flush(1_250_000_000) is None


@pytest.mark.skipif(
    not MODEL_PATH.is_dir() or not SPEAKER_MODEL_PATH.is_dir(),
    reason="ignored speech or speaker model unavailable",
)
def test_vosk_speaker_asset_hash_and_attachment() -> None:
    config = load_config()
    speaker = SpeakerVerificationConfig.from_json(SPEAKER_CONFIG_PATH)
    assert speaker.verify_model_asset() == speaker.model.tree_sha256
    recognizer = VoskSpeechRecognizer(config, speaker)
    assert recognizer.accept_audio(bytes(8000), 1_000_000_000) is None
    recognizer.reset_window()


def test_vosk_result_keeps_normalized_speaker_vector_out_of_repr() -> None:
    recognizer = object.__new__(VoskSpeechRecognizer)
    recognizer._window_started_at_ns = 1_000_000_000
    recognizer._window_blocks = 4
    evidence = recognizer._consume_result(
        json.dumps(
            {
                "text": "call",
                "result": [{"word": "call", "conf": 0.96}],
                "spk": [3.0, 4.0],
                "spk_frames": 55,
            }
        ),
        1_500_000_000,
    )
    assert evidence is not None
    assert evidence.speaker_frames == 55
    assert evidence.speaker_embedding is not None
    assert pytest.approx(float((evidence.speaker_embedding**2).sum())) == 1.0
    assert "speaker_embedding" not in repr(evidence)


@pytest.mark.parametrize(
    ("command", "action"),
    [
        ("fold", PlayerActionType.FOLD),
        ("check", PlayerActionType.CHECK),
        ("call", PlayerActionType.CALL),
        ("bet", PlayerActionType.BET),
        ("raise", PlayerActionType.RAISE),
    ],
)
def test_each_final_command_maps_to_one_candidate(
    command: str, action: PlayerActionType
) -> None:
    observation = SpeechObservationAdapter(load_config()).process(
        speech_evidence(command), context()
    )
    assert observation.evidence_state is ActionEvidenceState.CANDIDATE
    assert observation.candidate_action is action
    assert "evidence_source:speech" in observation.quality_flags


def test_controls_low_confidence_and_unknown_never_become_candidates() -> None:
    adapter = SpeechObservationAdapter(load_config())
    cancel = adapter.process(speech_evidence("cancel", 500), context())
    low = adapter.process(speech_evidence("call", 1000, confidence=0.1), context())
    unknown = adapter.process(speech_evidence("anything else", 1500), context())
    partial = adapter.process(
        speech_evidence("ra", 2000, is_final=False), context()
    )
    assert cancel.evidence_state is ActionEvidenceState.UNKNOWN
    assert low.evidence_state is ActionEvidenceState.UNKNOWN
    assert unknown.evidence_state is ActionEvidenceState.UNKNOWN
    assert partial.evidence_state is ActionEvidenceState.ACTION_START
    assert all(
        item.candidate_action is None for item in (cancel, low, unknown, partial)
    )


def test_vosk_unknown_markers_do_not_hide_one_closed_command() -> None:
    adapter = SpeechObservationAdapter(load_config())
    recovered = adapter.process(speech_evidence("[unk] fold [unk]"), context())
    pure_unknown = adapter.process(speech_evidence("[unk][unk]", 1200), context())
    assert recovered.candidate_action is PlayerActionType.FOLD
    assert pure_unknown.candidate_action is None


def test_repeated_command_is_suppressed_during_cooldown() -> None:
    adapter = SpeechObservationAdapter(load_config())
    first = adapter.process(speech_evidence("bet", 500), context())
    repeated = adapter.process(speech_evidence("bet", 900), context())
    rearmed = adapter.process(speech_evidence("bet", 1600), context())
    assert first.evidence_state is ActionEvidenceState.CANDIDATE
    assert repeated.evidence_state is ActionEvidenceState.NO_ACTION
    assert rearmed.evidence_state is ActionEvidenceState.CANDIDATE


def test_speech_candidate_matches_schema_and_game_keeps_authority() -> None:
    engine = HandEngine.start("speech-hand", Seat.A)
    observation_context = ActionObservationContext(
        engine.state.hand_id,
        engine.state.state_version,
        engine.state.acting_seat,  # type: ignore[arg-type]
    )
    candidate = SpeechObservationAdapter(load_config()).process(
        speech_evidence("call"), observation_context
    )
    schema = json.loads(
        (ROOT / "configs/contracts/action_observation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(observation_to_dict(candidate))
    result = engine.apply_observation(candidate)
    assert result.accepted
    assert engine.state.acting_seat is Seat.A
    assert engine.log.events[-1].payload["source"] == "voice_adapter"


def test_fusion_agreement_single_source_and_conflict() -> None:
    adapter = SpeechObservationAdapter(load_config())
    speech_call = adapter.process(speech_evidence("call", 500), context())
    gesture_call = replace(
        speech_call,
        observation_id="gesture:call",
        model_version="player-action-mediapipe-canned-gesture@test",
    )
    agreed = fuse_action_observations([speech_call, gesture_call])
    assert agreed.evidence_state is ActionEvidenceState.CANDIDATE
    assert agreed.candidate_action is PlayerActionType.CALL
    assert "multimodal_agreement" in agreed.quality_flags

    single = fuse_action_observations([speech_call])
    assert single.evidence_state is ActionEvidenceState.CANDIDATE
    assert "single_modality_candidate" in single.quality_flags

    speech_raise = adapter.process(speech_evidence("raise", 1600), context())
    conflict = fuse_action_observations([gesture_call, speech_raise])
    assert conflict.evidence_state is ActionEvidenceState.AMBIGUOUS
    assert conflict.candidate_action is None


def test_fusion_rejects_different_state_contexts() -> None:
    adapter = SpeechObservationAdapter(load_config())
    first = adapter.process(speech_evidence("call", 500), context())
    other_context = ActionObservationContext("speech-hand", 4, Seat.B)
    second = SpeechObservationAdapter(load_config()).process(
        speech_evidence("call", 700), other_context
    )
    with pytest.raises(ValueError, match="share hand"):
        fuse_action_observations([first, second])
