from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from poker_dealer.domain import ColorSpace, FramePacket, Seat
from poker_dealer.game import HandEngine, state_to_dict
from poker_dealer.perception.identity import (
    DetectedFaceFeature,
    FaceFrameEvidence,
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
    identity_observation_to_dict,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/perception/face_identity_session.json"
SCHEMA_PATH = ROOT / "configs/contracts/face_identity_observation.schema.json"


def load_config() -> FaceIdentityConfig:
    return FaceIdentityConfig.from_json(CONFIG_PATH)


def feature(vector: tuple[float, ...], timestamp: int = 1) -> DetectedFaceFeature:
    embedding = np.asarray(vector, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    embedding.setflags(write=False)
    return DetectedFaceFeature(timestamp, (10, 10, 100, 100), 0.99, embedding)


def frame_with(*features: DetectedFaceFeature) -> FaceFrameEvidence:
    return FaceFrameEvidence(
        observed_at_ns=max((item.observed_at_ns for item in features), default=0),
        detected_face_count=len(features),
        low_quality_face_count=0,
        features=tuple(features),
        inference_latency_ms=5.0,
    )


def enroll_two(gallery: SessionFaceGallery) -> None:
    gallery.enroll(
        "player_a", Seat.A, [feature((1, 0, 0))] * 5, consent_granted=True
    )
    gallery.enroll(
        "player_b", Seat.B, [feature((0, 1, 0))] * 5, consent_granted=True
    )


def test_config_enforces_session_only_consent_and_model_hashes() -> None:
    config = load_config()
    assert not config.save_frames
    assert not config.persist_embeddings
    assert config.clear_gallery_on_exit
    assert config.explicit_consent_required
    assert config.identity_role == "verification_only"
    assert config.verify_assets() == (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    )


def test_enrollment_requires_consent_samples_unique_player_and_seat() -> None:
    gallery = SessionFaceGallery(load_config(), "session")
    samples = [feature((1, 0, 0))] * 5
    with pytest.raises(PermissionError, match="consent"):
        gallery.enroll("player_a", Seat.A, samples, consent_granted=False)
    with pytest.raises(ValueError, match="insufficient"):
        gallery.enroll("player_a", Seat.A, samples[:2], consent_granted=True)
    gallery.enroll("player_a", Seat.A, samples, consent_granted=True)
    with pytest.raises(ValueError, match="already enrolled"):
        gallery.enroll("player_a", Seat.B, samples, consent_granted=True)
    with pytest.raises(ValueError, match="seat already"):
        gallery.enroll(
            "player_c", Seat.A, [feature((0, 0, 1))] * 5, consent_granted=True
        )
    with pytest.raises(ValueError, match="appears already enrolled"):
        gallery.enroll("player_b", Seat.B, samples, consent_granted=True)


def test_matching_accepts_clear_match_and_rejects_unknown_and_ambiguous() -> None:
    gallery = SessionFaceGallery(load_config(), "session")
    enroll_two(gallery)
    matched = gallery.match_frame(frame_with(feature((0.99, 0.05, 0))))
    assert matched.state is FaceIdentityState.MATCHED
    assert matched.player_id == "player_a"
    assert matched.registered_seat is Seat.A

    unknown = gallery.match_frame(frame_with(feature((-1, 0, 0))))
    assert unknown.state is FaceIdentityState.UNKNOWN
    assert unknown.player_id is None

    ambiguous = gallery.match_frame(frame_with(feature((1, 1, 0))))
    assert ambiguous.state is FaceIdentityState.AMBIGUOUS
    assert ambiguous.player_id is None


def test_no_face_multiple_faces_low_quality_and_empty_gallery_are_safe() -> None:
    gallery = SessionFaceGallery(load_config(), "session")
    no_face = gallery.match_frame(frame_with())
    assert no_face.state is FaceIdentityState.NO_FACE
    multiple = gallery.match_frame(
        frame_with(feature((1, 0, 0)), feature((0, 1, 0)))
    )
    assert multiple.state is FaceIdentityState.MULTIPLE_FACES
    low_quality = gallery.match_frame(
        FaceFrameEvidence(1, 1, 1, (), 2.0)
    )
    assert low_quality.state is FaceIdentityState.LOW_QUALITY
    enrollment = gallery.match_frame(frame_with(feature((1, 0, 0))))
    assert enrollment.state is FaceIdentityState.ENROLLMENT_REQUIRED


def test_temporal_match_and_seat_mismatch_are_schema_valid() -> None:
    config = load_config()
    gallery = SessionFaceGallery(config, "session")
    enroll_two(gallery)
    raw = gallery.match_frame(frame_with(feature((1, 0.01, 0))))
    adapter = FaceIdentityTemporalAdapter(config)
    context = FaceIdentityContext("session", 7, Seat.A)
    observation = None
    for timestamp_ms in (0, 100, 200, 300, 400):
        observation = adapter.process(raw, timestamp_ms * 1_000_000, context)
    assert observation is not None
    assert observation.identity_state is FaceIdentityState.MATCHED
    assert observation.player_id == "player_a"
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
        identity_observation_to_dict(observation)
    )
    serialized = identity_observation_to_dict(observation)
    assert "embedding" not in serialized

    mismatch_adapter = FaceIdentityTemporalAdapter(config)
    mismatch_context = FaceIdentityContext("session", 7, Seat.C)
    for timestamp_ms in (0, 100, 200, 300, 400):
        mismatch = mismatch_adapter.process(
            raw, timestamp_ms * 1_000_000, mismatch_context
        )
    assert mismatch.identity_state is FaceIdentityState.SEAT_MISMATCH
    assert mismatch.registered_seat is Seat.A


def test_identity_evidence_has_no_path_to_mutate_game_state() -> None:
    engine = HandEngine.start("identity-isolation", Seat.A)
    before = state_to_dict(engine.state)
    gallery = SessionFaceGallery(load_config(), "session")
    gallery.enroll(
        "player_a",
        Seat.A,
        [feature((1, 0, 0))] * 5,
        consent_granted=True,
    )
    gallery.match_frame(frame_with(feature((1, 0, 0))))
    assert state_to_dict(engine.state) == before


def test_gallery_clear_removes_all_metadata() -> None:
    gallery = SessionFaceGallery(load_config(), "session")
    gallery.enroll(
        "player_a",
        Seat.A,
        [feature((1, 0, 0))] * 5,
        consent_granted=True,
    )
    assert gallery.metadata() == (
        {"player_id": "player_a", "seat": "seat_a", "sample_count": 5},
    )
    gallery.clear()
    assert gallery.size == 0
    assert gallery.metadata() == ()


def test_official_models_load_and_blank_frame_has_no_face() -> None:
    config = load_config()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = FramePacket(
        0,
        1_000_000,
        "blank-face",
        0,
        320,
        240,
        ColorSpace.BGR,
        30.0,
        0,
        image,
    )
    evidence = OpenCvFaceIdentityAdapter(config).analyze(frame)
    assert evidence.detected_face_count == 0
    assert evidence.features == ()
