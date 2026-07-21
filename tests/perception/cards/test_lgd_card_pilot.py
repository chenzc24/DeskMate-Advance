from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from poker_dealer.domain import (
    CardIdentity,
    ColorSpace,
    FramePacket,
    ObservationStatus,
    Rank,
    Suit,
    VisionSlot,
)
from poker_dealer.perception.cards import (
    CardFrameEvidence,
    CardObservationPromoter,
    CardPilotConfig,
    OpenCvCardRecognitionAdapter,
    card_identity_from_code,
    card_observation_to_dict,
    crop_fixed_card_roi,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/perception/cards_lgd_pilot.json"
SCHEMA_PATH = ROOT / "configs/contracts/card_observation.schema.json"


def load_config() -> CardPilotConfig:
    return CardPilotConfig.from_json(CONFIG_PATH)


def frame(timestamp_ns: int = 1_000_000_000) -> FramePacket:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    return FramePacket(
        0,
        timestamp_ns,
        "synthetic-card",
        0,
        640,
        480,
        ColorSpace.BGR,
        30.0,
        0,
        image,
    )


class FakeNetwork:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def setInput(self, _blob: np.ndarray) -> None:
        pass

    def forward(self) -> np.ndarray:
        return self.output


def output_with(*items: tuple[int, float]) -> np.ndarray:
    output = np.zeros((1, 56, max(1, len(items))), dtype=np.float32)
    for index, (class_id, confidence) in enumerate(items):
        output[0, :4, index] = (320.0, 320.0, 120.0, 160.0)
        output[0, 4 + class_id, index] = confidence
    return output


def evidence(
    card: CardIdentity | None,
    timestamp_ns: int,
    sequence_id: int,
    confidence: float | None = 0.90,
) -> CardFrameEvidence:
    return CardFrameEvidence(
        "recorded-card",
        sequence_id,
        timestamp_ns,
        card,
        confidence if card is not None else None,
        (),
        1.0,
        () if card is not None else ("no_detection",),
    )


def test_config_pins_offline_assets_and_all_52_card_classes() -> None:
    config = load_config()
    assert config.pilot_status == "development_feasibility_only"
    assert not config.runtime_downloads
    assert not config.save_frames
    assert config.camera["backend"] == "dshow"
    assert config.max_seconds_default == 300
    assert config.verify_assets() == (
        "8b767cdfed2c8e954a9134013ac3d2f2c53be048768d559675be01277a8a8fd1",
        "8a2d7e9dacf245aca5ef5a402cb404def919e9994e9142644d80c6d6248ee038",
    )
    identities = tuple(card_identity_from_code(code) for code in config.model.class_codes)
    assert len(identities) == len(set(identities)) == 52
    assert card_identity_from_code("10H") == CardIdentity(Rank.TEN, Suit.HEARTS)
    assert card_identity_from_code("AS") == CardIdentity(Rank.ACE, Suit.SPADES)


def test_fixed_camera_roi_produces_owned_immutable_card_frame() -> None:
    source = frame()
    cropped, pixel_roi = crop_fixed_card_roi(
        source,
        load_config().fixed_roi,
        VisionSlot.BOARD_FLOP_1,
    )
    assert (pixel_roi.x, pixel_roi.y, pixel_roi.width, pixel_roi.height) == (
        160,
        48,
        320,
        384,
    )
    assert cropped.width == 320
    assert cropped.height == 384
    assert cropped.sequence_id == source.sequence_id
    assert cropped.captured_at_ns == source.captured_at_ns
    assert cropped.source_id.endswith("board_flop_1:fixed_roi")
    assert not cropped.image.flags.writeable


def test_model_output_decodes_project_card_identity() -> None:
    adapter = OpenCvCardRecognitionAdapter(load_config())
    adapter._network = FakeNetwork(output_with((39, 0.92)))  # type: ignore[assignment]
    result = adapter.analyze(frame())
    assert result.card == CardIdentity(Rank.ACE, Suit.SPADES)
    assert result.confidence is not None and result.confidence > 0.91
    assert result.quality_flags == ()
    assert len(result.detections) == 1


def test_two_corner_detections_of_one_card_produce_one_card_record() -> None:
    output = np.zeros((1, 56, 2), dtype=np.float32)
    output[0, :4, 0] = (120.0, 160.0, 60.0, 80.0)
    output[0, :4, 1] = (520.0, 400.0, 60.0, 80.0)
    output[0, 4 + 39, 0] = 0.91  # AS, first printed corner
    output[0, 4 + 39, 1] = 0.87  # AS, opposite printed corner

    adapter = OpenCvCardRecognitionAdapter(load_config())
    adapter._network = FakeNetwork(output)  # type: ignore[assignment]
    result = adapter.analyze(frame())

    assert len(result.detections) == 2
    assert result.card == CardIdentity(Rank.ACE, Suit.SPADES)
    assert result.confidence is not None and result.confidence > 0.90
    assert result.quality_flags == ("multiple_same_identity_detections",)


def test_low_confidence_and_conflicting_labels_are_unknown() -> None:
    adapter = OpenCvCardRecognitionAdapter(load_config())
    adapter._network = FakeNetwork(output_with((39, 0.45)))  # type: ignore[assignment]
    low = adapter.analyze(frame())
    assert low.card is None
    assert low.quality_flags == ("low_confidence",)

    adapter._network = FakeNetwork(  # type: ignore[assignment]
        output_with((39, 0.90), (46, 0.85))
    )
    ambiguous = adapter.analyze(frame())
    assert ambiguous.card is None
    assert ambiguous.quality_flags == ("ambiguous_card_identity",)


def test_actual_model_loads_offline_and_blank_frame_is_unknown() -> None:
    result = OpenCvCardRecognitionAdapter(load_config()).analyze(frame())
    assert result.card is None
    assert result.quality_flags == ("no_detection",)
    assert result.inference_latency_ms >= 0.0


def test_temporal_confirmation_emits_frozen_schema_only_after_three_frames() -> None:
    config = load_config()
    promoter = CardObservationPromoter(config)
    ace = CardIdentity(Rank.ACE, Suit.SPADES)
    first = promoter.process(
        VisionSlot.BOARD_FLOP_1,
        evidence(ace, 1_000_000_000, 1),
    )
    second = promoter.process(
        VisionSlot.BOARD_FLOP_1,
        evidence(ace, 1_100_000_000, 2),
    )
    third = promoter.process(
        VisionSlot.BOARD_FLOP_1,
        evidence(ace, 1_200_000_000, 3),
    )
    assert first.status is ObservationStatus.FACE_UP_UNCONFIRMED
    assert second.status is ObservationStatus.FACE_UP_UNCONFIRMED
    assert first.card is None and second.card is None
    assert third.status is ObservationStatus.CONFIRMED
    assert third.card == ace
    Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    ).validate(card_observation_to_dict(third))


def test_unknown_resets_candidate_and_duplicate_card_is_rejected() -> None:
    config = load_config()
    promoter = CardObservationPromoter(config)
    ace = CardIdentity(Rank.ACE, Suit.SPADES)
    for index, timestamp in enumerate((1_000_000_000, 1_100_000_000, 1_200_000_000)):
        confirmed = promoter.process(
            VisionSlot.BOARD_FLOP_1,
            evidence(ace, timestamp, index),
        )
    assert confirmed.status is ObservationStatus.CONFIRMED

    for index, timestamp in enumerate((1_300_000_000, 1_400_000_000, 1_500_000_000), start=3):
        duplicate = promoter.process(
            VisionSlot.BOARD_FLOP_2,
            evidence(ace, timestamp, index),
        )
    assert duplicate.status is ObservationStatus.UNKNOWN
    assert duplicate.card is None
    assert duplicate.quality_flags == (
        "duplicate_card_identity:board_flop_1",
    )

    reset = promoter.process(
        VisionSlot.BOARD_TURN,
        evidence(None, 1_600_000_000, 7, None),
    )
    assert reset.status is ObservationStatus.UNKNOWN
    assert reset.quality_flags == ("no_detection",)


def test_non_monotonic_timestamp_cannot_confirm() -> None:
    promoter = CardObservationPromoter(load_config())
    king = CardIdentity(Rank.KING, Suit.HEARTS)
    promoter.process(VisionSlot.BOARD_RIVER, evidence(king, 2_000_000_000, 1))
    result = promoter.process(
        VisionSlot.BOARD_RIVER,
        evidence(king, 1_900_000_000, 2),
    )
    assert result.status is ObservationStatus.UNKNOWN
    assert result.quality_flags == ("non_monotonic_timestamp",)
