"""Card localization, normalization and rank/suit inference (Stage 2)."""

from .config import (
    CardConfirmationConfig,
    CardInferenceConfig,
    CardModelAsset,
    CardPilotConfig,
    NormalizedCardRoi,
    card_identity_from_code,
)
from .opencv_adapter import (
    CardDetection,
    CardFrameEvidence,
    CardModelError,
    OpenCvCardRecognitionAdapter,
    decode_card_detections,
)
from .roi import PixelCardRoi, crop_fixed_card_roi
from .temporal import CardObservationPromoter, card_observation_to_dict

__all__ = [
    "CardConfirmationConfig",
    "CardDetection",
    "CardFrameEvidence",
    "CardInferenceConfig",
    "CardModelAsset",
    "CardModelError",
    "CardObservationPromoter",
    "CardPilotConfig",
    "NormalizedCardRoi",
    "OpenCvCardRecognitionAdapter",
    "PixelCardRoi",
    "card_identity_from_code",
    "card_observation_to_dict",
    "crop_fixed_card_roi",
    "decode_card_detections",
]
