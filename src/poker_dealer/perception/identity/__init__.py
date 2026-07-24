"""Session-only, consent-gated face identity verification."""

from .config import FaceIdentityConfig, FaceModelAsset
from .domain import (
    FaceIdentityContext,
    FaceIdentityObservation,
    FaceIdentityState,
    identity_observation_to_dict,
)
from .gallery import (
    DuplicateFaceEnrollmentError,
    FaceMatchResult,
    SessionFaceGallery,
)
from .opencv_adapter import (
    DetectedFaceFeature,
    FaceFrameEvidence,
    FacePreviewEvidence,
    FaceIdentityModelError,
    OpenCvFaceIdentityAdapter,
)
from .temporal import FaceIdentityTemporalAdapter

__all__ = [
    "DetectedFaceFeature",
    "DuplicateFaceEnrollmentError",
    "FaceFrameEvidence",
    "FacePreviewEvidence",
    "FaceIdentityConfig",
    "FaceIdentityContext",
    "FaceIdentityModelError",
    "FaceIdentityObservation",
    "FaceIdentityState",
    "FaceIdentityTemporalAdapter",
    "FaceMatchResult",
    "FaceModelAsset",
    "OpenCvFaceIdentityAdapter",
    "SessionFaceGallery",
    "identity_observation_to_dict",
]
