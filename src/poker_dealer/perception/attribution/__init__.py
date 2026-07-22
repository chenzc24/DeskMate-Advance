"""Actor-bound perception primitives."""

from .config import ActorAttributionConfig
from .domain import (
    ActorBinding,
    ActorBindingState,
    AttributedActionCandidate,
    actor_binding_to_dict,
)
from .lease import ActorBindingLease
from .hands import (
    HandAttributionResult,
    HandAttributionState,
    attribute_hands_to_target,
)
from .pose import (
    LandmarkPoint,
    MediaPipePoseAdapter,
    PersonPoseEvidence,
    PoseFrameEvidence,
    PoseModelError,
    TargetPersonTrack,
    TargetPersonTracker,
)
from .speaker import (
    SessionSpeakerGallery,
    SpeakerVerificationResult,
    SpeakerVerificationState,
)

__all__ = [
    "ActorAttributionConfig",
    "ActorBinding",
    "ActorBindingLease",
    "ActorBindingState",
    "AttributedActionCandidate",
    "HandAttributionResult",
    "HandAttributionState",
    "LandmarkPoint",
    "MediaPipePoseAdapter",
    "PersonPoseEvidence",
    "PoseFrameEvidence",
    "PoseModelError",
    "SessionSpeakerGallery",
    "SpeakerVerificationResult",
    "SpeakerVerificationState",
    "TargetPersonTrack",
    "TargetPersonTracker",
    "attribute_hands_to_target",
    "actor_binding_to_dict",
]
