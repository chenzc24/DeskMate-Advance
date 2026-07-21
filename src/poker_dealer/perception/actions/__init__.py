"""State-focused player-action perception for Stage 2A."""

from .config import (
    GestureConfirmationConfig,
    GestureModelConfig,
    GesturePilotConfig,
    NormalizedRoi,
)
from .mediapipe_adapter import (
    GestureModelError,
    MediaPipeGestureAdapter,
)
from .fusion import fuse_action_observations
from .speech import (
    SpeechModelConfig,
    SpeechModelError,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeechUtteranceEvidence,
    VoskSpeechRecognizer,
)
from .seats import (
    MultiSeatGesturePilotConfig,
    SeatRoiRouter,
    SeatRoutingResult,
)
from .temporal import (
    ActionObservationContext,
    GestureFrameEvidence,
    GestureTemporalAdapter,
    observation_to_dict,
)

__all__ = [
    "ActionObservationContext",
    "GestureConfirmationConfig",
    "GestureFrameEvidence",
    "GestureModelConfig",
    "GestureModelError",
    "GesturePilotConfig",
    "GestureTemporalAdapter",
    "MediaPipeGestureAdapter",
    "MultiSeatGesturePilotConfig",
    "NormalizedRoi",
    "SpeechModelConfig",
    "SpeechModelError",
    "SpeechObservationAdapter",
    "SpeechPilotConfig",
    "SpeechUtteranceEvidence",
    "SeatRoiRouter",
    "SeatRoutingResult",
    "VoskSpeechRecognizer",
    "fuse_action_observations",
    "observation_to_dict",
]
