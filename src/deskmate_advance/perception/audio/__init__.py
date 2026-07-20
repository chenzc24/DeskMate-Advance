"""Public API for microphone capture and explicit read status handling."""

from .adapter import (
    AudioRead,
    AudioReadStatus,
    MicrophoneConfig,
    MicrophoneError,
    SoundDeviceMicrophone,
)

__all__ = [
    "AudioRead",
    "AudioReadStatus",
    "MicrophoneConfig",
    "MicrophoneError",
    "SoundDeviceMicrophone",
]
