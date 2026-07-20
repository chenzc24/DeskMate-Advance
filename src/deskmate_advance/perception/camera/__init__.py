"""Public API for camera capture and explicit read status handling."""

from .adapter import (
    CameraConfig,
    CameraError,
    CameraRead,
    CameraReadStatus,
    OpenCVCamera,
)

__all__ = [
    "CameraConfig",
    "CameraError",
    "CameraRead",
    "CameraReadStatus",
    "OpenCVCamera",
]
