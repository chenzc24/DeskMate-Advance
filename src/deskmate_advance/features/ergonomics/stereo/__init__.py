"""Isolated, hardware-independent stereo screen-distance geometry.

This package deliberately accepts only calibration records and sparse pixel
correspondences. Camera handles, frames, MediaPipe results and temporal event
logic remain outside its boundary.
"""

from .calibration import CameraCalibration, StereoCalibration
from .estimator import StereoScreenDistanceEstimator
from .geometry import (
    fit_screen_plane,
    point_to_plane_distance,
    triangulate_correspondences,
    undistort_pixel_points,
)
from .types import (
    PlaneFitResult,
    ScreenPlane,
    StereoCorrespondences,
    StereoDistanceEstimate,
    StereoEstimateState,
    StereoQualityConfig,
    TriangulatedPoint,
    TriangulationBatch,
)

__all__ = [
    "CameraCalibration",
    "PlaneFitResult",
    "ScreenPlane",
    "StereoCalibration",
    "StereoCorrespondences",
    "StereoDistanceEstimate",
    "StereoEstimateState",
    "StereoQualityConfig",
    "StereoScreenDistanceEstimator",
    "TriangulatedPoint",
    "TriangulationBatch",
    "fit_screen_plane",
    "point_to_plane_distance",
    "triangulate_correspondences",
    "undistort_pixel_points",
]
