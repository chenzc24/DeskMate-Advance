"""Framework-independent records for stereo screen-distance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


PixelPoint = tuple[float, float]
Point3D = tuple[float, float, float]


class StereoEstimateState(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    ERROR = "error"


def _pixel_points(value: object, label: str) -> tuple[PixelPoint, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    output: list[PixelPoint] = []
    for index, item in enumerate(value):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{label}[{index}] must contain x and y")
        x, y = item
        if isinstance(x, bool) or isinstance(y, bool):
            raise TypeError(f"{label}[{index}] coordinates must be numeric")
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError(f"{label}[{index}] coordinates must be finite")
        output.append((x_value, y_value))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class StereoQualityConfig:
    """Development quality gates, not frozen product acceptance thresholds."""

    max_sync_skew_ms: float = 5.0
    max_reprojection_error_px: float = 2.0
    min_triangulation_angle_deg: float = 0.25
    min_depth_m: float = 0.05
    max_depth_m: float = 5.0
    screen_inlier_threshold_m: float = 0.01
    max_screen_plane_rmse_m: float = 0.005
    min_screen_span_m: float = 0.03
    min_screen_inliers: int = 4
    min_face_points: int = 1

    def __post_init__(self) -> None:
        positive = (
            "max_sync_skew_ms",
            "max_reprojection_error_px",
            "min_triangulation_angle_deg",
            "min_depth_m",
            "max_depth_m",
            "screen_inlier_threshold_m",
            "max_screen_plane_rmse_m",
            "min_screen_span_m",
        )
        for field_name in positive:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{field_name} must be positive and finite")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")
        if not 0 < self.min_triangulation_angle_deg < 180:
            raise ValueError("min_triangulation_angle_deg must be between 0 and 180")
        for field_name in ("min_screen_inliers", "min_face_points"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.min_screen_inliers < 3:
            raise ValueError("min_screen_inliers must be at least 3")


@dataclass(frozen=True, slots=True)
class StereoCorrespondences:
    """Ordered raw pixel correspondences from one synchronized stereo pair."""

    face_left_xy: tuple[PixelPoint, ...]
    face_right_xy: tuple[PixelPoint, ...]
    screen_left_xy: tuple[PixelPoint, ...]
    screen_right_xy: tuple[PixelPoint, ...]
    synchronization_skew_ms: float

    def __post_init__(self) -> None:
        face_left = _pixel_points(self.face_left_xy, "face_left_xy")
        face_right = _pixel_points(self.face_right_xy, "face_right_xy")
        screen_left = _pixel_points(self.screen_left_xy, "screen_left_xy")
        screen_right = _pixel_points(self.screen_right_xy, "screen_right_xy")
        if len(face_left) != len(face_right):
            raise ValueError("left/right face correspondence counts must match")
        if len(screen_left) != len(screen_right):
            raise ValueError("left/right screen correspondence counts must match")
        if isinstance(self.synchronization_skew_ms, bool) or not isinstance(
            self.synchronization_skew_ms,
            (int, float),
        ):
            raise TypeError("synchronization_skew_ms must be numeric")
        skew = float(self.synchronization_skew_ms)
        if not math.isfinite(skew) or skew < 0:
            raise ValueError("synchronization_skew_ms must be finite and non-negative")
        object.__setattr__(self, "face_left_xy", face_left)
        object.__setattr__(self, "face_right_xy", face_right)
        object.__setattr__(self, "screen_left_xy", screen_left)
        object.__setattr__(self, "screen_right_xy", screen_right)
        object.__setattr__(self, "synchronization_skew_ms", skew)


@dataclass(frozen=True, slots=True)
class TriangulatedPoint:
    source_index: int
    point_xyz_m: Point3D
    left_reprojection_error_px: float
    right_reprojection_error_px: float
    triangulation_angle_deg: float

    @property
    def max_reprojection_error_px(self) -> float:
        return max(
            self.left_reprojection_error_px,
            self.right_reprojection_error_px,
        )


@dataclass(frozen=True, slots=True)
class TriangulationBatch:
    attempted: int
    points: tuple[TriangulatedPoint, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScreenPlane:
    point_xyz_m: Point3D
    normal_xyz: Point3D
    inlier_source_indices: tuple[int, ...]
    rmse_m: float
    max_residual_m: float


@dataclass(frozen=True, slots=True)
class PlaneFitResult:
    plane: ScreenPlane | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class StereoDistanceEstimate:
    state: StereoEstimateState
    calibration_sha256: str
    synchronization_skew_ms: float
    distance_m: float | None
    face_point_xyz_m: Point3D | None
    screen_plane: ScreenPlane | None
    face_points_used: int
    screen_points_used: int
    max_reprojection_error_px: float | None
    absolute_distance_claimed: bool
    reason: str | None = None
