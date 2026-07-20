"""Fail-closed face-to-screen distance estimation over sparse stereo matches."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import StereoCalibration
from .geometry import (
    fit_screen_plane,
    point_to_plane_distance,
    triangulate_correspondences,
)
from .types import (
    ScreenPlane,
    StereoCorrespondences,
    StereoDistanceEstimate,
    StereoEstimateState,
    StereoQualityConfig,
)


class StereoScreenDistanceEstimator:
    """Estimate metric face-to-screen distance without camera/runtime handles."""

    def __init__(
        self,
        calibration: StereoCalibration,
        config: StereoQualityConfig | None = None,
    ) -> None:
        if not isinstance(calibration, StereoCalibration):
            raise TypeError("calibration must be StereoCalibration")
        if config is not None and not isinstance(config, StereoQualityConfig):
            raise TypeError("config must be StereoQualityConfig")
        self.calibration = calibration
        self.config = config or StereoQualityConfig()

    def estimate(self, matches: StereoCorrespondences) -> StereoDistanceEstimate:
        if not isinstance(matches, StereoCorrespondences):
            raise TypeError("matches must be StereoCorrespondences")
        if matches.synchronization_skew_ms > self.config.max_sync_skew_ms:
            return self._missing(matches, "stereo_pair_not_synchronized")
        if len(matches.face_left_xy) < self.config.min_face_points:
            return self._missing(matches, "insufficient_face_points")
        if len(matches.screen_left_xy) < self.config.min_screen_inliers:
            return self._missing(matches, "insufficient_screen_points")
        try:
            face = triangulate_correspondences(
                matches.face_left_xy,
                matches.face_right_xy,
                self.calibration,
                self.config,
            )
            if len(face.points) < self.config.min_face_points:
                return self._missing(
                    matches,
                    "face_triangulation_unavailable",
                    face_points_used=len(face.points),
                )
            screen = triangulate_correspondences(
                matches.screen_left_xy,
                matches.screen_right_xy,
                self.calibration,
                self.config,
            )
            if len(screen.points) < self.config.min_screen_inliers:
                return self._missing(
                    matches,
                    "screen_triangulation_unavailable",
                    face_points_used=len(face.points),
                    screen_points_used=len(screen.points),
                )
            plane_result = fit_screen_plane(screen.points, self.config)
            if plane_result.plane is None:
                return self._missing(
                    matches,
                    plane_result.reason or "screen_plane_unavailable",
                    face_points_used=len(face.points),
                    screen_points_used=len(screen.points),
                )
            face_point_array = np.median(
                np.asarray([item.point_xyz_m for item in face.points], dtype=np.float64),
                axis=0,
            )
            plane = self._orient_plane_toward_face(plane_result.plane, face_point_array)
            distance = point_to_plane_distance(face_point_array, plane)
            if not np.isfinite(distance):
                return self._error(
                    matches,
                    "non_finite_distance",
                    face_points_used=len(face.points),
                    screen_points_used=len(plane.inlier_source_indices),
                )
            maximum_reprojection = max(
                item.max_reprojection_error_px
                for item in (*face.points, *screen.points)
            )
            return StereoDistanceEstimate(
                state=StereoEstimateState.VALID,
                calibration_sha256=self.calibration.calibration_sha256,
                synchronization_skew_ms=matches.synchronization_skew_ms,
                distance_m=float(distance),
                face_point_xyz_m=tuple(float(item) for item in face_point_array),
                screen_plane=plane,
                face_points_used=len(face.points),
                screen_points_used=len(plane.inlier_source_indices),
                max_reprojection_error_px=float(maximum_reprojection),
                absolute_distance_claimed=True,
            )
        except (
            ArithmeticError,
            FloatingPointError,
            cv2.error,
            np.linalg.LinAlgError,
        ) as exc:
            return self._error(matches, f"stereo_geometry_error:{type(exc).__name__}")

    @staticmethod
    def _orient_plane_toward_face(
        plane: ScreenPlane,
        face_point: np.ndarray,
    ) -> ScreenPlane:
        normal = np.asarray(plane.normal_xyz, dtype=np.float64)
        origin = np.asarray(plane.point_xyz_m, dtype=np.float64)
        if float(np.dot(normal, face_point - origin)) < 0:
            normal = -normal
        return ScreenPlane(
            point_xyz_m=plane.point_xyz_m,
            normal_xyz=tuple(float(item) for item in normal),
            inlier_source_indices=plane.inlier_source_indices,
            rmse_m=plane.rmse_m,
            max_residual_m=plane.max_residual_m,
        )

    def _missing(
        self,
        matches: StereoCorrespondences,
        reason: str,
        *,
        face_points_used: int = 0,
        screen_points_used: int = 0,
    ) -> StereoDistanceEstimate:
        return self._empty(
            StereoEstimateState.MISSING,
            matches,
            reason,
            face_points_used,
            screen_points_used,
        )

    def _error(
        self,
        matches: StereoCorrespondences,
        reason: str,
        *,
        face_points_used: int = 0,
        screen_points_used: int = 0,
    ) -> StereoDistanceEstimate:
        return self._empty(
            StereoEstimateState.ERROR,
            matches,
            reason,
            face_points_used,
            screen_points_used,
        )

    def _empty(
        self,
        state: StereoEstimateState,
        matches: StereoCorrespondences,
        reason: str,
        face_points_used: int,
        screen_points_used: int,
    ) -> StereoDistanceEstimate:
        return StereoDistanceEstimate(
            state=state,
            calibration_sha256=self.calibration.calibration_sha256,
            synchronization_skew_ms=matches.synchronization_skew_ms,
            distance_m=None,
            face_point_xyz_m=None,
            screen_plane=None,
            face_points_used=face_points_used,
            screen_points_used=screen_points_used,
            max_reprojection_error_px=None,
            absolute_distance_claimed=False,
            reason=reason,
        )
