"""Sparse stereo triangulation and deterministic screen-plane fitting."""

from __future__ import annotations

from itertools import combinations
import math

import cv2
import numpy as np

from .calibration import CameraCalibration, StereoCalibration
from .types import (
    PlaneFitResult,
    ScreenPlane,
    StereoQualityConfig,
    TriangulatedPoint,
    TriangulationBatch,
)


def _point_array(points: object, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} must have shape (N, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return array


def undistort_pixel_points(
    points_xy: object,
    calibration: CameraCalibration,
) -> np.ndarray:
    """Return ideal pixel coordinates using the original camera matrix."""

    if not isinstance(calibration, CameraCalibration):
        raise TypeError("calibration must be CameraCalibration")
    points = _point_array(points_xy, "points_xy")
    if len(points) == 0:
        return points
    distortion = calibration.distortion_array()
    if len(distortion) == 0 or np.allclose(distortion, 0.0):
        return points.copy()
    result = cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        calibration.matrix_array(),
        distortion,
        P=calibration.matrix_array(),
    )
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


def _project(projection: np.ndarray, point_xyz: np.ndarray) -> np.ndarray | None:
    homogeneous = projection @ np.append(point_xyz, 1.0)
    if not np.all(np.isfinite(homogeneous)) or abs(float(homogeneous[2])) <= 1e-12:
        return None
    return homogeneous[:2] / homogeneous[2]


def _triangulation_angle_deg(
    point_xyz: np.ndarray,
    left_center: np.ndarray,
    right_center: np.ndarray,
) -> float:
    left_ray = point_xyz - left_center
    right_ray = point_xyz - right_center
    denominator = float(np.linalg.norm(left_ray) * np.linalg.norm(right_ray))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(left_ray, right_ray) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def triangulate_correspondences(
    left_points_xy: object,
    right_points_xy: object,
    calibration: StereoCalibration,
    config: StereoQualityConfig,
) -> TriangulationBatch:
    """Triangulate ordered matches and reject geometrically weak points."""

    if not isinstance(calibration, StereoCalibration):
        raise TypeError("calibration must be StereoCalibration")
    if not isinstance(config, StereoQualityConfig):
        raise TypeError("config must be StereoQualityConfig")
    left_raw = _point_array(left_points_xy, "left_points_xy")
    right_raw = _point_array(right_points_xy, "right_points_xy")
    if len(left_raw) != len(right_raw):
        raise ValueError("left/right correspondence counts must match")
    if len(left_raw) == 0:
        return TriangulationBatch(attempted=0, points=(), rejection_reasons=())
    left = undistort_pixel_points(left_raw, calibration.left)
    right = undistort_pixel_points(right_raw, calibration.right)
    left_projection, right_projection = calibration.projection_matrices()
    homogeneous = cv2.triangulatePoints(
        left_projection,
        right_projection,
        left.T,
        right.T,
    ).T
    left_center, right_center = calibration.camera_centers_left_frame()
    rotation = calibration.rotation_array()
    translation = calibration.translation_array()
    accepted: list[TriangulatedPoint] = []
    rejected: list[str] = []
    for index, point_h in enumerate(homogeneous):
        left_width, left_height = calibration.left.image_size
        right_width, right_height = calibration.right.image_size
        if not (
            0 <= left_raw[index, 0] < left_width
            and 0 <= left_raw[index, 1] < left_height
            and 0 <= right_raw[index, 0] < right_width
            and 0 <= right_raw[index, 1] < right_height
        ):
            rejected.append("point_outside_image")
            continue
        if not np.all(np.isfinite(point_h)) or abs(float(point_h[3])) <= 1e-12:
            rejected.append("invalid_homogeneous_point")
            continue
        point = point_h[:3] / point_h[3]
        point_right = rotation @ point + translation
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(point_right)):
            rejected.append("non_finite_point")
            continue
        if (
            point[2] < config.min_depth_m
            or point_right[2] < config.min_depth_m
            or point[2] > config.max_depth_m
            or point_right[2] > config.max_depth_m
        ):
            rejected.append("depth_out_of_range")
            continue
        angle = _triangulation_angle_deg(point, left_center, right_center)
        if angle < config.min_triangulation_angle_deg:
            rejected.append("triangulation_angle_too_small")
            continue
        left_reprojected = _project(left_projection, point)
        right_reprojected = _project(right_projection, point)
        if left_reprojected is None or right_reprojected is None:
            rejected.append("projection_invalid")
            continue
        left_error = float(np.linalg.norm(left_reprojected - left[index]))
        right_error = float(np.linalg.norm(right_reprojected - right[index]))
        if max(left_error, right_error) > config.max_reprojection_error_px:
            rejected.append("reprojection_error_too_high")
            continue
        accepted.append(
            TriangulatedPoint(
                source_index=index,
                point_xyz_m=tuple(float(item) for item in point),
                left_reprojection_error_px=left_error,
                right_reprojection_error_px=right_error,
                triangulation_angle_deg=angle,
            )
        )
    return TriangulationBatch(
        attempted=len(left),
        points=tuple(accepted),
        rejection_reasons=tuple(rejected),
    )


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    first, second, third = points
    normal = np.cross(second - first, third - first)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return first, normal / norm


def fit_screen_plane(
    points: tuple[TriangulatedPoint, ...],
    config: StereoQualityConfig,
) -> PlaneFitResult:
    """Fit a plane with deterministic small-set RANSAC and SVD refinement."""

    if not isinstance(config, StereoQualityConfig):
        raise TypeError("config must be StereoQualityConfig")
    if len(points) < config.min_screen_inliers:
        return PlaneFitResult(None, "insufficient_screen_points")
    coordinates = np.asarray([item.point_xyz_m for item in points], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(
        np.isfinite(coordinates)
    ):
        return PlaneFitResult(None, "invalid_screen_points")
    best_indices: np.ndarray | None = None
    best_rmse = math.inf
    for triple in combinations(range(len(points)), 3):
        candidate = _plane_from_three(coordinates[list(triple)])
        if candidate is None:
            continue
        plane_point, normal = candidate
        residuals = np.abs((coordinates - plane_point) @ normal)
        inliers = np.flatnonzero(residuals <= config.screen_inlier_threshold_m)
        if len(inliers) < config.min_screen_inliers:
            continue
        rmse = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
        if (
            best_indices is None
            or len(inliers) > len(best_indices)
            or (len(inliers) == len(best_indices) and rmse < best_rmse)
        ):
            best_indices = inliers
            best_rmse = rmse
    if best_indices is None:
        return PlaneFitResult(None, "screen_plane_consensus_unavailable")
    inlier_points = coordinates[best_indices]
    centroid = np.mean(inlier_points, axis=0)
    _, singular_values, vectors_t = np.linalg.svd(
        inlier_points - centroid,
        full_matrices=False,
    )
    if len(singular_values) < 2 or singular_values[1] < config.min_screen_span_m:
        return PlaneFitResult(None, "screen_points_degenerate")
    normal = vectors_t[-1]
    normal /= np.linalg.norm(normal)
    residuals = np.abs((inlier_points - centroid) @ normal)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    maximum = float(np.max(residuals))
    if rmse > config.max_screen_plane_rmse_m:
        return PlaneFitResult(None, "screen_plane_rmse_too_high")
    source_indices = tuple(points[index].source_index for index in best_indices)
    return PlaneFitResult(
        ScreenPlane(
            point_xyz_m=tuple(float(item) for item in centroid),
            normal_xyz=tuple(float(item) for item in normal),
            inlier_source_indices=source_indices,
            rmse_m=rmse,
            max_residual_m=maximum,
        ),
        None,
    )


def point_to_plane_distance(point_xyz: object, plane: ScreenPlane) -> float:
    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_xyz must be a finite three-vector")
    if not isinstance(plane, ScreenPlane):
        raise TypeError("plane must be ScreenPlane")
    origin = np.asarray(plane.point_xyz_m, dtype=np.float64)
    normal = np.asarray(plane.normal_xyz, dtype=np.float64)
    return abs(float(np.dot(normal, point - origin)))
