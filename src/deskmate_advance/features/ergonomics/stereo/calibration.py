"""Immutable, validated calibration records for the isolated stereo core."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DISTORTION_LENGTHS = {0, 4, 5, 8, 12, 14}


def _matrix(
    value: object,
    *,
    shape: tuple[int, int],
    label: str,
) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return tuple(tuple(float(item) for item in row) for row in array)


def _vector(value: object, *, length: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in {(length,), (length, 1), (1, length)}:
        raise ValueError(f"{label} must contain {length} values")
    array = array.reshape(length)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """One pinhole camera calibration.

    Image size is expressed as ``(width, height)``. Distortion coefficients use
    an OpenCV-supported layout and may be empty for an already rectified source.
    """

    camera_matrix: tuple[tuple[float, ...], ...]
    distortion_coefficients: tuple[float, ...]
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        matrix = _matrix(
            self.camera_matrix,
            shape=(3, 3),
            label="camera_matrix",
        )
        distortion_array = np.asarray(
            self.distortion_coefficients,
            dtype=np.float64,
        ).reshape(-1)
        if len(distortion_array) not in _DISTORTION_LENGTHS:
            raise ValueError(
                "distortion_coefficients must contain 0, 4, 5, 8, 12 or 14 values"
            )
        if not np.all(np.isfinite(distortion_array)):
            raise ValueError("distortion_coefficients must contain only finite values")
        if (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.image_size)
            or any(item <= 0 for item in self.image_size)
        ):
            raise ValueError("image_size must be a positive integer (width, height) tuple")
        matrix_array = np.asarray(matrix, dtype=np.float64)
        if matrix_array[0, 0] <= 0 or matrix_array[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(matrix_array[2], (0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("camera_matrix last row must be [0, 0, 1]")
        if math.isclose(float(np.linalg.det(matrix_array)), 0.0, abs_tol=1e-12):
            raise ValueError("camera_matrix must be invertible")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(
            self,
            "distortion_coefficients",
            tuple(float(item) for item in distortion_array),
        )

    def matrix_array(self) -> np.ndarray:
        return np.asarray(self.camera_matrix, dtype=np.float64)

    def distortion_array(self) -> np.ndarray:
        return np.asarray(self.distortion_coefficients, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    """Rigid right-camera pose expressed in the left-camera coordinate frame.

    The transform convention is ``X_right = R_right_from_left @ X_left +
    t_right_from_left_m``. Metric output is possible only when the translation
    is expressed in metres.
    """

    rig_id: str
    calibration_sha256: str
    left: CameraCalibration
    right: CameraCalibration
    rotation_right_from_left: tuple[tuple[float, ...], ...]
    translation_right_from_left_m: tuple[float, ...]
    schema_version: str = "deskmate.stereo-calibration/1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.rig_id, str) or not _SAFE_ID.fullmatch(self.rig_id):
            raise ValueError("rig_id must be a bounded safe identifier")
        if (
            not isinstance(self.calibration_sha256, str)
            or not _SHA256.fullmatch(self.calibration_sha256)
        ):
            raise ValueError("calibration_sha256 must be 64 lowercase hex characters")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if not isinstance(self.left, CameraCalibration) or not isinstance(
            self.right, CameraCalibration
        ):
            raise TypeError("left and right must be CameraCalibration records")
        rotation = _matrix(
            self.rotation_right_from_left,
            shape=(3, 3),
            label="rotation_right_from_left",
        )
        translation = _vector(
            self.translation_right_from_left_m,
            length=3,
            label="translation_right_from_left_m",
        )
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if not np.allclose(
            rotation_array.T @ rotation_array,
            np.eye(3),
            atol=1e-6,
        ) or not math.isclose(
            float(np.linalg.det(rotation_array)),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("rotation_right_from_left must be a proper rotation")
        if float(np.linalg.norm(translation)) <= 1e-6:
            raise ValueError("stereo baseline must be greater than one micrometre")
        object.__setattr__(self, "rotation_right_from_left", rotation)
        object.__setattr__(self, "translation_right_from_left_m", translation)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.translation_array()))

    def rotation_array(self) -> np.ndarray:
        return np.asarray(self.rotation_right_from_left, dtype=np.float64)

    def translation_array(self) -> np.ndarray:
        return np.asarray(self.translation_right_from_left_m, dtype=np.float64)

    def projection_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        left_extrinsic = np.column_stack((np.eye(3), np.zeros(3)))
        right_extrinsic = np.column_stack(
            (self.rotation_array(), self.translation_array())
        )
        return (
            self.left.matrix_array() @ left_extrinsic,
            self.right.matrix_array() @ right_extrinsic,
        )

    def camera_centers_left_frame(self) -> tuple[np.ndarray, np.ndarray]:
        left_center = np.zeros(3, dtype=np.float64)
        right_center = -self.rotation_array().T @ self.translation_array()
        return left_center, right_center
