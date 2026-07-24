from __future__ import annotations

import numpy as np
import pytest

from scripts.data.build_card_rotation_only import (
    ROTATION_ANGLES,
    rotation_homography,
)


def test_rotation_angles_include_cardinals_without_mirroring() -> None:
    assert len(ROTATION_ANGLES) == 10
    assert {0, 90, 180, 270}.issubset(ROTATION_ANGLES)
    quad = np.array(
        [[10.0, 20.0], [30.0, 20.0], [30.0, 60.0], [10.0, 60.0]],
        dtype=np.float32,
    )

    for angle in ROTATION_ANGLES:
        homography = rotation_homography(quad, angle)
        assert np.linalg.det(homography[:2, :2]) == pytest.approx(1.0)
        assert homography[2].tolist() == [0.0, 0.0, 1.0]
