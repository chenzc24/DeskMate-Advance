# Part A Stereo Geometry Foundation

## Outcome And Owned Paths

Create one isolated Part A package for hardware-independent stereo geometry:
validated calibration records, undistortion, sparse triangulation, robust
screen-plane fitting, face-to-screen distance and fail-closed quality gates.

Owned paths:

- `src/deskmate_advance/features/ergonomics/stereo/`
- `tests/ergonomics/stereo/`
- `plan/chenzc24/2026-07-20-part-a-stereo-geometry/plan.md`

## Dirty Paths Left Read-Only

All existing source, configuration, tests, documentation, data, artifacts and
other plans remain read-only. Do not integrate the new package with the current
single-camera UI, camera adapter, temporal rules, shared domain records or
model manifest in this target.

## External Dependencies

- Existing project NumPy and OpenCV dependencies.
- Later robotics inputs: real camera models, synchronized capture, rigid mount,
  shared field of view and a versioned stereo calibration artifact.
- Later product inputs: face reference point, working range and acceptance
  thresholds. Numeric defaults in this package are development quality gates,
  not release claims.

No dataset, checkpoint, network service or physical hardware is required for
this target.

## Validation And Physical Motion

Use deterministic synthetic projection fixtures to verify exact distance,
camera-motion invariance, calibration rejection, synchronization rejection,
insufficient evidence, outlier screen points and reprojection quality. Run the
scoped stereo tests, the practical full Python suite, `git diff --check` and
scoped `git status --short --branch`.

No camera is opened and no physical robot motion is involved.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.
