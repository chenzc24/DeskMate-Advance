# Laptop camera input target

- Outcome and owned paths: add one Windows/OpenCV camera adapter for the
  laptop's built-in HP True Vision FHD Camera, a device probe script, focused
  tests, and the minimum Python package metadata. Owned paths are
  `pyproject.toml`, `src/deskmate_advance/domain/frame.py`,
  `src/deskmate_advance/perception/camera/`, package `__init__.py` files,
  `scripts/runtime/probe_camera.py`, `scripts/runtime/preview_camera.py`,
  `tests/perception/test_camera.py`, and this plan.
- Dirty paths left read-only: `ADVANCE_MODEL_MACRO_PLAN.md`, `.gitignore`,
  `AGENTS.md`, `configs/`, `docs/`, and all pre-existing `plan/` targets.
- External dependencies: the laptop's HP True Vision FHD Camera and Microsoft
  USB Video driver, Python 3.11+, NumPy, OpenCV, and Windows DirectShow. The
  installed DroidCam virtual device is outside this target and is not used.
- Validation: unit tests use a fake capture device; the probe script performs a
  bounded read-only scan of camera indexes and reports negotiated properties;
  the preview displays live frames until the operator presses Q or Escape.
  Neither path records nor saves images. No controller integration or physical
  robot motion is involved.
- Commit intent: do not commit, push, create a branch, or open a PR.
