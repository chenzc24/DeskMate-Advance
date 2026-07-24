# Track-Line Vision Prototype

## Outcome And Owned Paths

Build an independent laptop-side OpenCV prototype that detects a high-contrast
guide line from an image, video file, or ordinary computer camera. The module
outputs bounded perception evidence only: lateral offset, heading, curvature,
confidence, loss state, source points, frame sequence and monotonic timestamp.
It does not write GPIO, motor PWM, serial commands or game transitions.

Owned tracked path:

- this plan
- `src/track_line/`
- `tests/track_line/test_detector.py`
- the `track_line` package-data entry in `pyproject.toml`
- removal of the temporary `src/track_line/` rule from `.gitignore`

Read-only dirty paths:

- all existing card-recognition model, training, plan and test changes
- all model assets and runtime code outside `src/track_line/`

External dependencies:

- project-pinned NumPy and OpenCV packages
- a future hardware-team camera mount and motor-controller contract

## Implementation And Validation

1. Add immutable observation and validated detector configuration types.
2. Implement three-band ROI extraction, Otsu thresholding, morphology,
   component filtering, continuity-aware line selection and confidence.
3. Add annotated debug rendering without changing the detector result.
4. Add one CLI for laptop camera, video and still-image inputs.
5. Add deterministic synthetic tests for centred, displaced, curved and
   missing dark lines.
6. Run Python compilation, unit tests, a headless CLI smoke test,
   `git diff --check`, scoped status and ignore verification.

## Physical Motion And Commit Intent

This target is perception-only. It authorizes no physical motion and does not
connect to a motor adapter. Recorded/synthetic replay must pass before future
low-speed, operator-supervised hardware work.

The user requested repository integration on 2026-07-24. Commit and publish
only the scoped line-tracking module, this plan and the ignore-rule removal;
keep all unrelated card-recognition work unstaged.

## Completed Outcome

- Added a validated JSON configuration and immutable `LineObservation`
  contract.
- Added dark/light line segmentation, three-band continuity-aware component
  selection, normalized offset/heading/curvature, confidence and explicit
  no-guess line-loss output.
- Added debug overlays and one CLI supporting computer camera indices, videos,
  still images, stream URLs, headless execution and optional annotated output.
- Changed the default target to a white line on green cloth: low-saturation,
  high-value HSV candidates are accepted only inside a substantial green-floor
  hull. The original grayscale dark/light mode remains an explicit fallback.
- Added six deterministic synthetic tests covering centred, displaced, curved,
  absent and inverse-polarity lines plus invalid input.
- Python compilation passed; all six tests passed; CLI help and a synthetic
  curved-line smoke inference passed.
- After green-cloth support and repository packaging, all nine scoped tests
  pass and the supplied no-line cloth image is correctly rejected.
- The practical full suite reports 416 passed and 8 unrelated existing
  failures: missing Vosk runtime dependency, the current card ONNX/OpenCV-DNN
  incompatibility, the direct `demo_stage1.py` import path, and the resulting
  Part A preflight failure.
- No camera, GPIO, serial adapter, motor command or physical motion was used.
- Publication intent: push a dedicated branch, open a pull request, merge it
  into `main`, and verify the module from a clean checkout.
