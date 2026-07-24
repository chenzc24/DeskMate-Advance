# Chip Rim-Colour Binary Classifier

## Outcome And Owned Paths

Add a separate development-only live mode for the user's fixed two-chip set:
blue/flesh alternating rim means value 10, and green/dark-green alternating
rim means value 20. Keep the existing colour-plus-digit-template entry point
and all existing model/template assets operational and unchanged.

Owned tracked paths:

- `chip_recognition_workspace/chip_rim_color_value.py`
- `chip_recognition_workspace/live_chip_yolo11_rim_color.py`
- `chip_recognition_workspace/test_chip_rim_color_value.py`
- `models/assets/chip_recognition/rim-colour-binary-10-20-v1/model.json`
- `models/manifest.yaml`
- `scripts/data/build_chip_rim_color_binary_model.py`
- `scripts/evaluation/evaluate_chip_v2_rim_color.py`
- `docs/evaluation/chip-rim-color-binary.md`
- the narrow value-engine-name hook in
  `chip_recognition_workspace/live_chip_yolo11.py`
- this plan

Owned ignored paths:

- `runs/chip_evaluation/chip-v2-rim-color-binary-v1/`

Read-only inputs:

- `runs/chip_finetune/yolo11n-localization-chip-v2-v1/weights/best.pt`
- `data/raw/chips/2026-07-24-chip-v2-source/`
- `data/work/chips/2026-07-24-chip-v2-optimization/`
- existing template libraries and all unrelated dirty/ignored files

## Method And Validation

1. Reuse YOLO only for single-class chip localization.
2. Fit the detected top-face ellipse and sample only its outer annulus before
   perspective warping.
3. Compare the annulus colour signature against value-10 and value-20
   prototypes only. Do not run digit matching or emit values 1/5.
4. Keep unknown/rejection for insufficient size, excessive tilt, poor ellipse
   fit or ambiguous colour. Feed accepted evidence through the existing
   per-track multi-frame confirmation.
5. Compare candidate colour libraries on the 66 reviewed oblique instances,
   record per-value errors/rejections, then perform recorded-frame smoke
   inference with the new localization checkpoint.
6. Run targeted tests, practical chip tests, `git diff --check` and scoped
   `git status --short --branch`.

This work is offline perception and an optional camera display only. It does
not connect to the robot controller, mutate the game state or authorize
physical motion. The user explicitly requested publication on 2026-07-24, so
stage and push only this scoped chip work and register the classifier as a
development model in `models/manifest.yaml`.

## Completed Outcome

- Added the separate `live_chip_yolo11_rim_color.py` entry point; the existing
  colour-plus-digit entry point and its behavior remain available.
- Corrected two visually verified derived denomination labels: the left chip
  in `mixv3` and upper-right chip in `mixv6` are value 20, not value 10.
- Built the 40D rim-pattern logistic model. JSON SHA-256:
  `c0be8c9dcbed5aaca933d5566e0d5275db3fc0a3cf6a81f0206f158ccfda52c2`.
- Final runtime-refit result on 66 reviewed oblique instances: 66 accepted,
  66 correct, zero wrong and zero rejected. This is development/in-sample
  evidence; the pre-refit complete `chip_v2:20` holdout result was 21/23.
- Direct YOLO-box annulus processing averages 3.68 ms per chip, down from the
  earlier fitted-ellipse experiment's roughly 87 ms and without digit search.
- A 30-frame Raspberry Pi MJPEG smoke run completed with zero missing reads,
  zero camera errors and the expected
  `track-best-frame-rim-colour-binary-10-20-v1` engine. No chips were present
  in those 30 frames, so this smoke validates transport/runtime, not live
  classification accuracy.
- `python -m pytest chip_recognition_workspace -q`: 42 passed.
- Changed modules compile, the new JSON parses, CLI help exits zero, and
  `git diff --check` passes.

Commit intent: publish this scoped development classifier with the chip-v2
localizer on a dedicated branch and open a draft pull request. Keep unrelated
card training work unstaged.
