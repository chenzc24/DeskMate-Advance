# Chip crop and round rectification

- Status: completed for the supplied three-image offline evaluation set.
- Objective: use the existing single-class YOLO11n chip-localization checkpoint
  to crop each detected chip, fit the visible top-face ellipse, and normalize it
  to a fixed-size circular view without attempting denomination recognition.
- Owned paths:
  - `chip_recognition_workspace/rectify_chip_images.py`
  - `chip_recognition_workspace/test_rectify_chip_images.py`
  - `data/work/chips/2026-07-23-round-rectification/` (ignored derived QA output)
  - this plan file
- Dirty/read-only paths:
  - `data/raw/round test/` contains immutable user-provided evaluation images;
  - `runs/chip_localization/yolo11n_public_target_v2/weights/best.pt` is the
    ignored trained detector input and must not be modified;
  - all other existing untracked chip experiments, datasets, runs, card assets,
    and repository changes remain read-only.
- External dependencies: project `.venv`, Ultralytics 8.4.104, Torch
  2.13.0+cu130, OpenCV and CUDA device 0 when available.
- Validation: compile and CLI-help checks; run all three files under
  `data/raw/round test/`; verify one high-confidence YOLO crop per input; inspect
  the fitted top-face ellipse overlays and fixed-size rectified circular views;
  record rejection reasons instead of guessing when ellipse evidence is weak.
- Physical-motion status: offline image-only development utility; no camera,
  ledger, robot, GPIO, serial command or physical motion.
- Commit intent: keep raw/derived images, weights and the experimental chip
  workspace uncommitted unless the user explicitly requests publication.
- Validation outcome:
  - all three source images produced exactly one YOLO crop at detector
    confidence `0.895518`, `0.904669`, and `0.919002`;
  - all three fitted views passed the ellipse gate with final quality
    `0.86133`, `0.89590`, and `0.89062`;
  - the fixed centre-label ellipse is used as the coplanar reference so the
    central number disc is circular after normalization and the lower visible
    side wall does not bias its aspect ratio;
  - the most oblique image retains a small amount of physical side wall at the
    outer rim. A single RGB ellipse cannot recover full 3D thickness, but this
    area is outside the future centre-number template region;
  - the utility emits crop, segmentation mask, fitted-ellipse overlay,
    normalized view, comparison image and a machine-readable `report.json`.
  - focused synthetic geometry tests passed (`2 passed`), report/output
    assertions passed, Python compilation and CLI help passed, and
    `git diff --check` passed;
  - the practical full suite completed with `292 passed`, `4 skipped`, and four
    unrelated failures caused by missing ignored YuNet/SFace face-identity
    assets under `models/assets/face_identity/`.
