# Chip localization hard-negative fine-tune

- Status: completed as a development candidate; not admitted as a release.
- Objective: continue the existing single-class YOLO11n localization checkpoint
  on the 53 target-camera hard-negative frames, reducing false detections
  without replacing the current working checkpoint or changing denomination
  recognition.
- Owned paths:
  - `chip_recognition_workspace/build_chip_hard_negative_dataset.py`
  - `chip_recognition_workspace/evaluate_chip_negative_images.py`
  - `chip_recognition_workspace/test_build_chip_hard_negative_dataset.py`
  - `chip_recognition_workspace/train_chip_yolo11n.py` (new allowed development
    status only)
  - `chip_recognition_workspace/chip_yolo11n_hard_negative_v3.json`
  - `data/work/chips/2026-07-23-localization-hard-negative-v1/`
  - `runs/chip_localization/yolo11n_public_target_v3_hard_negative/`
  - this plan file
- Dirty/read-only paths:
  - `data/chips/negtive/` contains the user's 53 immutable raw captures;
  - `data/work/chips/2026-07-22-localization-public-target-v2/` is the immutable
    base snapshot;
  - `runs/chip_localization/yolo11n_public_target_v2/weights/best.pt` is the
    immutable base checkpoint;
  - card, face, template, game and unrelated dirty paths remain untouched.
- External dependencies: project `.venv`, pinned Ultralytics/Torch, CUDA device
  0 (RTX 4060 Laptop GPU), existing public/target localization snapshot and
  checkpoint. No runtime download is permitted.
- Split/weighting policy: preserve every base split exactly. The complete
  negative capture session remains train-only. Six training entries per unique
  negative provide about 15% hard-negative exposure; repeats are recorded as
  controlled exposure, never represented as unique or held-out evidence.
- Validation: builder unit test; manifest and YAML/hash validation; baseline
  versus candidate false-positive frames/boxes on all 53 fit negatives; compare
  original public validation/test localization metrics; focused tests,
  `git diff --check`, and scoped status.
- Outcome:
  - derived dataset manifest SHA-256:
    `930df9e122ae7c5a311b03adea80521ed266b4b144758a90f4ec896e86fd6457`;
  - training completed for 15 epochs; epoch 13 produced the selected
    `best.pt`, SHA-256
    `7ab3a870bf6865127e3c65fc6c8e771a3c990aed94e97f6977bfdba1ac09e3c3`;
  - unchanged public validation split: candidate precision `0.98065`, recall
    `0.96350`, mAP50 `0.98758`, mAP50-95 `0.88786`; previous checkpoint
    mAP50-95 was `0.86760`;
  - unchanged public test split, previous -> candidate: precision
    `0.94686 -> 0.94722`, recall `0.88309 -> 0.87913`, mAP50
    `0.94794 -> 0.92734`, and mAP50-95 `0.81152 -> 0.82508`;
  - on the 53 training negatives at confidence `0.25`, previous -> candidate:
    frames with detections `20 -> 0`, detections `31 -> 0`, and maximum
    confidence `0.86016 -> none`. This is resubstitution/fit evidence only and
    is not an independent false-positive estimate;
  - all 35 chip-workspace tests pass. The practical full suite is 292 passed,
    4 skipped and 4 unrelated failures because the YuNet face model asset is
    absent.
- Physical-motion status: offline data preparation, training and evaluation
  only. No camera, robot, GPIO, serial, ledger or game-state mutation.
- Commit intent: the user requested publishing the chip-development program
  after synchronizing `origin/main`. Commit source, configs, tests and plans;
  exclude raw/private captures, derived data, runs, caches, third-party
  checkpoints and the unadmitted candidate weight.
