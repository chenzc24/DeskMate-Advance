# Chip localization training resume

- Status: in progress
- Objective: resume the interrupted YOLO11n single-class poker-chip localization run from its last complete checkpoint and continue toward the configured 60-epoch limit.
- Owned paths:
  - `runs/chip_localization/yolo11n_public_target_v2/`
  - this plan file
- Dirty/read-only paths:
  - `chip_recognition_workspace/` is an existing untracked experimental workspace; its scripts and configuration are inputs only for this resume.
  - `data/work/chips/2026-07-22-localization-public-target-v2/` is ignored training data and must not be modified or committed.
- External dependencies: project `.venv`, Ultralytics 8.4.104, Torch 2.13.0+cu130, CUDA device 0 (RTX 4060 Laptop GPU).
- Resume source: `runs/chip_localization/yolo11n_public_target_v2/weights/last.pt` after 39 complete epochs; epoch 40 was incomplete and is repeated by Ultralytics resume semantics.
- Validation: confirm the log reports resume from epoch 40 to 60, inspect final `results.csv`, and verify final `best.pt`/`last.pt` exist if training completes or early stopping activates.
- Physical-motion status: no camera, robot, GPIO, serial, or physical motion is used.
- Commit intent: do not commit datasets, run outputs, weights, or the untracked chip workspace. This plan remains uncommitted unless the user explicitly requests a repository commit.
