# Chip Card Hard-Negative 20x V3

## Outcome And Scope

Retrain chip localization from the pre-card-negative chip-v2 fallback using
the same 21 face-up-card hard-negative sources, with 20 deterministic
augmentations per source plus the unchanged original. Compare the resulting
441-negative candidate against both chip-v2 and the current 126-negative
runtime model without replacing the tracked runtime asset.

Owned tracked paths:

- `scripts/data/build_chip_card_hard_negative_view.py`
- `tests/data/test_build_chip_card_hard_negative_view.py`
- `chip_recognition_workspace/chip_yolo11n_card_hard_negative_20x_v3.json`
- `docs/evaluation/chip-card-hard-negative-20x-v3.md`
- this plan

Owned ignored paths:

- `data/work/chips/2026-07-25-card-hard-negative-20x-v3/`
- `runs/chip_finetune/yolo11n-localization-card-hard-negative-20x-v3/`
- `runs/chip_evaluation/chip-card-hard-negative-20x-v3/`

Read-only inputs:

- `data/raw/chips/2026-07-25-card-hard-negative-source/`
- `data/work/chips/2026-07-24-chip-v2-optimization/dataset/`
- `models/assets/chip_recognition/yolo11n-localization-chip-v2-v1-fallback/best.pt`
- `models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt`
- all unrelated dirty card-recognition and track-line files

## Data And Training Policy

1. Keep `chip_neg` as one complete train-only sequence and `cards_neg` as one
   untouched held-out sequence.
2. Generate 20 augmentations plus the original for every one of the 21
   train-negative sources: 441 empty-label negative images.
3. Replay all 2241 previous chip training images. Preserve the 442-image
   validation, 460-image test and 13-image target-camera holdout unchanged.
4. Initialize from chip-v2 SHA-256
   `80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`,
   never from the current negative-tuned checkpoint.
5. Use a lower learning rate and short training schedule to limit
   catastrophic forgetting. Do not alter denomination recognition.

## Evaluation

Evaluate chip-v2, the current 126-negative runtime model and the new
441-negative candidate with identical settings on:

- all 36 raw card-negative images at confidence 0.05, 0.25, 0.40 and 0.50;
- the untouched 15-image card-back sequence separately;
- the unchanged 460-image chip test;
- the unchanged 13-image target-camera holdout.

Report false detections, maximum false confidence, precision, recall, F1,
mAP50 and mAP50-95. A lower negative false-positive count does not justify
replacement if chip recall or target-camera performance regresses materially.

## Safety And Commit Intent

This is offline perception training only. It does not open a camera, mutate
game state, send robot commands or authorize physical motion. Raw/private data,
derived views and runs stay ignored. After reviewing the live result, the user
explicitly requested that this checkpoint replace the tracked development
runtime weight on `main`. Commit only the owned builder/test, training config,
evaluation, plan, model manifest and replacement LFS weight; preserve all
unrelated dirty card-recognition and track-line files.

## Outcome

Training completed for 10 epochs from the pinned pre-negative chip-v2
checkpoint. The candidate weight is:

- `runs/chip_finetune/yolo11n-localization-card-hard-negative-20x-v3/weights/best.pt`
- SHA-256
  `311748e2ee5eefb332e5f7e1b4167ce399fb5861dc08cbd847292b3105a71d1d`

The 441-negative candidate eliminated all detections on the 36-image card
replay even at confidence 0.05. The current 126-negative runtime model already
eliminated all detections at the production confidence of 0.40 and retained
better test and target-camera mAP50-95. The offline recommendation therefore
remains conservative, but after live review the user explicitly selected the
441-negative checkpoint as the new development runtime weight. Full comparison
metrics and the selection caveat are recorded in
`docs/evaluation/chip-card-hard-negative-20x-v3.md`.
