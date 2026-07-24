# Poker Dealer Card Model V4 Rotation Fine-Tune Plan

Status: completed.

## Outcome And Owned Paths

Fine-tune the frozen Poker Dealer V3 candidate with source-isolated
rotation-only data while replaying the complete V3 mixed training view to limit
catastrophic forgetting.

Owned tracked paths:

- `scripts/data/build_card_v4_training_view.py`;
- `tests/data/test_build_card_v4_training_view.py`;
- `models/manifest.yaml`;
- `.gitignore` and `.gitattributes` model-asset exceptions;
- the frozen V4 metadata/evaluation files;
- this plan.

Owned ignored paths:

- `data/work/poker_big_data_v4/`;
- `runs/card_finetune/finetune_v4_rotation_replay_15ep/`;
- V3/V4 comparison evaluation runs.

All unrelated dirty paths are read-only.

## Data And Split

- Replay base: V3 mixed view with 2,125 train and 375 validation images.
- Rotation pool: 520 images, ten siblings from each of 52 source images.
- Preserve the existing local source split. All rotations of 44 training
  sources go to train (440 images); all rotations of the eight held-out sources
  `2C`, `3D`, `4H`, `7D`, `AS`, `KC`, `KH`, and `KS` go to validation
  (80 images).
- Final V4 view: 2,565 train and 455 validation images. No augmented siblings
  or selected external image paths cross the split.

## Training

Start from `models/assets/card_recognition/poker-dealer-v3/best.pt`. Use CUDA 0,
image size 960, batch 4, AdamW, initial learning rate 0.0001, cosine final
factor 0.05, 0.5 warm-up epochs, seed 20260724, zero additional geometry or
colour augmentation, 15 requested epochs and early-stopping patience 5.

Select `best.pt` by validation mAP50-95 rather than final training loss.

## Evaluation And Admission

Evaluate V3 and V4 on the exact same V4 validation list. Persist aggregate,
per-class, macro per-rank/per-suit and confusion evidence. Record hashes,
training losses, latency and known limitations in `models/manifest.yaml`.

V4 remains a candidate and must not replace runtime V2/V3 configuration without
separate live target-camera evidence and explicit selection.

## Physical Motion And Commit Intent

This task performs offline data preparation, GPU training and evaluation only.
It does not authorize camera capture or robot motion. Do not commit or push
unless separately requested.

## Result

- Training stopped at epoch 11 after five epochs without improving on the
  epoch-6 best checkpoint.
- On the exact 455-image, 1,180-instance V4 validation list, V4 achieved
  precision 0.93654, recall 0.92814, mAP50 0.97914 and mAP50-95 0.91079.
- Against V3 on the same list, V4 improved precision by 1.72, recall by 0.89
  and mAP50-95 by 1.61 percentage points.
- The frozen candidate is
  `models/assets/card_recognition/poker-dealer-v4/best.pt`; it is not selected
  for runtime use pending live robot-camera rotation and distance testing.
