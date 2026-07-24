# Poker Dealer Card Model V3 Training Plan

Status: completed on 2026-07-24.

## Outcome And Owned Paths

Train Poker Dealer card detector V3 from the pinned original
`lgd-cards-gen3/best.pt`, not from Poker Dealer V2, using the source-isolated
2,500-image V3 dataset view.

Owned tracked paths:

- this plan;
- `models/manifest.yaml`;
- compact V3 model metadata and evaluation evidence under
  `models/assets/card_recognition/poker-dealer-v3/`.

Owned ignored paths:

- `runs/card_finetune/retrain_v3_big_mixed_30ep/`;
- `runs/card_finetune/retrain_v3_big_mixed_30ep.launch.json`;
- validation/evaluation runs for the original, V2 and V3 checkpoints.

All unrelated dirty runtime/mobile-console paths are read-only.

## Immutable Inputs

- Original training-format checkpoint:
  `models/assets/card_recognition/lgd-cards-gen3/best.pt`, SHA-256
  `3513b77ab418ac2e86d3bf53be6566522666e656cbccb12678ba93feb6201004`.
- Dataset entry point:
  `data/work/poker_big_data_v3/dataset/dataset.yaml`.
- Dataset manifest:
  `data/work/poker_big_data_v3/dataset/manifest.json`.
- Dataset counts: 2,125 train and 375 validation images, exact 85/15 split.
- Dataset construction keeps local augmentation siblings and selected external
  images source-disjoint across train and validation.

## Training

Use the repository training launcher with CUDA device 0, image size 960, batch
size 4, AdamW, initial learning rate 0.0003, cosine final factor 0.05,
0.5 warm-up epochs, seed 20260724, zero additional geometry/colour augmentation,
30 requested epochs, checkpoint each epoch and early-stopping patience 6.

Training must include validation every epoch. The selected checkpoint is
`best.pt` according to Ultralytics validation fitness, not the final epoch by
default.

## Evaluation And Admission

After training:

- parse the complete results history and identify the best epoch;
- report train box/class/DFL losses and validation precision, recall, mAP50 and
  mAP50-95;
- evaluate the original, V2 and candidate V3 checkpoint on the exact same V3
  validation list;
- retain per-class metrics/confusion artifacts and document known validation
  limitations;
- verify checkpoint/class compatibility and exact hashes;
- copy the selected checkpoint and class sidecar into
  `models/assets/card_recognition/poker-dealer-v3/`;
- record the candidate in `models/manifest.yaml` without changing the runtime
  release selection;
- run targeted tests, `git diff --check`, and scoped Git status.

Accuracy alone does not make V3 a release. Target-camera live testing,
unknown/rejection behaviour, duplicate handling, per-slot stability and latency
remain separate admission evidence.

Completed result:

- Training started from the original pinned checkpoint and stopped normally at
  epoch 13 after six epochs without a new validation best.
- Epoch 7 was selected with validation precision 0.9311, recall 0.9121, mAP50
  0.9740 and mAP50-95 0.9004 on 375 images and 1,020 instances.
- On that exact validation list, the original model scored mAP50-95 0.5702 and
  V2 scored 0.8765. V3 improved mAP50-95 over V2 by 2.39 percentage points but
  reduced aggregate precision by 2.64 percentage points.
- Per-class, macro per-rank/per-suit and full confusion evidence is frozen in
  `models/assets/card_recognition/poker-dealer-v3/evaluation.json`.
- The optimizer-stripped candidate checkpoint is frozen at
  `models/assets/card_recognition/poker-dealer-v3/best.pt`, SHA-256
  `c7b67016cf4db2d8cfa0e45e79454c41c37ac5245f145a9e0fe4e2abe76e9cb5`.
- V3 is recorded as a candidate and is not selected by runtime configuration.

## Physical Motion And Commit Intent

This is offline GPU training and evaluation only. It does not open cameras,
change runtime model selection, connect to robot control or authorize physical
motion. Do not commit or push unless the user separately asks.
