# Poker Dealer Card Model V5 Corner-Case Fine-Tune Plan

Status: completed locally on 2026-07-25; Git commit remains intentionally
pending.

## Outcome And Owned Paths

Use frozen Poker Dealer V4 only to propose and review labels for the 156
project-captured `corner_case` source images. Build a source-isolated view with
exactly 1,500 locally augmented images and 1,500 selected external images,
split 85% train and 15% validation. Fine-tune from the original LGD
`lgd-cards-gen3/best.pt` checkpoint and freeze the best validation checkpoint
as Poker Dealer V5 candidate.

Owned tracked paths:

- `scripts/data/auto_label_card_images.py`;
- `tests/data/test_auto_label_card_images.py`;
- `scripts/data/build_card_corner_v5.py`;
- `tests/data/test_build_card_corner_v5.py`;
- `scripts/data/augment_poker_cards.py` and its focused tests, limited to
  making card-body extraction tolerate the wider corner spacing in this
  target-camera capture;
- `models/manifest.yaml` and the frozen V5 asset metadata;
- model-asset Git ignore/LFS declarations if required;
- this plan.

Owned ignored paths:

- `data/work/card_corner_v5/`;
- `runs/card_finetune/finetune_v5_corner_external/`;
- V5 comparison/evaluation runs.

The currently dirty runtime, camera, mobile UI and other plan paths are
unrelated and remain read-only.

## Data And Split

- Raw local source: 156 immutable images, comprising 52 identities in each of
  `heng`, `slope1` and `slope2`.
- Labels: proposed by frozen V4 and checked against the filename identity,
  prediction confidence, box count, conflict flags and rendered review sheets.
- Split the 156 original local sources by complete capture session before
  augmentation: `heng` and `slope1` are training sources; `slope2` is the
  held-out validation session. All augmentation siblings of one source remain
  in one split.
- Local output: 1,275 train and 225 validation images.
- External output: 1,275 train and 225 validation images selected from the
  existing converted external view, with all 52 classes represented in both
  splits and no exact or perceptual-near-duplicate image crossing the split.
- Combined output: exactly 2,550 train and 450 validation images
  (85% / 15% of 3,000).
- Augmentation may rotate, scale, translate and apply bounded photometric
  degradation, but must not mirror cards or change the number of cards in an
  image.

## Training

Start from
`models/assets/card_recognition/lgd-cards-gen3/best.pt`, not V4. Use CUDA 0,
image size 960, batch 4, AdamW, deterministic seed 20260725, validation-based
best checkpoint selection and early stopping. Do not add runtime downloads.

## Evaluation And Admission

Evaluate the original LGD model, V4 and V5 on the exact same V5 validation
list. Persist aggregate, per-class, macro per-rank/per-suit and confusion
evidence. V5 remains a candidate until live target-camera testing, unknown
rejection calibration, duplicate-card checks and per-slot stability gates are
closed.

## Physical Motion And Commit Intent

This task performs offline labeling, augmentation, training and evaluation
only. It authorizes no robot motion, camera capture, face/audio persistence or
runtime selection. Do not commit or push unless separately requested.

## Completed Outcome

- V4 proposed 312 corner boxes for 156 images; all three 52-card contact
  sheets were visually reviewed and accepted without box edits.
- The final view contains 1,500 local and 1,500 external images: 2,550 train
  and 450 validation. `heng` and `slope1` are train-only source sessions;
  `slope2` is validation-only. External cross-split exact and dHash-near
  duplicate counts are zero.
- The rejected intermediate view that visually left a card remnant was
  quarantined and never used. The final contact sheets show one physical card
  per generated image, without mirroring, across all four orientation and
  distance bins.
- Training started from the original LGD checkpoint, stopped at epoch 13
  after six epochs without improvement, and selected epoch 7.
- On the common 450-image validation list, V5 achieved precision 0.94874,
  recall 0.81075, mAP50 0.91409 and mAP50-95 0.76895. It improved over V4 by
  0.54, 2.61, 3.60 and 1.28 percentage points respectively.
- V5 is frozen locally as a non-runtime-selected candidate under
  `models/assets/card_recognition/poker-dealer-v5/`. It remains blocked from
  release by class regressions, low 9D/4C recall, missing raw live replay,
  rejection/duplicate/stability evidence and offline export.
- Validation: 14 focused data tests and the full 409-test suite passed;
  manifest JSON and all declared frozen asset hashes were verified;
  `git diff --check` passed. No physical motion was performed.
