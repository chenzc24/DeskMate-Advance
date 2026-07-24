# Poker Big Data V3 Dataset Plan

Status: completed on 2026-07-24.

## Outcome And Owned Paths

Use the frozen, non-runtime-selected Poker Dealer v2 checkpoint to propose
corner-glyph labels for the 52 project-captured images in
`data/raw/poker_label/new big poker/`. Preserve the image bytes, review all
low-confidence or class-remapped boxes, then create an ignored derived view
with exactly 1,500 local augmented images and 1,000 selected external images.

Owned tracked paths:

- `scripts/data/auto_label_card_images.py`
- `scripts/data/build_card_big_data_v3.py`
- targeted tests for those scripts
- this plan

Owned ignored paths:

- `data/raw/poker_label/new big poker/labels/`
- `data/work/poker_big_data_v3/`
- `runs/card_finetune/poker_big_data_v3_label/`

Unrelated dirty runtime/mobile-console paths are read-only.

## Inputs And Split

- Frozen label proposal model:
  `models/assets/card_recognition/poker-dealer-v2/best.pt`,
  SHA-256
  `620718e8ac16c8a3f666a7e0ac5e1f533eff3385b0f689ad6ba09cab0724b29b`.
- Class mapping:
  `models/assets/card_recognition/poker-dealer-v2/model.classes.json`.
- External converted view:
  `data/work/card_finetune_v1/external/`, backed by the pinned CC0 snapshot.

Split source images before augmentation. Assign 44 local source images to
training and eight to validation with two held-out classes per suit. Produce
1,275 local training augmentations and 225 local validation augmentations.
Select 850 source-disjoint external training images and 150 external validation
images. The combined view is therefore exactly 2,125 training and 375
validation images (85/15) with no source overlap.

## Augmentation And Label Review

Use explicit orientation bins around 0, 90, 180 and 270 degrees. Include
very-far, far, medium and near scales, perspective, illumination, glare,
shadow, MJPEG degradation and bounded blur. Model boxes are checked against the
rank/suit encoded by each filename. Any class-remapped, missing, single-box or
low-confidence proposal remains visible in a machine-readable review report;
do not guess missing boxes.

## Validation

Verify all image/label pairs, 52-class external coverage, held-out local source
isolation, finite normalized boxes, exact counts, orientation/scale summaries,
dataset YAML parsing, model/class compatibility, targeted tests,
`git diff --check`, and scoped `git status --short --branch`.

Completed result:

- 52 source images and 104 corner boxes reviewed and accepted; no source image
  is missing either corner.
- Local source split is 44 training and eight validation sources. Held-out
  classes are `2C`, `3D`, `4H`, `7D`, `AS`, `KC`, `KH`, and `KS`.
- Local derived view is 1,275 training plus 225 validation images. Both splits
  cover upright, right, inverted and left orientations and all four distance
  bins.
- External selection is 850 training plus 150 validation images. Both external
  splits cover all 52 classes and share no image path.
- Combined view is exactly 2,125 training plus 375 validation images, an exact
  85/15 split of 2,500 images.

## Physical Motion And Commit Intent

This task is offline data preparation only. It does not open a camera, save new
video, select a runtime model, connect to robot control or authorize physical
motion. Do not commit, push or publish the private/raw or derived data unless
the user separately asks.
