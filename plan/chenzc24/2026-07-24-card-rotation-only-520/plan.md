# Card Rotation-Only 520 Augmentation Plan

Status: completed on 2026-07-24.

## Outcome And Owned Paths

Create exactly 520 training-only derived images from the 52 reviewed
target-camera source images: ten deterministic rotations per class. Rotate only
the single extracted card in its original scene. Do not mirror, translate,
scale, perspective-warp, recolour, blur, relight, occlude, duplicate or remove
the card.

Owned tracked paths:

- `scripts/data/build_card_rotation_only.py`;
- `tests/data/test_build_card_rotation_only.py`;
- this plan.

Owned ignored paths:

- `data/work/poker_big_data_v3/rotation_only_520/`.

All unrelated dirty paths and the frozen V3 model are read-only.

## Inputs And Transform Contract

- Source images: `data/raw/poker_label/new big poker/`, 52 images.
- Source labels: `data/raw/poker_label/new big poker/labels/`, two reviewed
  corner-glyph boxes per source image.
- Class order:
  `models/assets/card_recognition/poker-dealer-v3/model.classes.json`.
- Rotation angles per source:
  `0, 30, 60, 90, 135, 180, 225, 270, 300, 330` degrees.

The homography must have positive determinant and unit scale. The card centre
must remain fixed, both corner boxes must be transformed with the card, and
every output record must declare exactly one source card and one output card.
Do not translate a card to hide natural source-frame edge cropping: both
corner-glyph boxes must remain fully valid and at least 90% of the card body
must remain visible.

## Validation

Verify exactly 520 images and labels, ten variants for each of 52 classes, two
finite in-frame boxes per image, the exact angle set, no mirror flags, positive
unit transform determinants, one-card construction invariants, immutable source
hashes, contact-sheet review, targeted tests, `git diff --check`, and scoped Git
status.

Completed result:

- Generated 520 images and 520 labels with 1,040 transformed corner boxes.
- Every class has the same ten angles: 0, 30, 60, 90, 135, 180, 225, 270, 300
  and 330 degrees.
- All 520 records contain exactly one card, two same-class corner boxes, unit
  scale, zero translation, no perspective, no photometric change and no mirror.
- Eighteen variants from three edge-near sources have natural card-body edge
  cropping because the card was not translated or shrunk; the minimum visible
  card area is 98.13% and both corner boxes remain valid.
- The representative angle grid and all-class 90-degree sheet were visually
  reviewed after expanding the source-card removal mask to eliminate residual
  white card edges.

## Split, Motion And Commit Intent

This is a training-only augmentation pool and is not an independent validation
set. All ten siblings of a source must remain in the same future split. This
task performs no camera capture or robot motion and does not change the runtime
model. Do not commit or push unless separately requested.
