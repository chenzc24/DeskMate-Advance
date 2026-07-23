# Poker Small-Target Augmentation Plan

## Outcome And Owned Paths

Create a reproducible derived training view from the immutable 52-card source
set under `data/raw/poker_label/poker new data/`. The view emphasizes the
small-card and small-corner-glyph conditions observed on the Raspberry Pi
MJPEG camera while preserving synchronized YOLO labels.

Owned source paths:

- `scripts/data/augment_poker_cards.py`
- `tests/data/test_augment_poker_cards.py`
- this plan

Generated, ignored output:

- `data/work/poker_new_augmented_v1/`

The raw images and labels are read-only. The generated view is training-only;
it is not represented as an independent validation or Gate dataset.

## Dirty Read-Only Paths

Preserve all pre-existing runtime diagnostics changes, README changes and the
earlier pretrained-labeling plan. Do not modify runtime, model, game, robotics,
policy or raw-data paths.

## External Dependencies

Use the repository virtual environment's existing OpenCV and NumPy packages.
Use the pinned local card class mapping and do not download models, datasets or
runtime assets. The augmentation seed, source hashes and transform metadata
must make the view reproducible.

## Validation

Run a targeted synthetic unit test, generate the full derived view, and verify:

- all 52 class IDs are present with equal image counts;
- every image has a matching YOLO label;
- every label has five fields, finite normalized coordinates and positive
  in-frame area;
- both source corner-glyph annotations survive each accepted transformation;
- manifest counts and file SHA-256 values agree with generated artifacts;
- visual contact sheets cover near, medium, far, blur, compression, lighting,
  glare, shadow and partial-occlusion cases.

Run the practical Python suite if time and environment permit, followed by
`git diff --check` and scoped `git status --short --branch`.

## Physical Motion And Privacy

This task reads existing card still images only. It does not connect to robot
control, command motion, record the live camera, persist identity data or
create face embeddings.

## Commit Intent

Do not commit, stage, push, create a branch or open a pull request. Generated
media remains ignored under `data/work/`.

## Completed Outcome

- Added a deterministic OpenCV/NumPy augmentation pipeline with Windows-safe
  Unicode image I/O, parallel generation, interruption resume, manifest hashes
  and full output validation.
- Located the card body from the two corner-glyph annotations in all 52 source
  frames. The final geometry moves and scales the card object, not the complete
  camera frame.
- Built a shared card-free table fill from all masked session frames while
  retaining each source frame outside its card region.
- Generated 10,400 balanced `960x720` training images and 10,400 labels:
  200 images and 400 corner-glyph annotations for each of 52 classes.
- Generated 6,253 far, 3,095 medium and 1,052 near samples. The view includes
  Gaussian, motion and resampling blur; MJPEG quality changes; brightness,
  contrast and channel changes; glare; shadow; noise; perspective; rotation;
  translation; and bounded partial occlusion.
- Full validation passed with 20,800 annotations, zero missing pairs, zero hash
  mismatches and zero invalid or out-of-frame YOLO boxes.
- Reviewed both the all-class far-distance contact sheet and the augmentation
  mode contact sheet. An early whole-frame scaling artifact was rejected and
  overwritten before the final validation.
- The view remains training-only. Independent Raspberry Pi sessions are still
  required for validation and for a fine-tuning admission decision.
