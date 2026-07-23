# Pretrained Poker Labeling

## Outcome And Owned Paths

Use the pinned offline 52-class card detector to annotate the 52 operator-
provided PNG files under `data/raw/poker_label/poker new data/` in the same
five-field normalized YOLO text format as the prior dataset. Preserve the PNG
bytes and write paired text files only under
`data/raw/poker_label/poker new data/labels/`. This plan is the only tracked
owned path; raw labels and generated review artifacts remain ignored data.

## Dirty Read-Only Paths

Treat the existing dataset under `data/raw/poker_label/dataset/`, model assets,
configs, source, tests, documentation, archive and all unrelated dirty or
ignored paths as read-only. The old labels provide the class-ID convention and
filename-to-class reference only.

## External Dependencies

- Operator-provided 52-image local snapshot in
  `data/raw/poker_label/poker new data/`.
- Pinned local ONNX model and class sidecar in
  `models/assets/card_recognition/lgd-cards-gen3/`.
- Existing `configs/perception/cards_lgd_pilot.json` and project OpenCV runtime.
- No network access, runtime download, external media or identity data.

## Validation And Physical Motion

Verify the pinned asset hashes, infer every image offline, compare predicted
identity with the filename-derived class, retain two reviewed corner-glyph
boxes per image, and reject or manually correct wrong-class/missing-corner
proposals before writing labels. Validate one-to-one PNG/TXT pairing, integer
class IDs, five YOLO fields, finite normalized coordinates, positive in-image
boxes, per-file class consistency and filename agreement. Render a contact
sheet for visual review, record an immutable PNG input manifest SHA-256, run
`git diff --check`, and report scoped `git status --short --branch`. This task
does not connect to a camera or robot and authorizes no physical motion.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.

## Completed Outcome

- Ran the pinned offline ONNX detector over all 52 new PNG files. After
  normalizing the model's internal `T` rank spelling to dataset `10`, top-1
  identity agreed with every filename-derived class.
- Wrote 52 paired YOLO label files with 104 reviewed corner-glyph boxes under
  `data/raw/poker_label/poker new data/labels/`. Model outputs supplied 101 box
  locations; two rotated `8` corner proposals were class-corrected from the
  filename-confirmed identity, and three missed/merged corner boxes were
  visually adjusted.
- Final validation found exactly two boxes per image, all 52 class IDs, valid
  five-field rows, finite normalized in-image geometry, complete pairing and
  no filename/class disagreement. The validation error count is zero.
- The 52 immutable input PNG manifest SHA-256 is
  `ea0a2495cc39e86a2e585004cec15e18c8bfffc2f3de1c96b580f74e2b7afb1b`.
  The verified model SHA-256 is
  `8b767cdfed2c8e954a9134013ac3d2f2c53be048768d559675be01277a8a8fd1`.
- Generated ignored overlays, contact sheet, inference review and validation
  reports under `data/work/poker_new_final_review/`. `git diff --check` passed;
  unrelated pre-existing runtime-diagnostics work remains read-only. No camera
  or robot was connected and no physical motion occurred.
