# Poker Label Box Review

## Outcome And Owned Paths

Render the operator-provided raw poker-card YOLO labels as ordinary rectangular
box overlays so a human can review label placement before any fine-tuning.
Owned tracked paths are this plan and
`scripts/perception/render_poker_label_boxes.py`. Generated review artifacts
are written only beneath ignored
`data/work/poker_label_box_review/`; the original images and labels under
`data/raw/poker_label/dataset/` remain immutable.

## Dirty Read-Only Paths

All existing game, runtime, perception, model, config, documentation and
archive paths remain read-only. The existing Stage 2B pretrained-card-model
plan and model assets are inputs only and are not modified. Preserve unrelated
dirty and ignored files.

## External Dependencies

- Local raw dataset at `data/raw/poker_label/dataset/`, containing paired
  `images/*.png` and YOLO `labels/*.txt` files.
- Existing 52-class sidecar at
  `models/assets/card_recognition/lgd-cards-gen3/model.classes.json`.
- Local Pillow installation for rendering. No network or runtime model download
  is required.

## Validation And Physical Motion

Validate one-to-one image/label pairing, five-field YOLO rows, integer class
IDs, finite normalized coordinates, positive box dimensions, in-image box
bounds, per-file class consistency and filename-to-class agreement. Record an
immutable input-file manifest and its SHA-256 so later fine-tuning can identify
the exact reviewed snapshot. Inspect representative rendered previews, compile
the renderer, run `git diff --check`, and report scoped
`git status --short --branch`. This task neither loads a runtime model nor
authorizes robot connection or physical motion.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.

## Completed Outcome

- Rendered all 52 paired PNG/TXT items into ignored review output, plus four
  suit contact sheets and a JSON validation report.
- Parsed 102 YOLO boxes. Fifty images contain two visible corner-glyph boxes;
  `红桃3` and `红桃8` contain one box each, with their opposite corner visibly
  occluded by the holder's hand. This remains a human-review note rather than
  an automatic label error.
- All 52 filename-derived classes agree with the checked-in 52-class sidecar;
  there are no missing/orphan pairs, malformed rows, invalid class IDs,
  non-finite coordinates or out-of-image boxes.
- The 104-source-file input manifest SHA-256 is
  `82c6f8f75220d2fe46a487fab05b2eeb4b7651ca042ae287f08034f18215d149`.
- Renderer compilation, full dataset rendering and `git diff --check` passed.
  No model was trained or loaded, no robot was connected, and no physical
  motion occurred.
