# Fixed-chip denomination template matching

- Status: complete for the offline template-library and development-fit gate;
  live-camera temporal confirmation remains a separate follow-up.
- Objective: build a training-free `1`/`5`/`10`/`20` denomination template
  library from the user-provided front-view images, then classify the central
  number region after YOLO localization and circular normalization. Low-score
  or ambiguous inputs remain unknown.
- Owned paths:
  - `chip_recognition_workspace/build_chip_templates.py`
  - `chip_recognition_workspace/chip_template_matcher.py`
  - `chip_recognition_workspace/evaluate_chip_templates.py`
  - focused tests added under `chip_recognition_workspace/`
  - `data/work/chips/2026-07-23-template-matching/` (ignored derived output)
  - this plan file
- Dirty/read-only paths:
  - `data/raw/chip_templates/` contains immutable user-provided source images;
  - `data/chips/` and `data/raw/round test/` are read-only evaluation sources;
  - the existing localization checkpoint and all unrelated chip/card/identity
    artifacts remain read-only.
- External dependencies: project `.venv`, Ultralytics 8.4.104, Torch
  2.13.0+cu130, OpenCV, NumPy and the existing offline localization checkpoint.
  No OCR engine or runtime model download is used.
- Validation: verify all source images and hashes in a manifest; ensure each
  accepted template has one selected high-confidence detection and a valid
  centre-label ellipse; run synthetic geometry/matching tests; perform an
  offline confusion/rejection/latency evaluation on images not used as source
  templates before any live-camera integration.
- Outcome:
  - all 18 immutable front-view sources produced valid templates: `1`=5,
    `5`=5, `10`=4 and `20`=4;
  - the central crop uses 40% of the rectified chip diameter and the circular
    mask removes the surrounding brand text;
  - matching combines rotation-tolerant HOG number-shape evidence with a
    fixed-design ring-colour signature; colour is supporting evidence, not a
    generic denomination rule;
  - on 66 labelled development captures, localization and rectification
    processed 66/66; the conservative score/margin gate accepted 49/66 and all
    49 were correct, with the remaining 17 returned as `unknown`;
  - this is development-fit evidence from the user's existing sessions, not an
    independently held-out product accuracy claim;
  - template matching P50/P95 latency was 11.252/14.180 ms; the first-call
    maximum was 92.827 ms, while batched YOLO localization averaged 44.006 ms
    per image on the present machine;
  - focused geometry/template tests pass (`10 passed`); the practical suite is
    `292 passed, 4 skipped, 4 failed`, with all four failures caused by the
    pre-existing missing YuNet face-identity model asset, outside owned paths.
- Physical-motion status: offline perception evidence only; no camera control,
  ledger mutation, robot command, GPIO, serial output or physical motion.
- Commit intent: keep raw/derived images, weights and experimental scripts
  uncommitted unless the user explicitly requests publication.
