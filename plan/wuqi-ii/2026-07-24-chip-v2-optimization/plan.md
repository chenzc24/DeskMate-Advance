# Chip V2 Localization And Denomination Optimization

## Outcome And Owned Paths

Use the user's new oblique and frontal `chip_v2(1)` capture to improve both
single-class poker-chip localization and the fixed-design denomination matcher.
The capture contains only values `10` and `20`; preserve the existing `1/5`
evidence unchanged and do not fabricate unsupported `1/5` samples. Preserve
all existing release/candidate assets and build a new development-only
comparison candidate.

External read-only source:

- `C:/Users/ASUS/xwechat_files/wxid_08cxqt3rjj2822_8f6a/msg/file/2026-07/chip_v2(1)/`

Owned ignored paths:

- `data/raw/chips/2026-07-24-chip-v2-source/`
- `data/work/chips/2026-07-24-chip-v2-optimization/`
- `runs/chip_finetune/yolo11n-localization-chip-v2-v1/`
- `runs/chip_evaluation/chip-v2-optimization-v1/`

Owned tracked paths:

- `scripts/data/build_chip_v2_optimization_view.py`
- `scripts/data/build_chip_v2_template_library.py`
- `scripts/evaluation/evaluate_chip_localization.py`
- `scripts/evaluation/evaluate_chip_v2_denomination.py`
- `scripts/evaluation/select_chip_v2_templates.py`
- `tests/data/test_build_chip_v2_optimization_view.py`
- `chip_recognition_workspace/chip_template_matcher.py`
- `chip_recognition_workspace/chip_live_value.py`
- `chip_recognition_workspace/live_chip_yolo11.py`
- `chip_recognition_workspace/chip_yolo11n_chip_v2_v1.json`
- `docs/evaluation/chip-v2-optimization.md`
- the narrow training-status gate in
  `chip_recognition_workspace/train_chip_yolo11n.py`
- this plan

Read-only existing assets:

- `models/assets/chip_recognition/las-vegas-denomination-templates-v1/`
- existing chip datasets, runs and manifests outside the owned paths above
- runtime scripts and configs outside the narrow owned changes above
- unrelated dirty or ignored workspace paths

## Data And Split Policy

1. Copy every source byte into the ignored immutable raw snapshot and record
   relative path, byte size and SHA-256 in a snapshot manifest.
2. Audit image decodability, dimensions, exact duplicates, near duplicates,
   scene/sequence naming, visible chip count, denomination coverage and whether
   any image contains unrelated or ambiguous objects.
3. Treat adjacent `v`/`vv` images from one denomination and capture mode as one
   source session. Split complete sessions/source groups before augmentation;
   no original or derived sibling may cross train and validation.
4. Use the current hard-negative-v3 model only as a proposal generator.
   Visually review every proposed localization box and correct/reject it before
   training. Empty or ambiguous frames remain explicit negatives.
5. Localization stays one class (`poker_chip`). Denomination identity remains
   a separate non-authoritative template/color observation and is not encoded
   into the YOLO localization classes.
6. Apply bounded train-only augmentations that preserve chip geometry and
   identity: moderate projective tilt, in-plane rotation, scale/translation,
   exposure/gamma/contrast, color temperature, shadow/glare, blur, sensor
   noise and JPEG artifacts. Do not mirror digits and do not synthesize new
   physical chips into validation.

## Training And Evaluation

1. Establish the untouched hard-negative-v3 baseline on the held-out source
   groups and report localization precision, recall, F1, mAP50, mAP50-95,
   false positives and confidence.
2. Fine-tune from the untouched hard-negative-v3 checkpoint at low learning
   rate. Write all checkpoints to the new ignored run directory and keep the
   result in `development` state.
3. Build a development denomination library from frontal and accepted oblique
   crops only after localization/ellipse review. Retain the existing template
   library unchanged.
4. Compare baseline and candidate denomination results per visible
   denomination, view type and tilt bin. Report rejection rate, confusion,
   color-only/digit-only disagreement and accepted accuracy. Do not hide
   unsupported denominations.
5. Run recorded replay before live DroidCam or Raspberry Pi testing. The
   digital ledger remains authoritative; no observation mutates game state.

## Validation, Motion And Commit Intent

- Run structural tests, image/label pairing checks, split/hash overlap checks,
  label-bound checks, model smoke tests, targeted evaluation and practical
  workspace tests for changed code.
- Run `git diff --check` and scoped `git status --short --branch`.
- This target performs offline image processing and optional camera replay
  only. It sends no robot command and authorizes no physical motion.
- Publication now replaces the tracked default localization weight in
  `models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt`
  and updates its development record in `models/manifest.yaml`; the user
  explicitly requested this overwrite and push on 2026-07-24.

## Completed Outcome

- Frozen 40 exact-unique source images and visually reviewed 75 chip boxes.
- Built a group-safe dataset with 2241 train, 442 validation and 460 unchanged
  test images. The independent target-camera holdout is the complete 13-image
  `chip_v2:20` sequence with 23 instances.
- Trained the 12-epoch development candidate from the immutable
  hard-negative-v3 checkpoint. Best weight SHA-256:
  `80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`.
- On the target holdout, mAP50 changed from `0.200` to `0.995`; on the original
  test set it changed from `0.927` to `0.930`, while mAP50-95 changed from
  `0.825` to `0.817`. Keep the candidate in development pending new-session
  live replay.
- Restricted denomination output to values 10 and 20. The selected template
  candidate changed the 66-oblique-instance result from 45 correct / 2 wrong /
  19 rejected to 51 correct / 1 wrong / 14 rejected. The selection used this
  capture, so a new held-out 10/20 session is still required before promotion.
- No release/default/manifest was replaced and no physical motion was run.

Validation performed:

- `python -m py_compile` on all changed chip data, evaluation and runtime
  modules: pass.
- `python -m pytest chip_recognition_workspace -q`: 39 passed.
- Scoped data/matcher tests: 30 passed.
- Full `python -m pytest -q`: 341 passed, 4 skipped, 9 unrelated existing
  failures. Failures are due to a missing face model, the current card ONNX
  being incompatible with OpenCV DNN, and `demo_stage1.py` not resolving the
  package when launched directly; none are in this target's owned paths.

Commit intent: publish the scoped chip-v2 localization/runtime work on a
dedicated branch and open a draft pull request. Keep all unrelated card
training work unstaged.
