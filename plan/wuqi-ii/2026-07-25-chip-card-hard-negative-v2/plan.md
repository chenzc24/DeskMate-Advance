# Chip Localization Card Hard-Negative V2

## Outcome And Owned Paths

Reduce poker-card false positives in the current chip-v2 single-class YOLO
localizer without forgetting the previously learned public chips, target-camera
10/20 chips, or earlier hard negatives. Continue from the exact tracked
chip-v2 checkpoint and preserve the complete previous train/validation/test
view as replay.

External read-only source:

- `C:/Users/ASUS/xwechat_files/wxid_08cxqt3rjj2822_8f6a/msg/file/2026-07/neg_new/neg_new/`

Owned ignored paths:

- `data/raw/chips/2026-07-25-card-hard-negative-source/`
- `data/work/chips/2026-07-25-card-hard-negative-v2/`
- `runs/chip_finetune/yolo11n-localization-card-hard-negative-v2/`
- `runs/chip_evaluation/chip-card-hard-negative-v2/`

Owned tracked paths:

- `.gitattributes`
- `.gitignore`
- `scripts/data/build_chip_card_hard_negative_view.py`
- `scripts/evaluation/evaluate_chip_negative_images.py`
- `tests/data/test_build_chip_card_hard_negative_view.py`
- `chip_recognition_workspace/chip_yolo11n_card_hard_negative_v2.json`
- `chip_recognition_workspace/live_chip_yolo11.py`
- `chip_recognition_workspace/live_chip_stream.py`
- `docs/evaluation/chip-card-hard-negative-v2.md`
- `docs/evaluation/chip-recognition-development-pilot.md`
- `models/manifest.yaml`
- `models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt`
- `models/assets/chip_recognition/yolo11n-localization-chip-v2-v1-fallback/best.pt`
- this plan

Read-only inputs:

- `models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt`
  (actual chip-v2 SHA-256
  `80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`)
- `data/work/chips/2026-07-24-chip-v2-optimization/dataset/`
- all denomination recognition assets and runtime code
- all unrelated dirty card-recognition and track-line files

## Data And Split Policy

1. Copy all 36 source bytes into an immutable ignored raw snapshot and record
   byte SHA-256, dimensions, source folder and timestamp ordering.
2. Treat `chip_neg` (21 face-up-card frames) and `cards_neg` (15 card-back
   frames) as two complete capture sequences. Keep every adjacent frame and all
   derivatives of one sequence in one split.
3. Use the complete `chip_neg` sequence for hard-negative training. Generate
   deterministic train-only geometry, exposure, blur/noise and JPEG variants;
   every image and derivative has an empty YOLO label file.
4. Keep the complete `cards_neg` sequence untouched as an independent negative
   replay. It is never used for fitting or checkpoint selection.
5. Reference the entire previous chip-v2 train directory together with the new
   negative directory. Preserve the previous validation, test and target-camera
   validation paths exactly so positive performance remains comparable.
6. No source or derived image is committed. Identify the new view by manifest
   SHA-256.

## Training And Evaluation

1. Record the current checkpoint's detections on both negative sequences at
   confidence thresholds 0.05, 0.25 and 0.50.
2. Fine-tune from the current chip-v2 checkpoint at a low learning rate. Do not
   start from stock YOLO11n and do not replace the tracked model.
3. Evaluate baseline and candidate on:
   - the untouched `cards_neg` holdout;
   - the training hard-negative sequence, explicitly labelled fit-only;
   - the unchanged previous public test split;
   - the unchanged complete chip-v2 target-camera validation sequence.
4. Admit only a development candidate. Reject it if card false positives do
   not fall materially or if prior chip recall/F1/mAP regress materially.
5. Denomination recognition remains a separate unchanged stage.

## Safety, Validation And Commit Intent

- Run source audit, exact/near-duplicate checks, empty-label checks, split
  leakage checks, Python compilation, targeted tests, model smoke load,
  `git diff --check` and scoped status.
- This target is offline perception training only. It does not open a camera,
  mutate the ledger, send robot commands, or authorize physical motion.
- The user separately requested publication to `main` after the live comparison.
  Stage, commit and push only the owned paths listed above. Do not include the
  unrelated dirty card-recognition or track-line worktree paths.

## Baseline

The current chip-v2 checkpoint produced:

- confidence >= 0.05: 25 detections across 15/36 images;
- confidence >= 0.25: 16 detections across 11/36 images;
- confidence >= 0.50: 13 detections across 10/36 images;
- maximum false-positive confidence: 0.8411.

All 36 source files are exact-byte unique. The false positives are concentrated
in the `chip_neg` face-up sequence, while the untouched `cards_neg` sequence
still contains at least one high-confidence false positive and is therefore a
useful independent replay.

## Completed Result

- Immutable source snapshot: 36 exact-unique images.
- Dataset manifest SHA-256:
  `8f8a2f393e1d664168d2f1924b08991f016c1b59fd67a4548108d0bc54ed0efe`.
- Replay construction: 2241 previous train images plus 126 new train-negative
  views; previous 442-image validation and 460-image test retained; 15 new
  card-back images held out.
- Candidate checkpoint SHA-256:
  `d68548783f77a2144b1a4d2870e9dd55b4d6c208817d6cb4165735b4a89544a9`.
- Card false positives at confidence 0.25: 16 before, zero after.
- Maximum card false-positive confidence: 0.8411 before, 0.2041 after.
- Untouched card-back maximum: 0.7140 before, 0.0724 after.
- Original chip test changed from P/R/F1/mAP50/mAP50-95
  `0.930/0.896/0.912/0.930/0.817` to
  `0.941/0.881/0.910/0.936/0.830`.
- Target-camera holdout changed from `0.997/1.000/0.999/0.995/0.607` to
  `0.998/1.000/0.999/0.995/0.663`.

After an explicit operator-requested DroidCam comparison, the result replaces
the tracked runtime chip-localization weight and the default live threshold is
`0.40`. The old chip-v2 bytes are retained as the manifest fallback at
`models/assets/chip_recognition/yolo11n-localization-chip-v2-v1-fallback/best.pt`.
The separate 10/20 rim-colour classifier was not changed. No physical motion
was started.

Validation:

- New builder compilation and isolated test: passed.
- Chip-scoped regression suite: 43 passed.
- Practical full suite after synchronizing the latest remote main: 433 passed,
  8 unrelated failures (missing Vosk and hand-landmarker environment assets,
  existing game-demo import environment, and the unrelated current card ONNX
  being incompatible with OpenCV DNN).
- Base checkpoint and dataset-manifest hashes: verified.
- Scoped owned-file whitespace check: passed. Repository-wide
  `git diff --check` remains blocked by pre-existing trailing whitespace in
  the unrelated dirty `src/track_line/live_line_detection.py`.
