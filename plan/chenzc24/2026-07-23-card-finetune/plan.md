# Card Detector Fine-Tuning Plan

## Outcome And Owned Paths

Fine-tune the pinned local YOLO11s card-corner detector without downloading new
model weights. Use the external CC0 snapshot for source-domain adaptation,
followed by the project-owned small-target derived view for target-domain
adaptation.

Owned tracked paths:

- `scripts/data/build_card_training_view.py`
- `scripts/perception/train_card_detector.py`
- `tests/data/test_build_card_training_view.py`
- `requirements-training-cu130.txt`
- `pyproject.toml` training-extra pins
- this plan

Owned ignored paths:

- `data/work/card_finetune_v1/`
- `runs/card_finetune/`

The raw datasets, existing augmentation view, pinned baseline weights, runtime
configuration and model manifest are read-only during training.

## Dirty Read-Only Paths

Preserve all pre-existing runtime diagnostics, README, dataset-import,
pretrained-labeling and augmentation changes. Do not modify or stage unrelated
work.

## Inputs And Dependencies

- Baseline:
  `models/assets/card_recognition/lgd-cards-gen3/best.pt`
  SHA-256
  `3513b77ab418ac2e86d3bf53be6566522666e656cbccb12678ba93feb6201004`.
- External snapshot:
  `kaggle-andy8744-playing-cards-object-detection-dataset-v1`.
- Target derived view:
  `data/work/poker_new_augmented_v1/manifest.json`.
- Install pinned PyTorch, torchvision and Ultralytics packages in the existing
  project virtual environment. Do not allow Ultralytics to download weights.

Pascal VOC class names must be normalized and mapped by card code to the pinned
52-class order. Numeric source order must never be reused.

## Training And Validation

1. Create hard-linked derived image views and converted YOLO labels without
   changing raw bytes.
2. Run conversion tests and validate all image/label pairs, class coverage,
   finite normalized boxes and view manifests.
3. Run a bounded GPU smoke epoch on a small deterministic external subset.
4. Train stage A from the pinned baseline on the external training-only view.
5. Train stage B from stage A's best/last checkpoint on the target small-object
   view with a lower learning rate.

The external snapshot has no trustworthy deck/session split and the target
view consists of augmented siblings. Training validation is therefore disabled
and no run metric is represented as Gate evidence. Final selection requires an
independent Raspberry Pi target-camera session with per-class and rejection
reporting.

Run targeted tests, the practical full suite, `git diff --check` and scoped
`git status --short --branch`.

## Physical Motion And Privacy

Training is offline. It does not open cameras, save video, connect to robot
control, command physical motion or process identity-bearing media.

## Commit Intent

Do not stage, commit, push, create a branch, publish weights or change the
runtime-selected model. Checkpoints remain ignored development artifacts until
evaluation and a separate manifest update.

## Active Run

- Installed and verified `torch 2.13.0+cu130`, `torchvision 0.28.0+cu130` and
  `ultralytics 8.4.104` on the RTX 4070 Laptop GPU.
- Converted all 20,000 external images and 75,750 Pascal VOC objects by
  normalized class name. The 19,896-image formal train list excludes a
  104-image, all-class external health-monitor set. That set is not target or
  Gate validation.
- The 416-image, all-class GPU smoke run completed one epoch with finite
  losses, about 4.37 GiB peak allocated training memory, and local checkpoints.
- Stage A is active as `runs/card_finetune/stage_a_external_10ep`, starting
  from the pinned baseline for 10 external-domain epochs.
- Stage B is queued to start automatically after Stage A from its `last.pt` as
  `runs/card_finetune/stage_b_target_15ep`, using the target augmentation view,
  a lower learning rate and no additional color/geometric augmentation.
- After two Stage A epochs, train losses moved from
  `box=0.34016, cls=0.21882, dfl=0.79727` to
  `box=0.28096, cls=0.15966, dfl=0.78369`.

## Retraining Correction

The completed Stage B run is rejected as a development candidate: it trained on
10,400 augmented siblings from only 52 local source images, disabled per-epoch
validation, and sampled rotation only between -22 and +22 degrees. Its falling
training loss and external-domain final evaluation cannot establish target
generalization or target overfitting.

The replacement view is bounded to exactly 2,000 images:

- 800 locally derived training images from 52 source images;
- 200 locally derived validation images from the other 52 source images;
- 800 source-disjoint external training images;
- 200 source-disjoint external validation images.

For every card class, one old and one new local source are assigned to opposite
splits with a fixed seed; old/new origins are balanced 26/26 in each split.
Augmented siblings never cross the split. Full orientation bins around 0, 90,
180 and 270 degrees are included, together with very-far, far, medium and near
scale bins. Validation, plots, early stopping and best-checkpoint selection are
enabled for the replacement run. The resulting validation remains development
evidence rather than Gate evidence because only two local source images exist
per class and no third held-out target-camera session is available.
