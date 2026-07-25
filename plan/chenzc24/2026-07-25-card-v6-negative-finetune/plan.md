# Poker Dealer Card V6 Hard-Negative Fine-Tune Plan

Status: completed and frozen as a V6 development snapshot on 2026-07-25;
independent negative validation and runtime selection remain pending.

## Outcome And Owned Paths

Starting from the frozen Poker Dealer V5 candidate, add reviewed target-camera
hard-negative evidence from `data/raw/poker_label/neg_new/` while replaying the
complete V5 positive training view. Produce a reproducible V6 development
candidate only if it materially reduces false card detections without
regressing supported 52-class card recognition. Never overwrite V5 and leave
runtime model selection unchanged.

Planned tracked paths:

- `scripts/data/build_card_negative_v6.py`;
- `tests/data/test_build_card_negative_v6.py`;
- a focused negative-rejection evaluator and its tests if the existing card
  evaluator cannot report empty-frame false positives;
- `models/manifest.yaml` and
  `models/assets/card_recognition/poker-dealer-v6/` only after all available
  gates pass;
- model Git ignore/LFS declarations if required;
- this plan.

Planned ignored paths:

- `data/work/card_negative_v6/`;
- `runs/card_finetune/card_v6_negative/`;
- V5/V6 comparison, rejection and target-camera replay evidence.

Existing dirty runtime, camera, mobile UI, V5 and unrelated plan paths remain
read-only unless the user separately expands scope.

## Source Audit And Negative Contract

- `neg_new` contains 36 immutable PNG sources, all with distinct SHA-256
  values: 15 in `cards_neg` and 21 in `chip_neg`.
- Visual review classifies `cards_neg` as face-down red-backed cards, stacked
  or spread at several positions, scales and orientations.
- Visual review confirms that `chip_neg` contains face-up playing cards. It is
  a negative set for the chip detector, not the card detector, and is excluded
  from V6 card training. Writing empty card labels for it would teach the card
  model to suppress valid cards.
- Only the 15 `cards_neg` images are valid empty-label card-detector samples
  under this explicit contract: V6 recognizes supported face-up cards, while
  face-down backs must yield no card identity.
- The 36 frames come from short consecutive bursts and contain many perceptual
  near duplicates. Original bytes remain untouched, and source/burst identity
  is recorded in the derived manifest.

## V5 Baseline

Run V5 on all 36 supplied images before filtering and preserve every
prediction. This audit was also what exposed that `chip_neg` is unsafe for
card-negative training. On the accepted `cards_neg` set, at the current UI
confidence threshold of 0.15, V5 produced 10 false boxes on 6/15 images:

| Negative group | Images with false detections | False boxes | False boxes/image |
| --- | ---: | ---: | ---: |
| `cards_neg` | 6/15 | 10 | 0.667 |
| Excluded `chip_neg` audit | 18/21 | 63 | 3.000 |

At confidence 0.25, accepted `cards_neg` still produced four false boxes on
3/15 images. These accepted-set figures establish the hard-negative regression
target; excluded `chip_neg` figures are retained only as audit evidence.

## Split And Leakage Policy

- Keep the existing V5 positive split unchanged: 2,550 train images and the
  exact same held-out 450-image validation list. No positive validation image
  may enter training.
- All derivatives of a negative source remain in that source's group. No
  random augmented-image split is permitted.
- The accepted images appear to be one short capture session. Therefore using
  some adjacent frames for training and others for a claimed independent
  validation score would be leakage.
- Use the complete accepted `cards_neg` session for development training. A
  new, complete target-camera capture session containing face-down cards at
  varied distance, orientation, lighting and layout remains required for
  independent V6 candidate/release admission.
  Until then the final all-negative checkpoint remains `development`, even if
  the positive validation and group-holdout diagnostics pass.

## Negative Augmentation

Build an initial deterministic view of exactly 300 negative images:
20 outputs per accepted raw source including the unchanged source. Every output receives
an existing zero-byte YOLO label file and a manifest link to its source hash,
burst and transformation.

Allowed camera-realistic transformations:

- bounded exposure, contrast, gamma and colour-temperature changes;
- light sensor noise, JPEG degradation, defocus and motion blur;
- bounded resize, crop/translation and mild perspective distortion that keeps
  the hard distractor visible;
- modest in-plane rotation representative of the fixed camera view.

Do not mirror, copy-paste, mosaic, erase the principal distractor, synthesize
new ranks/suits, or change a scene's semantic card count. Contact sheets must
cover raw sources and every augmentation family. Determinism, source lineage,
empty labels, image readability and transform bounds require automated tests.

The initial mixed training view is therefore 2,850 images: the unchanged 2,550
V5 positive train images plus 300 negatives (10.5% negative). Do not inflate
the 15 accepted sources to thousands or allow repeated negatives to dominate
the positive replay. If a ratio experiment is needed, compare approximately
150/300/450 negatives using identical seeds and select the smallest ratio that
meets rejection gates while preserving positive recall.

## Fine-Tuning

- Initialize from
  `models/assets/card_recognition/poker-dealer-v5/best.pt`; do not restart from
  LGD and do not modify the frozen V5 asset.
- Replay every V5 positive training sample in each experiment so background
  gradients do not cause catastrophic forgetting.
- Use CUDA 0, image size 960, batch 4, deterministic seed 20260725, AdamW and a
  low fine-tuning learning rate. Select the best checkpoint by a compound
  decision rule, not positive mAP alone.
- Run short, bounded ratio pilots first; then train the selected view for at
  most 20 epochs with validation-based early stopping. Stop and reject a run
  immediately if positive recall or weak V5 classes regress beyond the gates.

## Evaluation And Admission Gates

Evaluate V5 and every V6 experiment on identical evidence:

1. The unchanged 450-image V5 positive validation list, reporting precision,
   recall, mAP50, mAP50-95, per-class precision/recall/F1 and confusion, plus
   macro per-rank and per-suit summaries.
2. Raw supplied negative group holdouts at confidence 0.15 and 0.25, reporting
   images with any false detection, false boxes/image, predicted-class
   distribution and confidence distribution separately for card backs,
   face-down-card layout and confidence band.
3. A newly captured, source-independent negative session before candidate
   admission.
4. Recorded target-camera replay for per-slot stability, duplicate-card hard
   error behaviour, latency and false accepted card identities per hour.

Minimum selection gates relative to V5:

- positive mAP50-95 loss no greater than 0.5 percentage point;
- aggregate recall loss no greater than 1.0 percentage point;
- no supported class recall loss greater than 5 percentage points, with 9D,
  4C, 5D and 5H called out explicitly;
- at confidence 0.15 on independent negatives, at least a 90% reduction from
  the V5 false-box rate and no more than 10% of frames with any false card;
- no duplicate identity, unknown/rejection or target-camera stability
  regression.

If no checkpoint satisfies both positive and negative gates, retain V5 and
collect more diverse raw negative sessions instead of adding more derivatives
of these 36 images.

## Validation, Physical Motion And Commit Intent

Run focused data/evaluation tests, the practical full test suite,
machine-readable manifest/hash validation, `git diff --check`, and scoped
`git status --short --branch`. This plan and any later offline augmentation or
training authorize no robot motion, ledger mutation, runtime model switch,
commit or push. Camera capture or live UI testing requires a separate explicit
request and an operator-present, no-motion setup.

## Completed Outcome

- Visual review rejected all 21 `chip_neg` frames from card-negative training
  because they contain face-up playing cards. Only the 15 `cards_neg`
  face-down-card sources were accepted.
- The deterministic view contains 2,550 unchanged V5 positive training images
  plus 300 empty-label negative images derived from those 15 sources. The
  original 450-image positive validation list is byte-for-byte unchanged.
- Fine-tuning initialized from frozen V5, stopped at epoch 8 with patience 5
  and selected epoch 3. On the common positive validation list, V6 precision,
  recall, mAP50 and mAP50-95 are 0.93693, 0.81552, 0.91407 and 0.76730.
- At confidence 0.15 on the fitted 15-source card-back set, V5 produced ten
  false boxes on six images and V6 produced zero. This is training-fit
  evidence, not independent negative validation.
- V6 remains `development`: 10D, 4D, 9S, 3D and JD recall regressions exceed
  five percentage points, and a new complete negative capture session is
  still required.
- The immutable development checkpoint is frozen under
  `models/assets/card_recognition/poker-dealer-v6/`; runtime selection remains
  unchanged and no physical motion was performed.
