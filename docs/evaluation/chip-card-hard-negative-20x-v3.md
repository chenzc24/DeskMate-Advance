# Chip Card Hard-Negative 20x V3 Evaluation

## Purpose

Measure whether increasing each face-up-card negative source from 6 views to
the original plus 20 augmented views reduces card-as-chip false detections
without damaging chip localization.

## Training Inputs

- Initialization: pre-negative chip-v2 fallback, SHA-256
  `80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`
- Previous chip training replay: 2241 images
- New train-only card negatives: 21 sources, 441 images
- Unchanged validation: 442 chip images
- Unchanged test: 460 chip images
- Untouched card-negative holdout: 15 images
- Unchanged target-camera holdout: 13 chip images
- Dataset manifest SHA-256:
  `ba771456d71d792f34578fc2b70cb3d0f1c2c6504f4dca5d23c4551d984c07e9`

The 441 negative images consist of each original source plus 20 deterministic
augmentations. Their YOLO label files are deliberately empty. The complete
previous positive training set is replayed to limit forgetting.

## Comparison Protocol

Compare these weights under identical inference settings:

1. pre-negative chip-v2 fallback;
2. current 126-negative runtime model;
3. new 441-negative candidate.

Negative replay thresholds are `0.05`, `0.25`, `0.40`, and `0.50`. Positive
localization is evaluated independently on the unchanged test and
target-camera holdout. The candidate remains a development artifact until the
comparison supports replacing the runtime model.

## Results

All values below were measured again under the comparison protocol above.

| Model | Card FP @ 0.05 | Card FP @ 0.40 | Max card confidence | Test precision | Test recall | Test F1 | Test mAP50 | Test mAP50-95 | Target F1 | Target mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-negative chip-v2 | 25 | 14 | 0.841058 | 0.929573 | 0.895641 | 0.912291 | 0.929717 | 0.817034 | 0.998712 | 0.607343 |
| current 126-negative | 3 | 0 | 0.204066 | 0.940907 | 0.881110 | 0.910027 | 0.936136 | 0.830247 | 0.998767 | 0.662773 |
| new 441-negative | 0 | 0 | 0.045572 | 0.950940 | 0.877807 | 0.912911 | 0.929047 | 0.819557 | 0.998719 | 0.621413 |

At confidence `0.25`, `0.40`, and `0.50`, both negative-tuned models
produce zero detections on all 36 card images. The new candidate additionally
eliminates every detection at confidence `0.05`, reducing the maximum false
confidence by 77.7% relative to the current 126-negative model.

The new candidate does not dominate the current runtime model on positive
localization. Relative to the current 126-negative model, test recall decreases
by 0.003303, test mAP50-95 decreases by 0.010691, and target-camera mAP50-95
decreases by 0.041360. Test precision and F1 increase slightly.

## Runtime Selection

The offline recommendation was to keep the 126-negative model because at the
runtime threshold of `0.40` it already eliminated all false detections in this
card replay and retained better localization quality. After live A/B testing,
the user explicitly selected the 441-negative checkpoint to replace the
tracked development runtime weight on 2026-07-25.

This is a development runtime selection, not release admission. The lower
target-camera mAP50-95 and the small, single-session negative replay remain
documented limitations. The immutable chip-v2 checkpoint remains the fallback.
