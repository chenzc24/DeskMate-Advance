# Chip Card Hard-Negative V2

## Purpose

This development target reduces poker-card false positives in the single-class
chip localizer. It continues from the tracked chip-v2 checkpoint and replays
the complete previous chip training view. It does not change the separate
10/20 rim-colour denomination classifier.

## Data Boundary

- 21 adjacent face-up-card frames in `chip_neg` are train-only hard negatives.
- 15 adjacent card-back frames in `cards_neg` remain an untouched negative
  replay.
- Every new negative image has an empty YOLO label.
- Six deterministic exposures of each train source are used, including the
  unchanged original.
- The previous chip-v2 train, validation, test and target-camera validation
  paths remain unchanged.

Raw images, derived images, labels and runs are ignored. The tracked scripts,
configuration and this report are sufficient to reproduce the view when the
private source is available.

## Baseline

On all 36 new images, the current chip-v2 checkpoint produced:

- 25 detections across 15 images at confidence 0.05;
- 16 detections across 11 images at confidence 0.25;
- 13 detections across 10 images at confidence 0.50;
- maximum false-positive confidence 0.8411.

The untouched 15-image `cards_neg` sequence included one false positive at
confidence 0.7140.

## Dataset And Training Identity

- Dataset view manifest SHA-256:
  `8f8a2f393e1d664168d2f1924b08991f016c1b59fd67a4548108d0bc54ed0efe`
- Base chip-v2 SHA-256:
  `80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`
- Candidate SHA-256:
  `d68548783f77a2144b1a4d2870e9dd55b4d6c208817d6cb4165735b4a89544a9`
- Training: 10 epochs, 960 px, AdamW, learning rate `0.00004`, complete
  previous chip-v2 replay plus 126 train-only card-negative views.

## Results

At confidence 0.25, the candidate reduced card false positives from 16 to
zero. Across all 36 images, its maximum false-positive confidence was 0.2041.
On the untouched `cards_neg` sequence, the maximum was 0.0724.

| Replay | Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original chip test, 460 images/1514 instances | chip-v2 | 0.930 | 0.896 | 0.912 | 0.930 | 0.817 |
| Original chip test, 460 images/1514 instances | candidate | 0.941 | 0.881 | 0.910 | 0.936 | 0.830 |
| Target-camera holdout, 13 images/23 instances | chip-v2 | 0.997 | 1.000 | 0.999 | 0.995 | 0.607 |
| Target-camera holdout, 13 images/23 instances | candidate | 0.998 | 1.000 | 0.999 | 0.995 | 0.663 |

The original-test recall decreased by 0.0145 and F1 by 0.0023, while precision,
mAP50 and mAP50-95 improved. The target-camera holdout did not lose recall and
its mAP50-95 improved by 0.0554. After live DroidCam comparison, the operator
explicitly selected this development checkpoint to replace the tracked runtime
weight. The previous chip-v2 bytes remain available as the manifest fallback.
This runtime selection does not promote either model to release status.

The live DroidCam and Raspberry Pi chip runners now default to confidence
`0.40`; callers can still override it with `--confidence`.

## Commands

Build the replay view:

```powershell
python scripts/data/build_chip_card_hard_negative_view.py
```

Evaluate a model on the immutable raw snapshot:

```powershell
python scripts/evaluation/evaluate_chip_negative_images.py `
  --model models/assets/chip_recognition/yolo11n-localization-chip-v2-v1-fallback/best.pt `
  --images data/raw/chips/2026-07-25-card-hard-negative-source `
  --output runs/chip_evaluation/chip-card-hard-negative-v2/baseline_negatives.json
```

Train:

```powershell
python chip_recognition_workspace/train_chip_yolo11n.py `
  --config chip_recognition_workspace/chip_yolo11n_card_hard_negative_v2.json
```

Evaluation reports are under
`runs/chip_evaluation/chip-card-hard-negative-v2/`. Physical chips remain
Plus-only evidence and never replace the digital ledger.
