# Kaggle Playing-Cards Raw Import Plan

## Outcome And Owned Paths

Download the public Kaggle dataset
`andy8744/playing-cards-object-detection-dataset` as an immutable external raw
snapshot. Preserve the original archive, extract a separate inspection copy,
and record source, license, byte hash, file inventory, split/class metadata and
annotation-shape findings before any later class remapping or training merge.

Owned, ignored data path:

- `data/raw/poker_label/external/kaggle_andy8744_playing_cards_object_detection/`

Owned tracked path:

- this plan

The existing local poker images, labels, derived augmentation view and model
assets remain read-only.

## Dirty Read-Only Paths

Preserve all pre-existing README, runtime diagnostics, pretrained-labeling and
small-target augmentation work. Do not modify their source, tests, plans,
generated outputs or model assets.

## External Dependencies

Source page:
`https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset`.
The page identifies the dataset as CC0 and provides YOLOv5-format exports.
Network availability and Kaggle's public download endpoint are external facts.
No credentials may be printed or persisted by this task.

## Validation

- Compute and record the downloaded archive's SHA-256 and byte size.
- Retain the original archive unchanged after download.
- Reject archive members with absolute paths or parent traversal before
  extraction.
- Inventory images, YOLO labels, YAML files, classes and declared splits.
- Validate five-field finite normalized YOLO rows and image/label pairing.
- Inspect whether boxes cover whole cards or rank-suit corners before claiming
  compatibility with the project dataset.
- Run `git diff --check` and scoped `git status --short --branch`.

## Physical Motion And Privacy

This is an offline public-dataset import. It does not open cameras, save the
Raspberry Pi stream, connect to robot control, command physical motion or
process identity-bearing media.

## Commit Intent

Do not commit, stage, push, create a branch or open a pull request. Raw data
remains ignored under `data/raw/`.

## Completed Outcome

- Downloaded Kaggle dataset version 1 without credentials and preserved the
  exact `1,691,570,001`-byte archive.
- Archive SHA-256:
  `00c07b0fc4fd359d74db13c02cfa316c0c52a50acf4a433d4125a9b253e35e16`.
- Safely extracted 40,001 files after confirming that the ZIP contains no
  absolute or parent-traversal paths.
- Verified 20,000 JPEG images and 20,000 paired Pascal VOC XML annotations at
  720x720. All 52 declared classes occur, with 75,750 total valid boxes, zero
  parse errors, zero invalid boxes and no missing pairs.
- Visual review confirms the boxes cover visible rank-and-suit corner regions,
  not whole cards. The source class ordering and lowercase suit suffixes still
  require name-based remapping before a YOLO training view is generated.
- Recorded the ignored raw snapshot metadata in
  `snapshot_manifest.json`; its SHA-256 is
  `6573e2cce08406da5e4f9ff9dc85887e44c332d87c13fda6ec701ced2ad7e446`.
- The archive declares no split, so this source remains external
  training-only and is not validation, test or Gate evidence.
