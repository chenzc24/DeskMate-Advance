# Poker Dealer Card V6 Negative-Sample Fine-Tune Plan

Status: planning; no training has started.

## Outcome And Owned Paths

Starting from the frozen Poker Dealer V5 candidate, add reviewed target-camera
negative evidence from `data/raw/poker_label/neg_new/` without degrading
52-class card recognition. Produce a source-isolated training view, select the
best validation checkpoint as a V6 development candidate only if both positive
and negative regression gates pass, and leave runtime selection unchanged.

Planned tracked paths:

- a dedicated negative-view builder and focused tests;
- a mixed V5-replay plus negative training-view builder and focused tests;
- `models/manifest.yaml` and a V6 candidate asset directory only after gates
  pass;
- model Git ignore/LFS declarations if required;
- this plan.

Planned ignored paths:

- `data/work/card_negative_v6/`;
- `runs/card_finetune/card_v6_negative/`;
- V5/V6 comparison and live-test evidence.

Existing dirty runtime, camera, mobile UI, V5 and unrelated plan paths remain
read-only unless the user separately expands scope.

## Pending Data Audit

`neg_new` currently contains 36 PNG files: 15 under `cards_neg` and 21 under
`chip_neg`. Before any split or augmentation, inspect every source, identify
complete capture bursts/sessions and near-duplicate groups, and confirm that
each image is a genuine zero-card target rather than a missed positive.

## Physical Motion And Commit Intent

Planning and any later offline training authorize no robot motion, ledger
mutation, frame persistence beyond the user-provided dataset, runtime model
switch, commit or push.
