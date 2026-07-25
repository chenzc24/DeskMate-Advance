# Freeze All Tracked Model Assets

Status: completed.

## Outcome And Owned Paths

Pull `origin/main`, then create one deterministic, file-level immutable
snapshot covering every Git-tracked file under `models/assets`. Register the
snapshot in the shared model manifest and publish only the freeze-related
changes to `main`.

Owned tracked paths:

- `models/manifest.yaml`;
- `models/frozen/all-model-assets-20260725.json`;
- `scripts/models/freeze_model_assets.py`;
- `tests/models/test_freeze_model_assets.py`;
- this plan.

All existing model bytes are read-only. The unrelated modified card trainer and
track-line file, local card experiments, ignored datasets/runs, and untracked
`models/assets/card_recognition/poker-dealer-v2/best.onnx` remain read-only and
out of the commit.

## Freeze Semantics

The snapshot records every tracked model file's path, byte size, SHA-256,
Git/Git-LFS storage mode and index blob identity. It also records every model
entry with an actual asset path and explicitly lists registered entries that
have no trained asset.

Freezing does not change runtime selection, promote development/candidate
models to release, close admission blockers, download assets or add untracked
local experiments.

## Validation

- Verify the generated snapshot against the working-tree model bytes and Git
  index.
- Confirm every manifest asset path is covered by at least one tracked file.
- Parse `models/manifest.yaml` and the generated snapshot.
- Run the targeted freeze test and practical model/config tests.
- Run `git diff --check` and scoped `git status --short --branch`.

## Safety, Physical Motion And Commit Intent

This is metadata-only model governance work. It does not run camera/audio
capture, persist identity data, perform inference, alter the game ledger or
authorize physical motion. The user explicitly requested direct publication
to `main`; commit and push only the owned paths after validation.

## Result

- `origin/main` was pulled with `--ff-only` and was already current.
- Freeze `all-tracked-model-assets-20260725` covers 14 registered models with
  assets, all 67 tracked files under `models/assets`, 32 Git-LFS files, 35
  regular Git files and 304,159,175 bytes.
- The assetless `player-action-landmark-tcn@untrained-v1` placeholder is
  explicitly recorded and no weight was fabricated for it.
- Runtime selection and every development/candidate/release state remain
  unchanged.
- Snapshot verification, JSON parsing and the targeted freeze test pass.
- The practical suite excluding the pre-existing 10,000-hand stress test
  reports 436 passed and 7 unrelated environment/local-asset failures: one
  Part A preflight, two unavailable Vosk dependency tests and four OpenCV DNN
  failures caused by the pre-existing untracked
  `poker-dealer-v2/best.onnx`.
- Scoped `git diff --check` passes. The full check still reports the unrelated
  pre-existing trailing whitespace in `src/track_line/live_line_detection.py`.
