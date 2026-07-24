# Freeze Perception Model Set

Status: completed.

## Outcome And Owned Paths

Record an immutable Git/Git-LFS snapshot for the user-selected perception
assets:

- Poker Dealer Card V4;
- chip-v2-v1 localization, retained at its historical directory path;
- the fixed-design 10/20 rim-colour denomination classifier;
- the original English speech, speaker-verification, face, gesture and
  pose-attribution assets.

Owned tracked paths are `models/manifest.yaml` and this plan. Existing model
bytes are read-only and must match their recorded hashes. All unrelated paths
remain read-only.

## Freeze Semantics

The freeze records exact versions, paths, byte sizes and SHA-256 identities in
Git. It does not silently promote development models to candidate or release,
does not change the selected runtime adapter, and does not close any admission
blocker.

Card V4 remains a candidate. Chip, speech, face and gesture models retain their
existing development state until their independent validation gates close.

## Validation

- Verify each file SHA-256 and each Vosk model-tree SHA-256.
- Verify all large binaries are represented by Git LFS.
- Load the Card V4 and chip localization YOLO checkpoints offline.
- Parse the denomination JSON, load the OpenCV face models, instantiate the
  MediaPipe tasks, and verify the Vosk assets through existing configuration
  contracts.
- Parse `models/manifest.yaml`, run targeted model/config tests, then run
  `git diff --check` and scoped Git status.

## Physical Motion And Commit Intent

No camera capture, audio capture, face enrollment, model inference session or
physical motion is authorized by this freeze. Commit and push only the freeze
metadata and plan after validation, as explicitly requested by the user.

## Result

- Frozen set `perception-baseline-20260724` records nine versioned assets.
- All file sizes, SHA-256 values, Vosk tree hashes and Git/Git-LFS tracking
  match the repository bytes.
- Card V4 and chip-v2-v1 YOLO checkpoints, the 10/20 classifier and the
  MediaPipe pose asset load successfully offline.
- Existing gesture, speech, speaker and face asset tests pass without camera,
  microphone or identity persistence.
- Model admission states and runtime selection are unchanged.
