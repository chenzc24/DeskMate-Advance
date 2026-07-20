# Resolve Stage 1 Environment And Select Perception Architecture

## Outcome And Owned Paths

Resolve the OpenCV/MediaPipe package-family conflict, validate the chosen
environment without breaking the existing camera boundary, select the initial
pretrained model bundles and fallbacks, and document the runtime perception
architecture for discussion.

Owned paths:

- `.gitignore`
- `pyproject.toml`
- `docs/evaluation/pretrained-selection.md`
- `configs/perception/candidates.json`
- `docs/architecture/perception-architecture.md`
- `models/manifest.yaml`
- `scripts/evaluation/smoke_mediapipe_tasks.py`
- `plan/chenzc24/2026-07-20-resolve-environment-select-architecture/plan.md`

## Dirty Paths Left Read-Only

- Existing camera/domain/runtime source and tests under `src/`,
  `scripts/runtime/`, and `tests/perception/`; only validation may exercise
  them.
- Existing Stage 0/Stage 1 planning and maintenance-rule changes outside this
  target's owned paths.
- Concurrent `plan/chenzc24/2026-07-20-split-model-workstreams/` target and its
  master-plan workstream rules. Its compatible additions to this target's
  architecture/config paths are preserved; its own plan remains read-only.
- Parent `DeepLearning` workspace and the separate Baseline submodule.

## External Dependencies

- PyPI packages: MediaPipe, the single selected OpenCV wheel family, and
  benchmark support dependencies.
- Official Google AI Edge Pose, Face, Hand, and EfficientDet-Lite0 model
  bundles downloaded to ignored local `models/assets/` paths with SHA-256
  recorded in `models/manifest.yaml`.
- Final robot camera remains pending; this target performs only asset
  initialization and deterministic synthetic-input smoke tests.

## Validation And Robot Motion

- Validate the proposed dependency set in a fresh temporary virtual
  environment before changing the project environment.
- Verify only one OpenCV wheel distribution is installed and `cv2` imports.
- Run existing camera tests unchanged.
- Initialize every selected MediaPipe task from the locally hashed asset and
  run bounded synthetic-image inference.
- Parse candidate config and model manifest; run `git diff --check` and scoped
  `git status --short --branch`.
- No live recording, controller integration, or physical robot motion is
  involved.

## Commit Intent

Do not stage, commit, push, create a branch, or open a PR unless the user asks.
