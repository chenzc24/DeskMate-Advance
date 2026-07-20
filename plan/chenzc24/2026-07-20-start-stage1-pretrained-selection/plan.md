# Start Stage 1 Pretrained-Model Selection

## Outcome And Owned Paths

Mark the current Stage 0 decisions and unresolved items, open Stage 1 with a
function-driven pretrained-component shortlist, and create one machine-readable
candidate configuration for future benchmark runs.

Owned paths:

- `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md`
- `docs/evaluation/pretrained-selection.md`
- `configs/perception/candidates.json`
- `.gitignore`
- `plan/chenzc24/2026-07-20-start-stage1-pretrained-selection/plan.md`

## Dirty Paths Left Read-Only

- Existing uncommitted laptop-camera target and its owned `pyproject.toml`,
  `src/`, `scripts/runtime/`, `tests/perception/`, and
  `plan/chenzc24/2026-07-20-laptop-camera-input/plan.md` paths.
- Existing maintenance-rule migration files outside this target's owned paths.
- Parent `DeepLearning` workspace and the separate Baseline submodule.

## External Dependencies

- Official Google AI Edge/MediaPipe documentation for Pose, Face, Hand, and
  Object Detector candidates.
- Laptop runtime inventory and locally available camera/microphone devices.
- Final robot camera source remains pending confirmation from the robotics
  team; no Baseline code or interface is assumed.
- No model asset download or physical hardware dependency in this target.

## Validation And Robot Motion

- Run `git diff --check` and scoped `git status --short --branch`.
- Parse `configs/perception/candidates.json` and verify IDs/status values.
- Verify every PPT perception function maps to a candidate or an explicit
  non-model/program-logic path.
- Verify unresolved Stage 0 items remain visibly marked and are not silently
  treated as frozen.
- No controller integration or physical robot motion is involved.

## Commit Intent

Do not stage, commit, push, create a branch, or open a PR unless the user asks.
