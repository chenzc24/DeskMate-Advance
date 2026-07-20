# Split Advance Model Work Into Two Parallel Workstreams

## Outcome And Owned Paths

Define two vertical, non-conflicting model-to-event workstreams and one later
single-owner integration phase. Update only:

- `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md`
- `docs/architecture/perception-architecture.md`
- `configs/perception/candidates.json`
- this target plan

## Dirty Paths Left Read-Only

- Existing camera/domain code, tests, runtime scripts, model assets and model
  manifest.
- Existing environment and Stage 1 evaluation work outside the owned paths.
- The parent workspace and separate Baseline repository.

## External Dependencies

No new data, model, service or hardware dependency is introduced. The final
robotics camera contract remains pending.

## Validation And Robot Motion

Parse the updated JSON, verify documented paths and ownership do not overlap,
then run `git diff --check` and scoped `git status --short --branch`. No model
execution, recording, controller integration or physical robot motion occurs.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
