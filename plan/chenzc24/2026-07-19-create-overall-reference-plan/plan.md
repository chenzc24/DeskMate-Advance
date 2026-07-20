# Create Function-Driven Overall Reference Plan

## Outcome And Owned Paths

Create one authoritative plan under `docs/` that defines the complete staged
workflow from function and event freeze through pretrained-model selection,
data, learned extensions, runtime, integration, validation, and delivery.

Owned paths:

- `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md`
- `AGENTS.md`
- `ADVANCE_MODEL_MACRO_PLAN.md`
- `plan/chenzc24/2026-07-19-create-overall-reference-plan/plan.md`

## Dirty Paths Left Read-Only

- Existing uncommitted maintenance-rule migration files outside the owned
  paths except for the explicitly owned `AGENTS.md` update.
- Parent `DeepLearning` workspace modifications outside this subrepository.
- Baseline submodule `../project` and all of its files and history.

## External Dependencies

- Read-only requirement deck: `DeskMate_Advance_Proposal (1).pptx`.
- No external data, model download, service, or hardware dependency for this
  documentation target.

## Validation And Robot Motion

- Run `git diff --check` and scoped `git status --short --branch`.
- Verify the master plan covers every PPT function and event.
- Verify every stage has entry conditions, tasks, outputs, exit gates,
  dependencies, validation, and fallback decisions.
- Verify `AGENTS.md` points to the new single source of truth and no obsolete
  top-level plan remains authoritative.
- No robot motion is involved.

## Commit Intent

Keep the documentation and maintenance-rule changes uncommitted until the user
explicitly requests staging, commit, or push.
