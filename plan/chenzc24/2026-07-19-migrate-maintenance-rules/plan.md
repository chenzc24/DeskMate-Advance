# Migrate Repository Maintenance Rules

## Outcome And Owned Paths

Adapt the reusable repository-maintenance rules from the separate Baseline
repository without importing any Baseline task, data, model, package, or
interface assumptions.

Owned paths:

- `AGENTS.md`
- `.gitignore`
- `plan/chenzc24/2026-07-19-migrate-maintenance-rules/plan.md`

## Dirty Paths Left Read-Only

- Parent `DeepLearning` workspace modifications outside the
  `advanced_project` gitlink.
- Baseline submodule `../project` and all of its files and history.

## External Dependencies

- Read-only source policy: `../project/AGENTS.md`.
- No external data, model, service, or hardware dependency.

## Validation And Robot Motion

- Run `git diff --check` and scoped `git status --short --branch`.
- Verify representative raw data, run, artifact, environment, and model-weight
  paths are ignored while manifests/configs remain trackable.
- Review the rule file for Baseline-specific task, label, weight, package, and
  interface leakage.
- No robot motion is involved.

## Commit Intent

Commit the adapted maintenance rules in the Advance subrepository, push its
`main` branch, then update only the parent `advanced_project` gitlink.
