# Poker Dealer Pivot

## Outcome And Owned Paths

Supersede DeskMate as the active product, retain its complete unfinished state
on local branch `codex/archive-deskmate-20260721` at commit `668528f`, migrate
only product-neutral camera/frame infrastructure, and establish an executable
Heads-Up Fixed-Limit Texas Hold'em robotic-dealer Core plan.

Owned paths: repository-root governance and packaging; active `src/`, `tests/`,
`scripts/`, `configs/`, `models/manifest.yaml`, `docs/`, `archive/`, and this
target plan.  DeskMate-specific tracked paths may be removed from `main`
because the named archive commit is the canonical snapshot.

## Dirty Paths Left Read-Only

The target starts from a clean `main` at `56155a9`, one commit ahead of
`origin/main`.  Ignored local datasets, model assets, environments, runs, and
artifacts remain read-only and will not be inventoried, copied, or deleted.

## External Dependencies

- Robotics confirmation of feeder, rotation/distribution geometry, sensors,
  safety interlock, homing, and hardware command/acknowledgement details.
- Human decisions for exact Fixed-Limit stakes, blind/button convention,
  action-input hardware, camera placement, deck/lighting, and demo timing.
- Later consented target-camera card data and any admitted pretrained weights.
- No Baseline repository dependency and no DeskMate product-runtime dependency.

## Validation And Physical Motion

Validate import/package boundaries, deterministic poker rules and evaluator
tests, migrated camera tests, config/manifest parsing, referenced paths, archive
pointer, `git diff --check`, and full practical Python tests.  No physical robot
motion is authorized or performed by this target.

Completed on 2026-07-21: the archive branch and commit exist and contain the old
PPT; the archived WIP passed 213 tests in the project `.venv`.  On `main`, both
JSON-compatible configuration files parse, editable packaging/import succeeds,
all local Markdown links resolve, active runtime paths contain no old package or
ergonomics/model IDs, the migrated/default camera contract is `table_camera`,
14 active tests pass, and `git diff --check` passes.  No camera recording,
download, hardware connection, or physical motion was performed.

## Commit Intent

The user authorized the archival branch/commit.  Keep the Poker Dealer pivot
changes uncommitted and do not push unless the user separately requests it.
