# Part A A3 Replay And Candidate Handoff

## Outcome And Owned Paths

Complete the privacy-safe offline portion of the A3 gate: replay bounded scalar
evidence through all eight independent ergonomic state machines, project stable
Part A candidate records, and summarize continuous-behaviour metrics without
claiming final product acceptance.

Owned paths:

- `src/deskmate_advance/temporal/ergonomics/`
- `scripts/ergonomics/replay_part_a.py`
- `tests/ergonomics/`
- `tests/fixtures/ergonomics/`
- `docs/evaluation/ergonomics-a3-replay.md`
- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- this target plan

## Dirty Paths Left Read-Only

- Existing A1/A2 perception, benchmark and live-camera implementation.
- Shared camera, domain, runtime, event, integration and controller paths.
- Part B paths and the untracked Part B file guide.
- Model assets, `models/manifest.yaml`, raw media and unrelated plans.

## External Dependencies

- Existing Part A scalar observations, calibration profile, event config and
  eight-rule engine.
- The committed fixture is synthetic contract evidence only and contains no
  image, audio, landmark, identity or private recording data.
- Product thresholds, final camera and final UnifiedEvent lifecycle remain
  unfrozen; candidate records must not silently freeze those contracts.

## Validation And Robot Motion

- Unit-test schema rejection, timestamp/source continuity, config hashes,
  missing/stale propagation, simultaneous candidates and duration metrics.
- Replay a deterministic synthetic fixture, run the full test suite, validate
  JSON output, run `git diff --check`, and inspect scoped status.
- Do not open a camera or microphone, retain media, integrate a controller or
  move a physical robot.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
