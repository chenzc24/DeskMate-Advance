# Part A Live Laptop Camera Probe

## Outcome And Owned Paths

Connect the existing laptop camera to Part A Pose/Face adapters, feature
extractors and luminance statistics in a bounded, non-recording live probe.

Owned paths:

- `configs/ergonomics/perception.json`
- `src/deskmate_advance/features/ergonomics/live.py`
- `src/deskmate_advance/features/ergonomics/__init__.py`
- `scripts/ergonomics/live_part_a.py`
- `tests/ergonomics/test_live.py`
- `docs/evaluation/ergonomics-live-laptop.md`
- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- this target plan

## Dirty Paths Left Read-Only

- Shared camera/domain/runtime/event/integration implementations.
- Part B paths, including the existing untracked Part B file guide.
- Model assets, model manifest, A2 benchmark and unrelated plans.

## External Dependencies

- Existing HP True Vision laptop camera through the shared OpenCV adapter.
- Existing local Pose Full/Lite and Face model assets.
- No network, recording, dataset or controller dependency is introduced.

## Validation And Robot Motion

- Unit-test cadence, stale state, missing/error propagation, bounded metrics
  and summary output with fake adapters.
- Run full tests, offline smoke and CLI help.
- Run a short headless laptop-camera smoke that stores no frames or media.
- No controller integration or physical robot motion occurs.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
