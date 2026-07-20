# Part A Ergonomics A2 Features And Recorded Benchmark

## Outcome And Owned Paths

Implement framework-independent Pose/Face feature extraction and a
manifest-driven recorded-video benchmark for comparing Pose Full, Pose Lite
and Face on identical evidence.

Owned paths:

- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- `docs/evaluation/ergonomics-a2-tooling.md`
- `configs/ergonomics/`
- `src/deskmate_advance/perception/ergonomics/observations.py`
- `src/deskmate_advance/perception/ergonomics/landmarkers.py`
- `src/deskmate_advance/features/ergonomics/`
- `scripts/ergonomics/`
- `tests/ergonomics/`
- `data/manifests/ergonomics-recordings.example.jsonl`
- this target plan

## Dirty Paths Left Read-Only

- Shared domain, camera, audio, runtime, event and integration code.
- Part A A1 signal calculations and public behaviour outside the provenance
  context additions owned above.
- All Part B paths and the untracked Part B file guide.
- `models/manifest.yaml`, model assets and unrelated plans.

## External Dependencies

- Existing local Pose Full/Lite and Face development assets.
- Target recordings are not yet available. The benchmark accepts only local,
  hash-verified manifest entries; it never records or copies media.
- The final robotics camera contract and acceptance thresholds remain pending.

## Validation And Robot Motion

- Unit-test geometry, normalization, missing masks, timestamped motion,
  blendshape extraction, matrix decomposition, manifest validation and metric
  aggregation.
- Run the full Python test suite, A1 smoke, config/schema parsing,
  `git diff --check` and scoped Git status.
- Do not activate a camera or microphone, store private media, integrate a
  controller or move a physical robot.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
