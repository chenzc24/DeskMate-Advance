# Part A A3 Calibrated Semantic States

## Outcome And Owned Paths

Add calibrated, timestamp-derived Part A semantic states over the existing
Pose/Face/luminance/audio evidence and expose them in the non-recording live
probe.

Owned paths:

- `configs/ergonomics/events.json`
- `src/deskmate_advance/temporal/ergonomics/`
- `src/deskmate_advance/features/ergonomics/audio_live.py`
- `src/deskmate_advance/features/ergonomics/__init__.py`
- `scripts/ergonomics/live_part_a.py`
- `tests/ergonomics/`
- `docs/evaluation/ergonomics-a3-foundation.md`
- `docs/evaluation/ergonomics-live-laptop.md`
- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- this target plan

## Dirty Paths Left Read-Only

- Shared camera, microphone, domain, runtime, event and integration code.
- Part B paths and the existing untracked Part B file guide.
- Model assets/manifest, A2 benchmark and unrelated plans.

## External Dependencies

- Existing Pose/Face observations, laptop microphone adapter and raw signal
  calculators.
- Product thresholds and final camera remain unfrozen; all numeric A3 values
  are explicit development defaults and configurable.
- No new model, data download or service dependency is introduced.

## Validation And Robot Motion

- Unit-test strict timestamps, unknown handling, entry/exit/cooldown, robust
  calibration, blink counting, bounded audio polling and all function rules.
- Run full tests and a bounded non-recording laptop camera/microphone smoke.
- Do not write private media or audio, integrate a controller or move a robot.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
