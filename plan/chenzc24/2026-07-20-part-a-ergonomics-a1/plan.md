# Part A Ergonomics A1 Foundation

## Outcome And Owned Paths

Create the detailed Part A execution plan and implement the first independent
Ergonomics boundary: Pose/Face model adapters plus deterministic luminance and
audio-level observations.

Owned paths:

- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- `configs/ergonomics/`
- `src/deskmate_advance/perception/ergonomics/`
- `tests/ergonomics/`
- `scripts/ergonomics/`
- this target plan

## Dirty Paths Left Read-Only

- Shared `domain/`, camera, runtime, events, integration and dependency files.
- Part B paths and all existing Part B model assets.
- `models/manifest.yaml`; Part A consumes registered development assets but
  does not promote or edit them.
- Existing Stage 0/1 documentation and unrelated dirty paths.

## External Dependencies

- Existing local MediaPipe Pose Full/Lite and Face model assets.
- Existing project MediaPipe/OpenCV environment.
- Final robotics camera, target recordings and acceptance thresholds remain
  pending; no release selection is possible in this target.

## Validation And Robot Motion

- Unit-test framework-object conversion, timestamps, missing observations,
  luminance and audio-level calculations.
- Run the existing local-asset MediaPipe smoke test and the full Python test
  suite.
- Parse Part A configuration, run `git diff --check`, and inspect scoped Git
  status.
- No recording, controller integration or physical robot motion is involved.

## Commit Intent

Do not stage, commit, push, create a branch or open a PR unless requested.
