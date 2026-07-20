# Part A A3 Labeled Evaluation Handoff

## Outcome And Owned Paths

Complete the hardware-independent labeled-evidence path: load one bounded,
versioned annotation artifact, bind it to an exact scalar replay and source,
then expose formal false-trigger and detection-latency evaluation through the
existing replay CLI.

Owned paths:

- `src/deskmate_advance/temporal/ergonomics/annotations.py`
- `src/deskmate_advance/temporal/ergonomics/__init__.py`
- `scripts/ergonomics/replay_part_a.py`
- `tests/ergonomics/test_annotations.py`
- `tests/ergonomics/test_replay_pipeline.py`
- `docs/evaluation/ergonomics-a3-replay.md`
- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- `plan/chenzc24/2026-07-20-part-a-a3-labeled-evaluation/plan.md`

## Dirty Paths Left Read-Only

The worktree is clean at target start. All camera, feature, rule, candidate,
stereo, Part B, shared-domain, model-manifest, data and artifact paths outside
the owned list remain read-only.

## External Dependencies

- Existing scalar replay, event configuration and evaluation APIs.
- Later human-reviewed target-camera scalar replays and annotation artifacts.
- Human-frozen acceptance thresholds remain external and are not invented by
  this target.

No camera, robot, dataset download, checkpoint, network service or new package
is required. Stereo calibration remains a hardware-blocked TODO.

## Validation And Physical Motion

Test valid annotations, exact replay/source binding, expected SHA-256, duplicate
keys, non-finite/deep or oversized input, interval semantics, overlap rejection,
data-status mismatch and a labeled CLI run. Run targeted tests, the full Python
suite, `git diff --check` and scoped Git status.

No physical robot motion is involved.

## Commit Intent

Do not commit or push this target unless the user explicitly requests it.
