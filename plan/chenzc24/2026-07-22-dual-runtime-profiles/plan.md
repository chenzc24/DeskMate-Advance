# Dual Runtime Profiles And Single Launch Boundary

## Outcome

- Add one `scripts/runtime/run_hand.py` entry for Laptop, robot-camera and
  robot-hardware profiles.
- Keep one `HandRuntime`; profiles replace only camera, controls, speech and
  dealer adapters.
- Make Laptop and robot-camera profiles runnable with `SimulatedDealer`.
- Make robot-hardware fail closed until a released physical dealer adapter,
  protocol and Stage 3 safety evidence exist.
- Add process-level resource locks so concurrent test profiles cannot silently
  share a local camera, microphone or physical dealer.
- Remove button-based betting from the public production runtime namespace and
  retain it only as an explicit pilot.

## Owned Paths

- `configs/contracts/runtime_profile.schema.json`
- `configs/runtime/`
- `src/poker_dealer/runtime/profile.py`
- `src/poker_dealer/runtime/live_hand_app.py`
- `src/poker_dealer/runtime/resource_lock.py`
- `src/poker_dealer/robotics/dealer/`
- `src/poker_dealer/pilots/`
- `scripts/runtime/run_hand.py`
- `scripts/pilots/laptop_button_betting.py`
- focused runtime/robotics/contract tests and scoped documentation

## Dirty And Read-Only Coordination

The worktree contains the uncommitted unified HandRuntime/Part B target. Patch
the current files and preserve it. Perception model implementations, model
assets/manifests, raw data, archived DeskMate/Baseline and physical protocol
schemas are read-only for this target.

## Stages

1. Freeze a small runtime-profile schema and three example profiles.
2. Add dealer ports and fail-closed factory behavior.
3. Add bounded cross-process resource locks.
4. Add `LiveHandApplication` dependency assembly and preflight.
5. Add the single CLI and prove both non-physical profiles select the intended
   camera/dealer adapters.
6. Move direct button betting into an explicit pilot namespace.
7. Run contract, runtime, camera and full repository validation.

## Validation

- Parse and schema-validate all runtime profiles.
- CLI config checks for `laptop` and `robot_camera` must return success without
  opening hardware.
- `robot_hardware` must return a clear non-zero unavailable result.
- Adapter, resource-lock and application unit tests.
- Practical full test suite, `compileall`, `git diff --check`, scoped status.

## Physical Motion Status

No physical motion is authorized. Both runnable profiles use a simulated
dealer. The hardware profile must reject startup before sending any command.

## Commit Intent

Do not commit, push, create a branch, publish a release or open a PR unless the
user explicitly asks.

## Implementation Result

- Added schema-validated `laptop`, `robot_camera` and fail-closed
  `robot_hardware` profiles.
- Added the shared `LiveHandApplication` composition root, semantic
  `DealerPort`, explicit simulated/unavailable adapters, per-resource locks and
  per-profile/session/hand ignored log paths.
- Added `scripts/runtime/run_hand.py` for device-free preflight and bounded
  camera smoke checks. The application explicitly reports that the complete
  live perception-to-hand path is not integrated yet.
- Relocated direct button betting to `poker_dealer.pilots` and
  `scripts/pilots`; it is no longer exported from or stored in the production
  runtime namespace.
- Verified all three profile JSON documents against the Draft 2020-12 schema;
  Laptop and robot-camera preflight return ready, while robot-hardware returns
  exit code 2 without opening devices or sending a command.
- Validation passed: 274 tests, Python compileall, `git diff --check`, profile
  JSON parse/schema checks and scoped status review.
- Physical motion remained disabled; no camera smoke or physical command was
  run during automated validation.
