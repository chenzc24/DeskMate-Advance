# Runtime Review Closure

## Outcome

Close every actionable software finding from the 2026-07-22 full-line review:
authoritative rule loading, frozen-roster identity authority, bounded Part A
attention, settled-log admission, thirteen-slot card geometry and spatial
binding, validated Dealer acknowledgements, attribution admission, registration
correctness, session-ledger continuity, operator recovery, resource-lock
correctness and clearer module boundaries.

Target-camera geometry, model admission, board-reveal mechanics and physical
Dealer validation remain evidence gates. Software must represent them honestly
and must not invent measurements, calibration or sensor success.

## Owned Paths

- `configs/contracts/`, `configs/runtime/`, and development card-geometry config
- `src/poker_dealer/game/` for extracted rules/state/log modules and ACK audit
- `src/poker_dealer/perception/cards/` for slot geometry/spatial binding
- `src/poker_dealer/runtime/` for roster, attention, session, recovery and ports
- `scripts/runtime/` for CLI/checker behavior
- focused tests and scoped architecture/evaluation documentation

## Dirty And Read-Only Paths

Preserve all unrelated existing changes. Baseline, DeskMate archive, raw/private
data, model assets, model thresholds and Robotics firmware/mechanics are
read-only. The existing unified runtime changes are coordinated inputs and may
be edited only where this target owns a review finding.

## Implementation Stages

1. Make `core_v1.json` the composition-root rule/stack authority.
2. Retain and enforce the frozen `seat -> player_id` roster in Part A.
3. Bound the whole post-rotation attention window and fail closed on loop stop.
4. Add thirteen-slot geometry plus deterministic multi-detection binding.
5. Split raw ACK receipt from validated command completion and log all evidence.
6. Gate attribution confidence and fix registration/control/resource defects.
7. Add a session runtime for ledger/button/roster continuity and explicit
   recovery decisions.
8. Extract cohesive modules without changing public contracts unnecessarily.
9. Add vertical Replay variants for fold, raise and all-in, then run the full
   validation suite and update documentation.

## Validation

- Targeted unit tests for every finding and negative/fail-closed branch.
- Complete check/call, fold, raise and all-in/side-pot vertical Replays.
- Runtime profiles and JSON Schema validation.
- Practical full suite, compileall, document-link check, `git diff --check` and
  scoped status.

## Physical Motion Status

No physical motion is authorized. `robot_hardware` remains disabled and the
unavailable real Adapter remains fail closed. All command tests use protocol
objects, mocks or `SimulatedDealerAdapter`.

## Commit Intent

Do not commit, push, create a branch, release or PR unless explicitly requested.

## Completion Record

Implemented all listed software findings. The formal Runtime now uses the Core
configuration and SessionRuntime, thirteen-slot development geometry, strict
roster/attention/attribution admission, semantic in-game controls, two-stage
ACK completion, strict log admission and audited recovery. Rules and hand event
storage were extracted from the engine; geometry and session continuity are
separate modules. Added fold and raise/short-all-in full vertical replays.

Validation completed on 2026-07-22: Python compileall; all configuration JSON
parsed; all Runtime profiles passed their JSON Schema; Laptop preflight passed
without device opening; full test suite passed 300 tests; `git diff --check`
passed. No physical motion, camera capture or microphone capture was performed.
Target-camera geometry, model admission, physical Dealer/sensor validation and
20-hand qualification remain their existing Stage gates rather than software
review defects.
