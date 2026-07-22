# Unified Hand Runtime And Part B Closure

## Outcome

- Add one application-level runtime that follows `HandEngine.state.phase` and
  hands control exclusively to Part A or Part B.
- Implement the missing no-burn hole/board delivery coordinator with unique
  rotate/dispense commands, matching terminal ACKs, visual confirmation and
  fail-safe pause behavior.
- Preserve `HandEngine` as the only authority for acting seat, legal actions,
  ledger, street transitions and settlement.
- Make the production setup path actually pass through `DEALING_HOLE` instead
  of assuming eight private cards already exist.
- Make confirmed-card updates idempotent and require showdown inputs to come
  from confirmed table slots.
- Keep existing Stage 1 `HandEngine.start()` as a documented pre-dealt
  simulator helper so existing oracle/replay coverage remains compatible.

## Owned Paths

- `src/poker_dealer/game/engine.py`
- `src/poker_dealer/game/__init__.py`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `src/poker_dealer/runtime/sequential_part_b.py`
- `src/poker_dealer/runtime/hand_runtime.py`
- `src/poker_dealer/runtime/__init__.py`
- focused tests under `tests/game/` and `tests/runtime/`
- scoped architecture/evaluation documentation required to describe the new
  runtime accurately

## Dirty And Read-Only Coordination

The worktree already contains broad uncommitted no-burn, registration,
announcer, button-control and card-pilot changes. Patch the current working
copies and preserve all unrelated work. Do not restore any file from `HEAD`.

The following remain read-only for this target unless a test proves a narrow
compatibility fix is required:

- raw/private data, runs and model weights
- `models/manifest.yaml` and all model admission metadata
- camera, gesture, speech, identity and card-model implementations
- dealer wire schemas and target vocabulary
- archived DeskMate and Baseline material

## Implementation Stages

1. Add explicit production hand start and hole-delivery reducer methods.
2. Add `SequentialPartBCoordinator` for hole, board and showdown phases.
3. Add a thin `HandRuntime` facade that follows, rather than duplicates,
   `HandPhase` and rotates between Part A and Part B.
4. Fix confirmed-card idempotency and require confirmed showdown slots.
5. Make Part A production defaults fail closed and synchronize fatal recovery
   with the authoritative engine.
6. Add a fully simulated registration-bound hand path through river/showdown,
   plus mismatch, duplicate ACK, unknown evidence and timeout tests.

## Validation

- Targeted game/runtime tests first.
- Practical full suite using `.venv/Scripts/python.exe -m pytest -q`.
- `git diff --check` and scoped `git status --short --branch`.
- No target camera, microphone, model retraining or physical motion.

## Physical Motion Status

No physical motion is authorized. All command/ACK behavior is exercised with
`SimulatedDealer`; target hardware still requires the Stage 3 safety procedure
and an operator.

## Implementation Result

- Added `HandRuntime` as the only Part A/Part B facade and added a frozen-roster
  production entry boundary.
- Added `SequentialPartBCoordinator` for eight private-card deliveries, five
  no-burn board deliveries and live-player showdown reveal.
- Production setup now enters `DEALING_HOLE`; the Stage 1 pre-dealt helper is
  retained but documented as non-product.
- Commands, ACKs and delivery-pending slot state are append-only logged.
  Duplicate ACKs are idempotent; restart after a successful dispense waits for
  vision, while restart with an unresolved command pauses.
- Card confirmation is idempotent, confirmed slots cannot silently downgrade,
  and same-slot identity changes are hard conflicts.
- Production showdown consumes only confirmed owned slots.
- Part A production defaults are fail closed and rotation, visual-settle and
  action windows now time out into the authoritative paused state.
- Direct button-to-engine submission requires an explicit pilot opt-in.
- Complete simulated hand validation covers 13 dispenses and 46 correlated
  command/ACK pairs.
- Full validation: `261 passed`; compileall and `git diff --check` passed.
- No physical motion was performed.

## Commit Intent

Do not commit, push, create a branch, publish a release or open a PR unless the
user explicitly asks.
