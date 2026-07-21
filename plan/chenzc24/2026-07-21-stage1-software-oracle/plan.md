# Stage 1 Software Oracle

## Outcome And Owned Paths

Implement the complete no-camera/no-robot four-player software oracle: evented
hand state and recovery, state-controlled action evidence promotion, candidate
Fixed-Limit betting, authoritative digital ledger and side pots, deterministic
5/7-card evaluation and settlement, three simulators, executable Stage 0
walkthroughs, CLI replay, property/fault tests and at least 10,000 randomized
legal hands.

Owned paths: `src/poker_dealer/game/`, `scripts/game/`, `tests/game/`, exports
from `src/poker_dealer/game/__init__.py`, Stage 1 evaluation/status documents,
and this target plan. Existing domain/config/schema contracts are read-only
unless a proven incompatibility requires an explicit migration.

## Dirty Paths Left Read-Only

All DeskMate removals/archive, camera/perception implementation, robotics,
models/manifest and model assets, raw/derived data, existing contract/rule
configs, other stages and prior plans remain read-only.

## External Dependencies

- S0-07 product confirmation remains external. The current Fixed-Limit config
  is implemented as a candidate adapter and cannot be labelled product release.
- No camera, model, firmware or hardware is required for Stage 1.

## Validation And Physical Motion

Run targeted unit/golden/fault/property/recovery tests, all 18 executable
walkthrough replays, at least 10,000 randomized legal hands, the full practical
suite, import-boundary checks, CLI smoke, JSON/link checks and
`git diff --check`. No physical motion, camera capture or device connection is
authorized.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Validation

- `python -m pytest -q tests/game`: 56 passed.
- `python -m pytest -q tests`: 93 passed.
- All 18 frozen Stage 0 walkthroughs executed and matched expected outcomes.
- 10,000 seeded legal four-player hands settled: 84,644 actions, 422
  showdowns and 9,578 fold endings; no invariant failure.
- No-device full-hand CLI settled with all 13 visible cards confirmed, 21
  displayed transitions, 26 hash-chained events and a schema-valid snapshot.
- Stage 1 import-boundary scan found no CV, model-framework, serial, device or
  runtime imports. No camera, robot connection or physical motion was used.
- S0-07 remains an external product-release decision; the implemented
  Fixed-Limit reducer remains a tested candidate.
