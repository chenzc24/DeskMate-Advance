# Poker Dealer Stage 0 Contract Freeze

## Outcome And Owned Paths

Complete the non-hardware portion of Stage 0: freeze Core v1 game semantics,
logical table/vision slots, domain schemas, dealer protocol semantics, recovery
policy, rule walkthroughs and the Gate 0 evidence matrix.  Leave claims that
require a physical feeder, camera sample or Robotics sign-off explicitly open.

Owned paths:

- `configs/game/`
- `configs/contracts/`
- `configs/table/`
- `AGENTS.md`
- `src/poker_dealer/domain/`
- `docs/contracts/`
- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/stages/STAGE_0_SCOPE_AND_CONTRACTS.md`
- `docs/evaluation/stage-0-gate-audit.md`
- `tests/domain/`
- `tests/contracts/`
- this target plan

## Dirty Paths Left Read-Only

All DeskMate-removal and Poker Dealer-pivot changes already present in the
worktree remain read-only except the owned overlapping Poker Dealer paths above.
Ignored datasets, model assets, environments, runs and artifacts remain
read-only.  The archive branch/commit is not modified.

## External Dependencies

- Robotics evidence for feeder choice, dimensions, target geometry, sensors,
  interlock, E-stop, watchdog and command latency.
- Target-camera images for field of view, ROI pixel size, glare and occlusion.
- Human sign-off on the frozen Core/Plus boundary and demo parameters.

No Baseline or DeskMate implementation/model dependency is permitted.

## Validation And Physical Motion

Parse all JSON/schema files; validate examples against schemas where supported;
run at least ten rule/fault walkthroughs, domain/config invariant tests, the
full practical Python suite, link/reference checks, `git diff --check` and
scoped status.  No camera recording, hardware connection or physical motion is
authorized by this target.

Completed on 2026-07-21: 17 JSON/config files parse; Draft 2020-12 schemas and
all examples validate; Python enums exactly match rules, targets, slots,
commands, ACK states and errors; 12 walkthroughs cover the required normal and
fault cases; unsafe/ambiguous observations and success ACKs are rejected; all
16 local Markdown links resolve; 61 active text files pass whitespace/final
newline checks; the full active suite passes (`30 passed in 0.34s`); and
`git diff --check` passes.  Decision status is intentionally 4 frozen,
4 partially frozen and 4 evidence-required.  No camera recording, hardware
connection, model work or physical motion occurred.

## Commit Intent

Do not commit or push Stage 0 changes unless the user explicitly requests it.
