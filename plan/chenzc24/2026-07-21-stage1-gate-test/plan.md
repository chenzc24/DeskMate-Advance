# Stage 1 Gate Test

## Outcome And Owned Paths

Execute the current repository against the Stage 1 Gate and produce an honest
readiness result that distinguishes passing Stage 00 foundations from runnable
Stage 1 game-engine behaviour.

Owned paths: `docs/evaluation/stage-1-gate-test.md` and this target plan.

## Dirty Paths Left Read-Only

All source, tests, configs, models, datasets, robotics/camera work, archived
DeskMate material, existing plans and unrelated dirty paths remain read-only.
Testing does not authorize implementation fixes.

## External Dependencies

None for the current software audit. Full Gate 1 remains dependent on a Stage 1
state reducer, ledger/pot builder, evaluator, simulators, executable replay
harness and S0-07 product confirmation for the betting-reducer total Gate.

## Validation And Physical Motion

Collect and run the practical test suite, isolate domain/contract results,
inventory Stage 1 source/tests/executors, map evidence to every Gate 1 clause,
validate the evaluation report and run `git diff --check`. No camera, model,
hardware connection or physical motion is involved.

Completed on 2026-07-21: 37 tests collected and passed; the four-position game
contract subset passed 11 tests and the domain/JSON contract subset passed 15.
Inventory confirmed zero `tests/game` files, no executable walkthrough runner,
state reducer, ledger/pot builder, evaluator, simulator, recovery or 10,000-hand
test. The evaluation therefore records a foundation sub-Gate pass and Stage 1
Gate not runnable. `git diff --check` passed; no device was opened.

## Commit Intent

Do not commit or push unless the user explicitly requests it.
