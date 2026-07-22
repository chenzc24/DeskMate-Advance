# Registration, Controls And Announcer

## Outcome

- Keep `seat_a` through `seat_d` as internal physical targets only; present the
  current-hand roles `Button`, `Small Blind`, `Big Blind`, and `UTG` to users.
- Require the live four-player setup to select the initial Button explicitly
  instead of treating the first registration target as Button by default.
- Extract registration sequencing from the monolithic live Part A script into
  a tested `RegistrationRuntime`.
- Add one semantic control contract shared by laptop keys and a future robot
  button adapter.  Laptop input remains the operator fallback.
- Add an event-driven announcer boundary and a laptop test implementation;
  announcements consume committed runtime events and never mutate game state.
- Make session voice enrollment observable and finite: three longer prompted
  English phrases, per-sample accepted/retry feedback, automatic completion,
  and `V` cancellation.  Audio is discarded while laptop TTS is prompting so
  the computer voice is not enrolled as the participant.
- Confirm Fixed-Limit for Core v1 while keeping stakes, stack, cap, and timeout
  as configurable defaults.
- Keep the digital ledger authoritative.  Near-term physical-chip vision is
  non-authoritative reconciliation evidence (Plan A); authoritative physical
  chip accounting remains a later Plan B investigation.

## Owned Paths

- `src/poker_dealer/domain/game.py`
- `src/poker_dealer/domain/controls.py`
- `src/poker_dealer/domain/__init__.py`
- `src/poker_dealer/runtime/registration.py`
- `src/poker_dealer/runtime/announcer.py`
- `src/poker_dealer/runtime/button_betting.py`
- `src/poker_dealer/runtime/__init__.py`
- `scripts/perception/live_sequential_part_a.py`
- `scripts/runtime/laptop_button_betting_pilot.py`
- `src/poker_dealer/game/engine.py`
- `configs/game/core_v1.json`
- `configs/contracts/core_rules.schema.json`
- `docs/contracts/CORE_INTERFACES.md`
- `docs/contracts/GAME_RULES.md`
- `docs/architecture/system-architecture.md`
- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/stages/STAGE_0_SCOPE_AND_CONTRACTS.md`
- `docs/stages/STAGE_1_GAME_ENGINE_SIMULATOR.md`
- `docs/evaluation/stage-0-gate-audit.md`
- `docs/evaluation/stage-1-gate-test.md`
- scoped runtime/domain/contract tests

## Dirty Read-Only Paths

The following pre-existing Part B work is unrelated and must not be modified:

- `docs/evaluation/stage2b-lgd-card-pilot.md`
- `scripts/perception/live_card_pilot.py`
- `tests/perception/cards/test_live_card_pilot_cli.py`
- `plan/chenzc24/2026-07-22-card-pilot-robot-stream/`

## External Dependencies

- Laptop speech output uses an optional adapter and must degrade to console or
  disabled output when the host TTS facility is unavailable.
- The robot button transport is intentionally not chosen here; robotics will
  implement the same semantic control observation contract.

## Validation

- Dynamic role assignment, registration sequencing, laptop/robot semantic
  controls, announcer templates, and button-to-ledger submission have targeted
  tests.
- Targeted runtime/domain/contract/game validation: `28 passed`.
- Practical full suite: `248 passed` using the project `.venv`.
- JSON parse/schema validation is covered by the contract suite; final
  `git diff --check` and scoped status are recorded at handoff.

## Physical Motion Status

No physical motion is authorized.  The existing live pilot continues using the
simulated dealer for rotation ACKs.  Robot button support is an input interface
only and does not authorize motor commands.

## Commit Intent

Do not commit, push, branch, release, or open a PR unless the user asks.
