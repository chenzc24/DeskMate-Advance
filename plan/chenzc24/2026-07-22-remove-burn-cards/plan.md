# Remove Burn Cards From Core v1

## Outcome

- Freeze the human product decision that Core v1 uses a simplified no-burn
  dealing procedure.
- Keep four-player Texas Hold'em betting, board composition and showdown
  unchanged: eight private hole cards and five community cards are still dealt.
- Retain `burn_before_each_board_street` as an explicit rules field, but freeze
  it to `false` so the omission is deliberate and machine-verifiable.
- Remove `burn_tray` from the active mechanical target vocabulary, logical table
  layout and dealer protocol.  The active target count becomes nine: four seat
  targets and five board targets.
- Change board delivery order to:
  - Flop: `board_flop_1`, `board_flop_2`, `board_flop_3`.
  - Turn: `board_turn`.
  - River: `board_river`.
- Require one successful rotate/dispense acknowledgement per real delivered
  card.  Removing burn cards does not weaken command correlation, single-card
  evidence, visual confirmation or recovery rules.
- A complete hand through the river now dispenses thirteen cards: eight hole
  cards and five board cards.

## Product And Compatibility Decision

- Document the procedure as `robot_core_no_burn`; do not describe it as casino
  burn-card procedure.
- Add a new frozen Stage 0 decision recording the explicit human supersession,
  and update older decision text that currently claims ten targets or face-down
  burn cards so the active register contains no contradictions.
- Treat removal of `burn_tray` from dealer messages as a breaking protocol
  change.  Recommended contract action: issue dealer protocol schema 2.0 and
  reject old `burn_tray` commands instead of silently mapping them elsewhere.
- No compatibility alias for `burn_tray` is permitted in the Core runtime.  An
  old robotics adapter must fail version negotiation before motion.

## Owned Paths

### Rules And Decision Authority

- `configs/game/core_v1.json`
- `configs/contracts/core_rules.schema.json`
- `configs/contracts/stage0_decisions.json`
- `docs/contracts/GAME_RULES.md`
- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/stages/STAGE_0_SCOPE_AND_CONTRACTS.md`

### Mechanical And Protocol Contracts

- `src/poker_dealer/domain/dealer.py`
- `src/poker_dealer/domain/game.py`
- `configs/table/logical_layout_v1.json`
- `configs/contracts/table_layout.schema.json`
- `configs/contracts/dealer_message.schema.json`
- `docs/contracts/CORE_INTERFACES.md`

### Tests And Compact Evidence

- `tests/domain/test_game_contract.py`
- `tests/contracts/test_json_contracts.py`
- `tests/domain/test_contracts.py` when protocol version assertions require it
- `tests/game/test_simulators.py` when target enumeration/count assertions
  require it
- scoped Stage 0/Stage 1 gate evidence that explicitly reports target or
  dispense counts

## Dirty And Read-Only Coordination

The working tree already contains broad uncommitted registration, control,
announcer, Fixed-Limit and Part B robot-stream work.  Several intended owned
paths overlap those changes, especially:

- `configs/game/core_v1.json`
- `configs/contracts/core_rules.schema.json`
- `docs/contracts/GAME_RULES.md`
- `docs/contracts/CORE_INTERFACES.md`
- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/stages/STAGE_0_SCOPE_AND_CONTRACTS.md`
- `src/poker_dealer/domain/game.py`
- `tests/contracts/test_json_contracts.py`

Implementation must patch the current working copies and preserve every
unrelated change; it must not restore these files from `HEAD` or overwrite the
other target plans.

The following Part B pilot paths are read-only for this target because burn
cards are not vision slots and require no model change:

- `scripts/perception/live_card_pilot.py`
- `src/poker_dealer/perception/cards/`
- `configs/perception/cards_lgd_pilot.json`
- `docs/evaluation/stage2b-lgd-card-pilot.md`
- `tests/perception/cards/`
- `plan/chenzc24/2026-07-22-card-pilot-robot-stream/`

Part A identity, gesture, speech, actor-binding and ledger code are also
read-only.  Removing burn cards must not change player-action order or action
acceptance.

## Implementation Stages

### Stage 1: Freeze The No-Burn Rule

1. Set `deal.burn_before_each_board_street` to `false`.
2. Change the core rules schema from `const: true` to `const: false` and bump
   the rules contract/schema version because the frozen dealing procedure has
   changed.
3. Record a new Stage 0 decision: no burn cards, nine mechanical targets,
   thirteen dispensed cards through a complete river.
4. Reconcile S0-01, S0-03, S0-10 and S0-13 wording so none still requires a
   burn target or burn-card orientation.
5. State explicitly that board reveal, ACK safety and manual return remain
   unchanged.

Gate: the machine-readable rules and human rules document agree on no-burn
delivery and contain no active contradictory statement.

### Stage 2: Remove The Mechanical Target

1. Remove `DealerTargetSlot.BURN_TRAY`.
2. Remove `burn_tray` and `face_down_burn_card` from table layout data/schema.
3. Reduce the documented mechanical target count from ten to nine.
4. Remove `burn_tray` from the dealer command/ACK schema and bump the protocol
   version as a breaking change.
5. Require robotics adapters to negotiate the new version before commands are
   accepted.

Gate: no active layout or protocol can construct, validate or transmit a
`burn_tray` target.

### Stage 3: Simplify Board Deal Ordering

1. Update `board_deal_targets(Street.FLOP)` to return exactly three board
   targets.
2. Update Turn and River to return one-element tuples containing only their
   board target.
3. Do not modify `VisionSlot`, `STREET_BOARD_SLOTS`, card recognition or
   `confirm_board_dealt()`; these already represent only the five real board
   cards.
4. Ensure the future Part B coordinator consumes the revised target list and
   never carries a special burn state.

Gate: board target counts are Flop 3, Turn 1 and River 1, and no public function
returns a burn target.

### Stage 4: Update Tests And Evidence

1. Replace burn-order assertions with exact no-burn board-order assertions.
2. Update target vocabulary/schema parity tests from ten targets to nine.
3. Add a negative contract test proving `burn_tray` is rejected by the current
   dealer protocol.
4. Add a complete-deal count assertion: eight hole deliveries plus five board
   deliveries equals thirteen successful dispenses.
5. Preserve tests proving every retained delivery still requires matching
   rotate and dispense ACKs.
6. Preserve Part B tests proving all five board slots require stable visual
   confirmation and unknown/duplicate evidence cannot advance state.

Gate: scoped domain, contract, game and simulator tests pass before the full
suite is run.

### Stage 5: Documentation And Robotics Handoff

1. Update the master plan, interfaces and Stage 0 diagrams to show nine
   mechanical targets.
2. Give robotics the exact removed target and revised Flop/Turn/River command
   counts.
3. State that an existing physical burn tray may remain mechanically unused,
   but it is not calibrated, commanded or accepted by Core v1 software.
4. Mark all older protocol examples containing `burn_tray` obsolete rather
   than silently replayable.

Gate: DL, software and robotics use the same target vocabulary and command
counts.

## Validation

- Parse every modified JSON file.
- Validate rules, layout and dealer examples against their current schemas.
- Run targeted tests:
  - `tests/domain/test_game_contract.py`
  - `tests/domain/test_contracts.py`
  - `tests/contracts/test_json_contracts.py`
  - `tests/game/test_simulators.py`
  - `tests/game/test_engine.py`
  - `tests/game/test_scene_and_snapshot.py`
- Run the practical full suite with `.venv/Scripts/python.exe -m pytest -q`.
- Run an active-tree search for `burn_tray`, `BURN_TRAY`,
  `face_down_burn_card`, `burn_before_each_board_street: true` and Chinese
  burn-card requirements.  Matches are allowed only in the migration plan or
  explicitly labelled historical material.
- Run `git diff --check` and scoped `git status --short --branch`.

## Implementation Result

- Core rules are now v1.3 with `dealing_profile=robot_core_no_burn` and
  `burn_before_each_board_street=false`.
- The active domain, table layout and dealer protocol expose nine targets and
  no `burn_tray`; dealer message examples/schema are v2.0.
- Board delivery order is Flop 3, Turn 1 and River 1 with no intermediate
  target.
- S0-22 records the explicit product decision and the active Stage 0 register,
  master plan, rules, interfaces and mechanism documentation are reconciled.
- Negative protocol validation rejects `burn_tray`.
- A simulator test proves eight hole deliveries plus five board deliveries
  produce exactly thirteen successful dispenses.
- Scoped contract/domain/simulator validation: `40 passed`.
- Practical full suite: `249 passed` using the project `.venv`.
- No physical motion was performed.

## Non-Goals

- No Part A model, identity, speech, gesture or action-state change.
- No card-recognition retraining or dataset change.
- No change to five-card board composition, hand ranking, pot settlement or
  showdown rules.
- No implementation of the missing Part B coordinator in this target.
- No physical motion test or authorization.

## Physical Motion Status

This plan authorizes no physical movement.  Protocol and simulator tests must
precede any robotics verification.  Later physical testing still requires an
operator, clear area, guards, low force/speed, homing, sensors, watchdog and
manual stop.

## Commit Intent

Do not commit, push, create a branch, publish a release or open a PR unless the
user explicitly asks.
