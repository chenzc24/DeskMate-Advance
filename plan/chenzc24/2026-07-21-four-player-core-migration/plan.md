# Four-Player Core Migration

## Outcome And Owned Paths

Replace the two-player Heads-Up Core assumption with a fixed four-seat/four-
player table so Button, small blind, big blind and UTG are distinct.  Migrate
rules, positions, multi-pot semantics, dealer/vision slots, schemas,
walkthroughs, tests and all active plans.  Apply the human decisions for S0-05,
S0-06, S0-07 and S0-12 exactly; add explicit decisions for board reveal,
between-hand reset and Button/turn indication.

Owned paths: active Poker Dealer governance, packaging, `configs/`,
`src/poker_dealer/domain/`, `tests/domain/`, `tests/contracts/`, `docs/`, and
this target plan.

## Dirty Paths Left Read-Only

DeskMate removals, archive pointers, camera migration, model manifest, camera
source/tests/scripts, ignored local assets and prior target plans remain
read-only unless an owned active document must be kept consistent.  Do not
modify the DeskMate archive branch or commit.

## External Dependencies

- Human confirmation of the final betting structure; Fixed-Limit remains the
  candidate and its numeric values remain defaults, not product freeze.
- Robotics evidence for four player delivery targets, selective board reveal,
  feeder, geometry, sensors, safety and wire protocol.
- Target-camera evidence for thirteen fixed visual slots and four seat
  orientations.

## Validation And Physical Motion

Validate four-player deal/action rotation for every Button seat, folded/all-in
seat skipping contracts, main/side-pot walkthrough semantics, exact Python/
JSON vocabulary agreement, all schemas/examples, all active document links,
stale Heads-Up assumptions, the full practical test suite and
`git diff --check`.  No camera recording, hardware connection or physical
motion is authorized.

Completed on 2026-07-21: `python -m pytest -q tests` passed 36 tests; all 17
JSON files parsed, every Draft 2020-12 schema and contract example validated,
all local schema references resolved, all 16 active Markdown files had valid
local links, the active tree contained no stale two-player/Heads-Up wording,
and `git diff --check` passed.  Validation used simulation and static evidence
only; no camera or robot was operated.

## Commit Intent

Do not commit or push unless the user explicitly requests it.
