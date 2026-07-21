# Stage 2 Two-Person Worksplit Documentation

## Outcome And Owned Paths

Document the current two-person Stage 2 division: track A owns multimodal player-action evidence and track B owns fixed-slot card/table-scene evidence. Define exclusive paths, shared read-only contracts, independent handoff Gates, Laptop-first work and the later integration boundary.

Owned paths are `docs/plans/STAGE2_TWO_PERSON_WORKSPLIT.md`, the reference links added to `docs/plans/POKER_DEALER_MASTER_PLAN.md` and `docs/stages/STAGE_2_CARD_PERCEPTION.md`, and this plan.

## Dirty Paths Left Read-Only

All archived DeskMate changes, Stage 0/1 contracts and implementations, perception source/config/tests, model assets and manifest, Robotics artifacts, private media and unrelated plans remain read-only. This target documents the current state and does not change product contracts or model status.

## External Dependencies

The document relies on the existing Stage 0 decision register, Stage 1 Gate report, current Stage 2A gesture/multimodal reports and the Stage 2 Gate definitions. Target-camera, participant/session, deck/session and Robotics evidence remain external/open.

## Validation And Physical Motion

Verify all local Markdown links, referenced paths, track ownership, Gate labels and current development/open status. Run `git diff --check` and scoped `git status --short --branch`. Documentation only: no camera/audio capture, robot connection or physical motion.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Validation

- Added the standalone two-person Stage 2 work split and linked it from both the master plan and Stage 2 authority document.
- Verified all local Markdown links in the three touched documentation files.
- Confirmed the documented current action baseline with `23 passed` targeted action/fusion tests before writing the status.
- `git diff --check` passed; only pre-existing line-ending warnings outside this target were reported.
- No source, config, model, media, hardware or physical-motion state was changed.
