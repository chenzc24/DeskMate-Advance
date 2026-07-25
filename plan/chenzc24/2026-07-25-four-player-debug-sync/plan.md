# Four-Player Debug Sync

## Outcome

Audit every fix proven in the two-human A/D live path against the formal
four-human runtime, close any shared-runtime gaps, validate the formal
four-player and dual-camera paths, then commit and push the complete intended
worktree to `main`.

## Owned paths

- Shared runtime/perception/UI/config/test paths already modified by the
  completed 2026-07-25 targets
- Any narrowly required four-player regression tests
- this plan

## Dirty scope

The current worktree contains the complete, related sequence of uncommitted
runtime revisions requested in this conversation: unified live profiles,
full-frame flop binding and timeout, stale action guards, visible settlement,
card overlays and announcements, explicit street/showdown prompts, and mobile
continuous-session recovery. The user has asked to synchronize these to the
formal four-player path and push them together.

No raw/private media, run logs, credentials, face embeddings, or external
workspace files may be staged.

## Audit matrix

- Registration: four real participants; no simulator-only bypass.
- Player turns: all four seats use the authoritative acting-seat focus and
  actor-bound multimodal confirmation.
- Hand boundary: roster, ledger, Button rotation, and controller survive across
  hands.
- Cards: flop batch and turn/river/showdown recognition work on the table route.
- Results: showdown entry, card confirmations, awards, completion, and UI
  persistence work for two to four live players.
- Camera routing: formal `robot_camera_audiorelay` uses phone player camera and
  robot table camera; hardware remains fail-closed.
- Browser: WebSocket and MJPEG recover automatically after a true new-session
  cycle.
- Cross-hand isolation: reset card confirmation/batch caches and identity/action
  temporal evidence on `hand_id` change while retaining the consented
  session-level face/speaker enrollment galleries.

## Validation

- Targeted four-player state-machine, registration, multimodal, session,
  announcer, card, profile, UI, replay, and CLI tests.
- Formal live preflight/config parsing without opening hardware.
- Full Python suite, JavaScript syntax, JSON parsing, `git diff --check`.
- Inspect staged paths for prohibited data and unintended files.

Completed:

- Four-player targeted suite: 158 passed.
- Cross-hand context regression subset: 36 passed.
- Full suite: 435 passed.
- Twenty-hand four-player replay: 20/20 settled, every hand log and the session
  log passed, Button returned to seat A after rotation.
- `robot_camera_audiorelay` live preflight passed with all local perception
  assets and 13 logical slots.
- `robot_hardware` remains intentionally fail-closed because protocol, safety,
  and target geometry are pending.
- All config JSON parsed, browser JavaScript syntax passed, Python source
  compiled, and `git diff --check` passed.

## Physical-motion status

No physical motion. The real hardware adapter remains fail-closed.

## Commit intent

Commit the related worktree to `main` and push `main` to `origin`, as explicitly
requested. Do not open a PR.
