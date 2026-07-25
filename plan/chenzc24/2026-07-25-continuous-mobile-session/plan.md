# Continuous Mobile Session

## Outcome

Keep one browser connection and one enrolled player roster across multiple poker
hands. A normal between-hand transition is `table clear -> next hand`; a true
new session remains available for replacing players and must reconnect the
browser video automatically without a manual refresh.

## Owned paths

- `src/poker_dealer/runtime/mobile_web_assets/app.js`
- `tests/runtime/test_mobile_web_console.py`
- this plan

## Dirty read-only paths

All other existing worktree changes are preserved. Session/game semantics and
the memory-only identity gallery are not changed.

## Decisions

- Multiple hands with the same players use the existing `max_hands > 1`
  session loop. Registration, ledger continuity, Button rotation, browser
  controller ownership, and the camera process stay live.
- `New session` is reserved for a new roster/session. The backend intentionally
  clears memory-only identity data and reopens devices.
- When the WebSocket reconnects after that controlled backend cycle, the
  browser reloads `/video.mjpeg` with a cache-busting query so the camera view
  recovers without a page refresh.
- The current A/D test launch uses 20 hands, matching the existing acceptance
  horizon. The operator may end the session earlier after confirming table
  clearance.

## Validation

- JavaScript syntax check.
- Mobile web console asset tests.
- Practical full Python suite.
- Live restart with `--max-hands 20`, HTTP health check, and process command
  inspection.
- `git diff --check` and scoped status.

Completed:

- JavaScript syntax check passed.
- Targeted runtime/UI tests: 39 passed.
- Full suite: 433 passed.
- Live A/D runtime restarted with `--max-hands 20`.
- HTTP health check passed and the running command was inspected.
- `git diff --check` passed; existing Windows line-ending warnings remain.

## Physical-motion status

No physical motion or hardware command is authorized.

## Commit intent

Do not commit or push unless explicitly requested.
