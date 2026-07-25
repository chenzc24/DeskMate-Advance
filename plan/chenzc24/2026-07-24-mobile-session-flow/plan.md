# Mobile Session Flow Reorganization

## Outcome

Reorganize the phone console around the current state instead of displaying a
fixed bank of controls. Show only the one to three actions that are meaningful
at registration, hand, recovery, table-clearance, and between-hand boundaries.
Rename the ambiguous Replay action to Repeat voice and display it only when a
prompt exists.

Keep compact Camera, microphone, connection, and command-result feedback visible
on a 6.73-inch landscape phone. Use 44-pixel minimum touch targets, compact
role abbreviations, opt-in browser speech, and bounded reconnect backoff.

Separate table-clear confirmation from session termination. Reaching a
configured hand limit must first show a stable ready-to-end state and require an
explicit End session action; confirming Table clear must not immediately stop
the web service. Multi-hand live launches should allow the same registered
session to move from Table clear to Next hand without re-registration.
After End session, offer only New session and Stop UI. New session closes the
old perception session (clearing the memory-only face gallery), assigns a unique
session ID/log path, and relaunches enrollment at the same URL.

## Owned Paths

- `plan/chenzc24/2026-07-24-mobile-session-flow/plan.md`
- `src/poker_dealer/runtime/mobile_web_assets/index.html`
- `src/poker_dealer/runtime/mobile_web_assets/app.js`
- `src/poker_dealer/runtime/mobile_web_assets/styles.css`
- `src/poker_dealer/runtime/mobile_web_console.py`
- `src/poker_dealer/runtime/live_session.py`
- `scripts/runtime/run_hand.py`
- Scoped runtime tests for mobile and session-boundary behavior

## Dirty Read-Only Paths

Preserve all unrelated card-data, model, network configuration, and prior
runtime changes. Reuse the existing mobile console implementation without
replacing its visual system.

## External Dependencies

- Existing local mobile web console, camera stream, and AudioRelay microphone.
- No new packages, hosted services, or runtime downloads.

## Validation

- Targeted mobile/session/CLI tests: 35 passed.
- Runtime suite: 157 passed.
- Practical full Python suite: 402 passed.
- Mobile JavaScript syntax check: passed.
- Browser layout/state inspection: registration showed Camera, microphone,
  connection, feedback, face framing, and only Capture, Repeat voice, and
  Reset. At 1280x720 the document and body both matched the viewport exactly
  after the compact-height fix; all visible controls measured 50 pixels high.
- Mobile landscape CSS enforces 44-pixel controls, compact status pills,
  one-line feedback, and no fixed full-role labels.
- `git diff --check`: passed (line-ending notices only).
- `git status --short --branch`: inspected; unrelated dirty files preserved.
- Live four-human relaunch uses a 100-hand session cap so normal Table clear
  transitions to Next hand / End session. The web health endpoint returned OK,
  the camera/face overlay and AudioRelay microphone were live, and diagnostics
  started at
  `runs/diagnostics/robot_camera_audiorelay/live-20260724T082913.410208Z`.

## Physical-Motion Status

No physical motion is authorized. Validation uses the simulated dealer and
read-only camera/audio/UI state.

## Commit Intent

No commit, branch, push, release, or pull request unless explicitly requested.
