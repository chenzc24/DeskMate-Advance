# Visible Settlement Result And Announcement

Status: `completed`

Date: `2026-07-25`

Owner: `chenzc24`

## Outcome

Make both terminal paths observable to the operator:

- uncontested settlement shows and announces winner and awarded amount;
- showdown shows and announces winner, winning hand category and awarded amount;
- the result remains visible after the runtime moves to table-clearance state.

The deterministic engine remains the only settlement authority. UI and speech
only render committed engine state.

## Owned Paths

- `src/poker_dealer/runtime/mobile_web_console.py`
- `src/poker_dealer/runtime/mobile_web_assets/index.html`
- `src/poker_dealer/runtime/mobile_web_assets/app.js`
- `src/poker_dealer/runtime/mobile_web_assets/styles.css`
- `src/poker_dealer/runtime/announcer.py`
- `src/poker_dealer/game/engine.py`
- `configs/runtime/announcements_en.json`
- `tests/runtime/test_mobile_web_console.py`
- `tests/runtime/test_announcer.py`
- this plan

## Read-only Context

- `plan/wuqi-ii/2026-07-25-game-flow-revision/plan.md`
- current dirty card, runtime-profile and stale-evidence work
- game engine and evaluator implementations

## Root Cause

The engine already commits awards, but the mobile console immediately replaces
the settled hand view with table clearance and does not render awards.
Furthermore, the recovery-priority `table_not_clear` prompt clears queued
information-priority winner announcements before Windows TTS can play them.

## External Dependencies

- Windows `System.Speech` output device for audible TTS.
- Browser speech remains optional and is not game-state authority.

## Validation

- Targeted runtime/game suite: `65 passed`.
- Practical full Python suite: `431 passed`.
- Announcement JSON parsed successfully.
- `node --check src/poker_dealer/runtime/mobile_web_assets/app.js` passed.
- `git diff --check` passed; only expected Windows CRLF notices were emitted.
- Scoped `git status --short --branch` reviewed; unrelated and earlier dirty work
  remains preserved.

## Result

- A committed settlement now produces a structured UI result containing winner
  seat/player, reason, award and optional showdown hand category.
- The result persists from the settled hand view into table clearance.
- The table-clear prompt no longer clears queued winner announcements.
- Showdown settlement announcements include the winning hand category.

## Physical Motion

No physical motion is authorized or executed. This change is UI and speech
feedback only.

## Commit Intent

Do not commit or push unless the user explicitly asks.
