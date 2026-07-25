# Card Recognition Feedback

## Outcome

Add visible card detection boxes, labels, confidence, and audible confirmation to
the shared Live Runtime. The same implementation serves the current phone-front
test profile and the formal split-camera profiles where the phone supplies
identity/action/audio and the robot camera supplies card frames.

## Owned paths

- `src/poker_dealer/runtime/live_perception.py`
- `src/poker_dealer/runtime/mobile_web_console.py`
- `src/poker_dealer/runtime/mobile_web_assets/index.html`
- `src/poker_dealer/runtime/mobile_web_assets/app.js`
- `src/poker_dealer/runtime/mobile_web_assets/styles.css`
- `src/poker_dealer/runtime/announcer.py`
- `configs/runtime/announcements_en.json`
- `tests/runtime/test_mobile_web_console.py`
- `tests/runtime/test_announcer.py`
- this plan

## Dirty read-only paths

All other pre-existing modified and untracked paths are preserved. The game
rules, betting implementation, runtime profiles, card model, and prior
result-flow changes are not rewritten by this target.

## Scope and decisions

- Overlays come from in-memory detector evidence; no frames are persisted.
- Fixed-ROI boxes are translated back into full-frame coordinates.
- Browser and optional desktop preview receive the same overlay evidence.
- Speech occurs only on the first committed `confirmed` transition.
- A completed community-card batch announces the confirmed street explicitly.
- Showdown start is announced on the first committed transition into showdown;
  settlement announces showdown completion after the award and before overall
  hand completion.
- The implementation is shared below camera-profile routing.
- Core v1 remains deterministic Fixed-Limit; physical chip recognition remains
  disabled and is outside this target.

## External dependencies

Existing local OpenCV detector, announcement catalog, browser speech mirror, and
Windows speech output.

## Validation

- Targeted mobile-console and announcer tests.
- Runtime-profile tests for phone and robot configurations.
- JSON parsing and JavaScript syntax validation.
- Practical full Python test suite.
- `git diff --check` and scoped `git status --short --branch`.

Completed:

- Targeted runtime tests after explicit street/showdown prompts: 32 passed.
- Full suite: 433 passed.
- Announcement and phone/robot runtime JSON parsed.
- Browser JavaScript syntax check passed.
- Python runtime compile check passed.
- `git diff --check` passed; only the existing Windows line-ending warnings
  remain.

## Physical-motion status

No physical commands or motion. Robot-camera compatibility is software/profile
validation only.

## Commit intent

Do not commit or push unless the user explicitly asks.
