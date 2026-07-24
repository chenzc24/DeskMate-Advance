# Two-Human Live Test Scenario

## Outcome

Add an explicit development-only live test scenario for a four-seat Core v1
hand where seats A and D use the normal human face/voice/action path, while
seats B and C are clearly identified simulator participants and automatically
submit legal fold evidence when they receive action focus.

Expose the live human identity/action gate with a minimal runtime overlay:
one face box and one short status for verifying, verified/listening, identity
lost, and pending speech. Identity remains automatic. Start the 30-second
player-action timeout only after identity succeeds, while preserving a bounded
attention timeout for reaching that point. Record bounded transcript/confidence
and rejection reasons, but never audio or embeddings.

Temporarily make speaker similarity advisory in both the normal four-human mode
and the explicit A/D development scenario because one-word speaker embeddings
are not yet calibrated. Preserve the current face/pose actor binding, English
speech-recognition confidence, legal-action validation, state-version checks,
and bounded audit fields. Registration is temporarily face-only in both modes:
it does not request voice phrases, create speaker profiles, or block face
enrollment when the microphone is unavailable. The bounded microphone stream
remains open for game-stage English recognition, and no audio is persisted. A
clearly recognized game action proceeds directly; gameplay no longer waits for
E or spoken confirmation. Registration still uses E for deliberate face
capture. Show a recognized target-owned gesture as one small green wrist marker
over the video, without adding another panel.

Use the live gesture pilot's configured confirmation score as the live hand
engine promotion threshold. This removes the inconsistent path where the
gesture temporal adapter emitted a valid candidate at 0.60 or above but the
engine rejected it against an unrelated 0.90 default. Replay and other
non-live constructors retain the fail-closed default unless they explicitly
provide a promotion policy.

Pass that same live promotion policy to the development-only B/C auto-fold
source. Its deterministic simulated evidence must satisfy the configured
stable-frame and stable-duration gates before entering the normal engine path;
the engine itself receives no simulator bypass.

The scenario must preserve the normal dependency direction and must not bypass
identity attribution, action validation, the deterministic game engine, ledger
updates, or the state-version checks.

## Owned Paths

- `plan/chenzc24/2026-07-24-two-human-live-test/plan.md`
- `scripts/runtime/run_hand.py`
- `src/poker_dealer/runtime/registration.py`
- `src/poker_dealer/runtime/live_perception.py`
- `src/poker_dealer/runtime/mobile_web_console.py`
- `src/poker_dealer/runtime/mobile_web_assets/app.js`
- `src/poker_dealer/runtime/mobile_web_assets/styles.css`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `src/poker_dealer/runtime/two_human_test.py`
- Scoped runtime tests for the scenario

## Dirty Read-Only Paths

Preserve all pre-existing unrelated changes, including card-data v3 plans,
augmentation utilities, tests, and previous network/UI configuration work not
required by this scenario.

## External Dependencies

- Existing live camera, microphone, speaker, and mobile web console.
- Existing simulated dealer adapter only.
- No new packages or runtime downloads.

## Validation

- Targeted A/D source/session/CLI tests: 23 passed.
- Runtime suite: 150 passed.
- Practical full Python suite: 388 passed.
- Mobile JavaScript syntax check: passed.
- `git diff --check`: passed (line-ending notices only).
- `git status --short --branch`: inspected; unrelated dirty files preserved.
- Live launch: mobile health endpoint returned `ok`; startup evidence records
  seats B/C as simulated with no face or voice enrollment claimed.
- Relaunched the A/D development session from registration after propagating
  the gesture threshold and matching the simulated auto-fold evidence to that
  policy. The configured camera endpoint returned HTTP 200, the mobile health
  endpoint returned `ok`, and the live websocket reported `voice_target=0`,
  `voice_active=false`, `microphone_live=true`, with active AudioRelay callback
  blocks. Diagnostics started at
  `runs/diagnostics/robot_camera_audiorelay/live-20260724T055046.255453Z`.

## Physical-Motion Status

No physical motion is authorized. The scenario must reject any non-simulated
dealer/hardware profile and is intended only for camera/audio/UI/state-machine
testing.

## Commit Intent

No commit, branch, push, release, or pull request unless explicitly requested.
