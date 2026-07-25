# Phone-Front A/D Live Test

## Outcome

Launch the development-only A/D full-hand runtime with one DroidCam front
camera shared by registration, player identity/action perception, board-card
recognition and showdown-card recognition. Keep B/C as explicit simulated
participants that auto-fold through normal runtime gates.

## Owned Paths

- `configs/runtime/phone_front_ad_audiorelay.json`
- `plan/chenzc24/2026-07-25-phone-front-ad-live-test/plan.md`

## Dirty Read-Only Paths

- Preserve every unrelated tracked, untracked and ignored path.

## External Dependencies

- DroidCam virtual camera at the locally probed MSMF device index 1.
- AudioRelay virtual microphone for optional English action commands.
- Existing local model assets; runtime downloads remain prohibited.

## Validation

- New JSON profile parsed successfully.
- Runtime preflight reported `ready=true`, local camera source `1`, simulated
  dealer, no physical motion and one shared camera route.
- The MSMF DroidCam source delivered 30/30 frames in bounded camera smoke.
- The one-hand A/D live runtime started with diagnostics enabled.
- The UI health endpoint returned `ok` and `/` returned HTTP 200.
- The visible UI showed the live phone-front feed and reached preflop with D as
  the acting player. Showing A at that point produced the expected
  `WRONG PLAYER · EXPECTING D` fail-closed state.
- The selected card geometry is
  `card_view_cycle_robot_development_v1.json`, so board and showdown card
  observations use the same shared phone feed in state-directed full-frame
  mode.
- `git diff --check` passed; scoped status contains only this profile and plan.

## Physical-Motion Status

No physical motion is authorized. The dealer adapter is simulated; card
placement and camera redirection are manual.

## Commit Intent

Do not commit, push, create a branch, release or PR.
