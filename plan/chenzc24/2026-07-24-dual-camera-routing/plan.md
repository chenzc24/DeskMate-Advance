# Dual-Camera Perception Routing

## Outcome

Add an explicit, backwards-compatible live camera route where the optional
player camera supplies registration, session face verification, pose and
gesture observations, while the existing table camera supplies card delivery,
card recognition and table-view observations. Profiles without a player camera
continue to use the existing single shared camera.

The first target configuration uses the Windows `DroidCam Video` virtual camera
for the player route and the Raspberry Pi MJPEG stream for the table route.
Camera routing remains state-owned infrastructure; perception observations do
not select cameras or advance the game.

## Owned Paths

- `plan/chenzc24/2026-07-24-dual-camera-routing/plan.md`
- Runtime profile/schema and robot-camera AudioRelay configuration
- Camera application lifecycle and runtime frame-routing module
- Hand loop and live entry-point wiring
- Scoped runtime/profile/camera tests and architecture documentation

## Dirty Read-Only Paths

Preserve all unrelated existing work, including card-data/model work, the
mobile UI, A/D development mode, audio changes and prior plans except where
their active runtime wiring must call the new compatible camera interface.

## External Dependencies

- DroidCam Windows virtual camera, currently exposed through OpenCV MSMF index
  1 at 1280x720 and 30 FPS.
- Raspberry Pi MJPEG stream configured through the shared network endpoints.
- No new package or runtime download.

## Validation

- Targeted profile, camera lifecycle, frame-route and live CLI tests: 64 passed.
- Runtime suite: 154 passed.
- Practical full Python suite: 399 passed.
- All five runtime profiles validate against the JSON schema and parse.
- Non-recording live probe opened both routes together:
  `droidcam_player_camera` at 1280x720/30 FPS and
  `robot_mjpeg_camera` at 640x480/25 FPS.
- `git diff --check`: passed (line-ending notices only).
- Full four-human mobile live session restarted with both camera resource locks,
  registration on the player route, mobile health `ok`, and live AudioRelay
  callbacks. Diagnostics started at
  `runs/diagnostics/robot_camera_audiorelay/live-20260724T072133.437395Z`.
- Scoped `git status --short --branch`: inspected; unrelated dirty files
  preserved.

## Physical-Motion Status

No physical motion is authorized. All live validation uses the simulated dealer
and must not command robot hardware.

## Commit Intent

No commit, branch, push, release or pull request unless explicitly requested.
