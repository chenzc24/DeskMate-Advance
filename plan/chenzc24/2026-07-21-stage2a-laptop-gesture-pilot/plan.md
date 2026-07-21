# Stage 2A Laptop Gesture Pilot

## Outcome And Owned Paths

Build a one-camera, one-focused-seat feasibility pilot for five configurable
static hand gestures. The pilot loads an official MediaPipe Gesture Recognizer
bundle offline, maps canned labels to the frozen poker action vocabulary,
applies ROI and temporal confirmation, and emits `PlayerActionObservation`
evidence without changing game state.

Owned paths: `configs/perception/actions_laptop_pilot.json`,
`src/poker_dealer/perception/actions/`, `scripts/perception/`,
`tests/perception/actions/`, the development entry in `models/manifest.yaml`,
`docs/evaluation/stage2a-laptop-gesture-pilot.md`, and this plan. The downloaded
model bundle stays ignored under `models/assets/`.

## Dirty Paths Left Read-Only

All archived DeskMate removals, Stage 0/1 source and contracts, card
perception, robotics, game rules, existing camera implementation, unrelated
model assets and prior plans remain read-only. The existing Poker Dealer model
manifest is updated only by adding the new development model metadata.

## External Dependencies

- Official Google MediaPipe Gesture Recognizer float16 task bundle and its
  published model/task documentation.
- Installed MediaPipe 0.10.35, OpenCV 5.0.0 and the Laptop camera.
- The five canned-gesture mappings are pilot defaults, not a frozen product
  interaction grammar or admitted model.

## Validation And Physical Motion

Validate config parsing, ROI clipping, label mapping, unknown/rejection,
temporal confirmation/cooldown, schema-compatible observations, offline model
loading, recorded/synthetic replay, practical full tests, and a bounded Laptop
camera smoke test. Camera frames are not saved. No robot connection or
physical motion is involved.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Validation

- Official float16 bundle downloaded, loaded offline and matched SHA-256
  `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`.
- Stage 2A action pilot tests: 11 passed; practical full suite: 104 passed.
- Four official public samples produced the expected canned labels; the
  ignored `Victory` label remained outside the Poker action mapping.
- Laptop DirectShow camera index 0 negotiated 1280x720 at about 30 FPS. A
  1,000-frame preview had zero missing reads, 27.34 effective FPS and 12.98 ms
  P95 model latency. No hand was visible, so the live five-action matrix
  remains open and no accuracy claim was made.
- No frames were saved. No robot connection or physical motion occurred.
