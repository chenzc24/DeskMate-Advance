# Stage 2A Multiseat Laptop UI

## Outcome And Owned Paths

Add a four-seat Laptop feasibility UI that detects up to four hands, assigns
each hand to exactly one configurable fixed seat ROI, and allows only the
state-selected focus seat to enter temporal action confirmation. Seat means a
fixed table region, never biometric identity. The quadrant layout is a Laptop
test fixture and not target-table geometry.

Owned paths are `configs/perception/actions_multiseat_laptop_pilot.json`, the
multiseat additions under `src/poker_dealer/perception/actions/`, the new live
UI under `scripts/perception/`, scoped action tests, Stage 2A evaluation docs,
and this plan.

## Dirty Paths Left Read-Only

Game rules, seat order, action/card schemas, ledger, card perception, robotics,
hardware protocols, archived DeskMate content, unrelated plans and private
media remain read-only. Existing single-seat gesture and English speech pilots
must remain independently usable.

## External Dependencies

- Existing offline MediaPipe Gesture Recognizer development asset.
- OpenCV camera index 0 and the Laptop display.
- Final four-seat ROI geometry, target camera, table dimensions and cross-seat
  evidence remain external Gate 0/2A dependencies.

## Validation And Physical Motion

Validate complete non-overlapping seat ROI coverage, multi-hand scalar output,
centroid-to-seat routing, out-of-layout and multiple-hand rejection, focus-seat
switch reset, cross-seat candidate isolation, schema compatibility, practical
full tests and a bounded camera/UI smoke test. No frames are saved. No robot
connection or physical motion is authorized.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Validation

- Four non-overlapping Laptop quadrant ROIs parse for Seat A/B/C/D and retain
  the explicit non-target-geometry status.
- Multi-hand MediaPipe output, centroid routing, center-gap rejection,
  non-focused-seat isolation, same-seat multiple-hand rejection and temporal
  focus reset are covered; scoped action tests: 29 passed.
- Practical full suite: 122 passed; configs parse and Python sources compile.
- DirectShow camera 0 ran 8.51 seconds at 1280x720 with 113 frames and zero
  missing reads. Effective rate was 13.29 FPS; four-hand inference P95 was
  24.99 ms.
- No hand was visible during the automated smoke, so the live A/B/C/D matrix
  remains open. Zero frames were saved; no biometric identity, robot
  connection or physical motion was used.
