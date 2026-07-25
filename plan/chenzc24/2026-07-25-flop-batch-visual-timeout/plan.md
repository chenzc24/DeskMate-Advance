# Flop Batch Visual Confirmation And Live Timeout

## Outcome

Change the no-burn Flop path from three interleaved
`dispense -> visual-confirm` steps to one board rotation, three consecutive
sensor-acknowledged single-card dispenses, and one three-card full-frame visual
confirmation window. Keep Turn and River as one-card delivery/confirmation
steps.

Make live runtime timeouts explicit in runtime profiles. Use 60 seconds for
face-up card visual confirmation while retaining the short dealer-command
timeout.

## Owned Paths

- Part B coordinator, hand loop and live card perception.
- Runtime profile timeout parsing/schema and live session propagation.
- Runtime profiles and Core v1 dealing contract.
- Scoped Part B, card-perception and runtime-profile tests.
- Directly affected game-rule/interface documentation.

## Dirty Read-Only Paths

- Preserve the existing uncommitted phone-front A/D profile and its plan except
  for adding the new timeout configuration.
- Preserve all ignored captures, diagnostics, models and unrelated work.

## External Dependencies

- Existing 52-class YOLO card asset.
- Robotics continues to acknowledge three `dispense_one` commands; no new
  hardware command or transport format is assumed.

## Validation

- Flop simulator tests prove one rotation, three consecutive dispense ACKs and
  no visual gate until all three succeed.
- Mid-batch restart requests the remaining dispenses instead of guessing that
  the visual window is ready.
- Shared-frame tests bind three unique cards left-to-right and reject an
  incomplete batch.
- Runtime profiles parse and expose `card_visual_ms=60000`; the live
  application propagates the profile values through `SessionRuntime` into
  `HandRuntime`.
- Targeted runtime/card/profile/UI tests: 64 passed, then 11 focused regression
  tests passed after updating diagnostics expectations.
- Practical full Python suite: 427 passed.
- All config JSON parsed; mobile JavaScript syntax check passed; runtime
  preflight reported ready with the shared phone camera and simulated dealer.
- `git diff --check` passed with line-ending notices only; scoped status was
  inspected and no commit or push was made.

## Physical-Motion Status

No physical motion is authorized. Protocol and simulator behavior only.

## Commit Intent

Do not commit, push, create a branch, release or PR.
