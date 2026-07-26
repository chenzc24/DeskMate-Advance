# Navigation Inter-Motion Delay

## Outcome

Add a non-blocking 2500 ms minimum delay between two consecutive successful
physical navigation movements. The delay starts from the successful navigation
ACK timestamp. During the delay the game loop, mobile UI, controls, watchdog and
emergency-stop path remain responsive.

The same delay is carried in every `NavigationCommand` so the Raspberry Pi
navigation adapter must also apply it between motor primitives inside one
compound semantic command such as `move_and_align_to_target`.

## Owned Paths

- `src/poker_dealer/domain/robot.py`
- `src/poker_dealer/runtime/hand_runtime.py`
- `src/poker_dealer/runtime/hand_loop.py`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `src/poker_dealer/runtime/sequential_part_b.py`
- `src/poker_dealer/robotics/navigation/timing.py`
- `src/poker_dealer/robotics/navigation/__init__.py`
- `src/poker_dealer/robotics/navigation/README.md`
- `README.md`
- `tests/domain/test_robot_interfaces.py`
- `tests/runtime/test_robot_navigation_and_chip_gates.py`
- `tests/runtime/test_sequential_part_b.py`
- `plan/wuqi-ii/2026-07-27-navigation-motion-gap/plan.md`

## Dirty Read-Only Paths

Preserve all unrelated card-model, dataset, endpoint, line-tracking and model
asset changes already present in the working tree. The uncommitted table-route
interface work from `2026-07-26-table-route-state-machine` is a required base
and is modified only where the new navigation timing contract must pass through.

## External Dependencies

- A real Raspberry Pi/MCU navigation adapter is not present in this repository.
- The receiving adapter must parse `inter_motion_delay_ms` and enforce it for
  lower-level motor primitives that are hidden inside one semantic command.
- Both hosts must use monotonic timestamps for durations; wall-clock time is not
  an admissible cooldown source.

## Validation

- Domain validation for a non-negative `inter_motion_delay_ms`.
- Unit test for the monotonic cooldown gate.
- Runtime test proving a second physical navigation command is not emitted
  before 2500 ms and is emitted at the boundary.
- Simulator/replay test proving non-physical adapters do not incur real delay.
- Run scoped navigation/runtime tests, `git diff --check`, and scoped status.

## Physical-Motion Status

No physical motion is authorized or executed by this target. Validation uses
the simulator and a test-only adapter that reports `physical_motion=True`.
Real hardware still requires the safety gates, operator and protocol validation
defined by repository policy.

## Commit Intent

The user explicitly requested on 2026-07-27 that this navigation/state-machine
integration be committed and pushed to `main`. Stage only this plan's owned
paths and the two linked state-machine integration plans; preserve unrelated
dirty paths.

## Result

- Implemented the command field and physical-only host timing gate.
- The synchronous runner polls the monotonic delay in bounded 50 ms slices, so
  a 2500 ms delay cannot exhaust the step budget through a busy loop.
- Targeted domain, timing, physical-gate and full-hand replay tests:
  `19 passed`.
- Practical full suite: `465 passed, 9 failed`. The nine failures are the same
  pre-existing workspace/environment failures outside this target: unavailable
  Vosk, an incompatible local ONNX/OpenCV asset, direct demo import setup, the
  edited network endpoint assertion and its dependent preflight result.
- Python compilation and scoped `git diff --check` passed. Ruff is not installed
  in the current environment.
- No physical motor command was sent.
- Added a Chinese integration README covering interface contracts, order,
  cooldown behavior, recovery rules and the Raspberry Pi adapter checklist.
- Explicitly froze the board wait gate: after Flop/Turn/River dispense ACKs the
  robot remains at the board target and navigation is rejected until all
  required visible board slots are confirmed.
- Board-wait, navigation-gate and complete replay regression tests:
  `18 passed`.
