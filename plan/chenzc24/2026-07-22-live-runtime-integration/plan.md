# Live Perception To HandRuntime Integration

## Outcome

- Extract model-neutral live/replay ports from the existing Part A and card
  pilots without changing their thresholds or model semantics.
- Drive one complete hand through `HandRuntime` with recorded observations,
  simulated dealer acknowledgements, append-only logs and an independent log
  checker.
- Add a single live mode to `scripts/runtime/run_hand.py` for both `laptop` and
  `robot_camera`; profiles may replace devices and calibration, never rules or
  state-machine code.
- Keep `robot_hardware` fail closed. This target specifies and tests the
  software handoff only; Robotics owns kinematics, firmware, sensors and safe
  physical execution.

## Owned Paths

- `src/poker_dealer/runtime/ports.py`
- `src/poker_dealer/runtime/event_log.py`
- `src/poker_dealer/runtime/hand_loop.py`
- `src/poker_dealer/runtime/replay.py`
- `src/poker_dealer/runtime/live_perception.py`
- `scripts/runtime/run_hand.py`
- `scripts/runtime/check_hand_log.py`
- `configs/runtime/` and runtime-profile contract when required
- focused runtime/replay/CLI tests
- scoped architecture/evaluation documentation

## Dirty And Read-Only Coordination

The uncommitted unified HandRuntime, dual-profile work, earlier perception
pilots and their tests are inputs to this target and must be preserved.
Existing perception thresholds, model assets/manifests, raw/private data,
DeskMate/Baseline archives and all Robotics firmware/mechanical artifacts are
read-only. Refactoring may add adapters around existing model classes but must
not silently change calibration or recognition semantics.

## Software Stages And Gates

1. Freeze ports and runtime contexts. Models only emit domain observations.
2. Prove a complete recorded hand, including all Part A/Part B transitions,
   showdown and ledger settlement.
3. Write append-only evidence/transition logs and verify them with a checker
   that does not trust a claimed winner or balance.
4. Add a bounded unified event loop with a single frame source and simulated
   dealer.
5. Compose existing face, gesture, English speech and card adapters for Laptop
   live operation; no direct button betting.
6. Run the identical composition under the robot-camera profile, changing only
   camera/calibration/device configuration.
7. Prove resource isolation and keep hardware startup rejected until Robotics
   releases a compatible real adapter and safety evidence.

## Validation

- Port authority and single-frame-source unit tests.
- Golden complete-hand replay plus stale, ambiguous, cross-seat, duplicate
  card, missing ACK and log-tamper cases.
- CLI config/replay/live preflight tests for Laptop and robot-camera profiles.
- No-device fake-live test from frozen roster to `SETTLED`.
- Practical full suite, `compileall`, schema/JSON checks, `git diff --check`
  and scoped status.

## Physical Motion Status

No physical motion is authorized. Runnable software paths use
`SimulatedDealerAdapter`. `robot_hardware` must reject startup before opening a
camera or transmitting any command. H1-H3 require Robotics-owned deliverables,
an operator and separate Stage 3/4 approval and therefore cannot be represented
as completed by this target.

## Commit Intent

Do not commit, push, create a branch, publish a release or open a PR unless the
user explicitly asks.

## Implementation Result

Completed in software on 2026-07-22:

- model-neutral Frame/Registration/Identity/Action/Card/Control/Event ports;
- one bounded `HandRuntimeLoop` shared by Replay, Laptop and robot-camera
  profiles, with state-owned routing and one shared camera frame per step;
- complete no-burn hand Replay, simulated command acknowledgements,
  append-only hash-chain log and independent deterministic checker;
- exact evidence Replay with context-override rejection;
- unified development Live composition for face, pose, multi-hand gesture,
  English speech, in-memory session speaker verification and face-up card
  recognition;
- state-selected player speaker/gesture attribution, speech-only confirmation,
  conflict/unknown rejection, and no direct button betting;
- runtime asset preflight for Laptop and robot-camera, separate resource locks,
  and a fail-closed `robot_hardware` profile;
- a frozen Robotics handoff contract and protocol/mock tests only.

Validation evidence:

- scripted Replay: `SETTLED`, 111 steps, state version 56, 149 engine events,
  53 evidence records, checker passed;
- exact recorded Replay under `robot_camera`: the same final values and checker
  result;
- both Live preflights verified Face/Gesture/Pose/Card/Vosk English/Vosk
  speaker assets; downloads, frame saving and audio saving are disabled;
- 284 repository tests passed; all runtime profiles passed their JSON Schema;
  all config JSON, relative Markdown links, compileall and diff checks passed.

External acceptance remains explicitly open rather than being simulated:

- a four-person Laptop run from registration to `SETTLED` has not been
  operator-witnessed in this target;
- target-camera geometry, every card slot, target-view metrics and long-run
  reconnect behaviour have not been physically calibrated or accepted;
- hole-card occupancy/back orientation still uses an explicit development
  operator fallback and is not Gate 2B evidence;
- Robotics H1-H3 remain Robotics-owned; no real adapter, mechanism command or
  physical motion was enabled or attempted.

No commit, branch, push, release or PR was created.
