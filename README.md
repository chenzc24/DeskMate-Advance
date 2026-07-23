# Poker Dealer

This repository now targets a four-player robotic Texas Hold'em dealer. Core v1
automatically deals a manually shuffled/loaded deck, tracks legal game state,
uses state-controlled fixed-seat action windows to collect temporal player
behaviour evidence, reads face-up cards, evaluates showdown deterministically,
and keeps an authoritative digital chip ledger. Models never choose the acting
seat, validate game rules, or mutate balances. Physical chip recognition,
collection/payment and automatic card return are Plus features.

Button, small blind, big blind and UTG are distinct seats. Fixed-Limit is the
current betting candidate, but remains a product decision; numeric stakes and
timeouts are configuration defaults.

Start with [the master plan](docs/plans/POKER_DEALER_MASTER_PLAN.md) and
[Stage 0](docs/stages/STAGE_0_SCOPE_AND_CONTRACTS.md). The active implementation
namespace is `poker_dealer`; DeskMate is available only through the immutable
[archive pointer](archive/deskmate/README.md).

The current Part A/Part B ownership and complete simulated hand path are
documented in [the unified hand runtime](docs/architecture/unified-hand-runtime.md).
Laptop/robot-camera dependency selection and the fail-closed hardware boundary
are documented in [runtime profiles](docs/architecture/runtime-profiles.md).
The shared perception ports, complete replay loop, development Live UI and
remaining card/geometry gates are documented in
[live runtime integration](docs/architecture/live-runtime-integration.md).

The single profile entry can be checked without opening any device:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --check-config
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_camera --check-config
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_hardware --check-config
```

The first two profiles use a simulated dealer. The hardware profile is
intentionally unavailable until the protocol and safety gates are released.
Camera-only smoke checks use `--camera-smoke-frames N`; they do not run a hand
or authorize physical motion.
The exact software/Robotics boundary is frozen in
[the Robotics handoff contract](docs/contracts/ROBOTICS_HANDOFF.md).

The complete no-device hand replay and independent log checker are:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --mode replay --session-id replay-001 --hand-id hand-001 --log-jsonl runs/runtime/replay-001/hand-001.jsonl
.\.venv\Scripts\python.exe scripts\runtime\check_hand_log.py runs/runtime/replay-001/hand-001.jsonl
```

Use `--mode live-preflight` to hash-check all profile-selected perception assets
without opening devices. The unified development Live mode exists for both
non-physical profiles, but target geometry and the hole-card back/orientation
model are not validated; its explicit operator fallback cannot close Gate 2B.
When speech is enabled, unified Live registration also enrolls three
memory-only speaker samples per participant. Spoken actions are discarded
unless the speaker matches the state-selected player; a speech-only action
still requires the same speaker to say `confirm` or an operator to press `C`.
The gallery, raw audio and embeddings are never persisted.

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\game\demo_stage1.py
.\.venv\Scripts\python.exe scripts\game\replay_stage1.py
.\.venv\Scripts\python.exe scripts\game\random_stage1.py --hands 10000 --seed 20260721
```

Stage 1's software oracle and tests are complete. Fixed-Limit was confirmed for
Core v1 by the 2026-07-22 S0-07 product decision; passing Stage 1 still does not
close the separate model, camera, mechanism, protocol, or safety gates.

The development-only Stage 2A Laptop gesture pilot can be installed and run
without recording frames:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[action-pilot]"
.\.venv\Scripts\python.exe scripts\perception\smoke_action_model.py
.\.venv\Scripts\python.exe scripts\perception\live_action_pilot.py --index 0 --backend dshow --max-seconds 60
```

Its five canned-gesture mappings and thresholds are feasibility defaults, not
a frozen interaction grammar or an admitted action model.

The four-seat Laptop fixture detects up to four hands and attributes them to
fixed, non-biometric seat ROIs. Keys `1` through `4` select the simulated focus
seat; only that seat can emit a candidate:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_multiseat_action_pilot.py --backend dshow --max-seconds 600
```

Its quadrant layout is not target-table geometry and cannot close the
four-player camera Gate.

The optional, offline English closed-vocabulary speech pilot emits the same
model-neutral action evidence and does not record audio:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[speech-pilot]"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_speech_pilot.py --list-devices
.\.venv\Scripts\python.exe scripts\perception\live_speech_pilot.py --device 1 --max-seconds 60
```

The Laptop microphone listening window follows the state machine's focused
seat, but it does not prove speaker identity. Gesture/voice conflicts remain
ambiguous and neither adapter bypasses game legality checks.

The optional session face identity pilot detects a face, produces an in-memory
embedding and matches it against a consented gallery for the current session.
It only verifies the state machine's already-selected seat; it never selects
the acting seat. No frames or embeddings are saved, and the gallery is cleared
on exit:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_face_identity_pilot.py --index 0 --backend dshow --consent-confirmed --max-seconds 600
```

Use keys `1`–`4` to select the simulated focus seat, `E` to enroll its default
player ID, `X` to clear the session gallery, and `Q`/`Esc` to exit. The pilot
has no liveness protection and its thresholds are not release-calibrated.

The following sequential Part A command is a legacy isolated perception pilot,
not the product runtime. It integrates simulated rotation acknowledgement,
session identity, gesture/English speech fusion, game legality and automatic
advance to the next acting seat:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_sequential_part_a.py --index 0 --backend dshow --speech-device 1 --consent-confirmed --max-seconds 900
```

To use the robot's HTTP MJPEG camera instead of the laptop camera, replace the
local camera flags with:

```powershell
.\.venv\Scripts\python.exe scripts\perception\live_sequential_part_a.py --stream-url http://100.80.46.54:5000/video_feed --speech-device 1 --consent-confirmed --max-seconds 900
```

This remains a perception-only run: frames are not saved and `rotate_to`
acknowledgements are still simulated. The stream adapter retains only the
latest received frame so slow model inference cannot accumulate stale MJPEG
frames.

During setup, use `1`–`4` and `E` to enroll all four players, then `S` to start.
The default `four_player_core` mode refuses an incomplete roster. During the
hand the state machine owns focus; identity must remain current for new action
evidence to enter. Matching gesture/speech agrees immediately, conflict is
rejected, gesture alone waits 500 ms, and speech alone requires `C` UI
confirmation. The separate two-player fixture requires the explicit
`--player-mode two_player_pilot` option. Rotation is simulated and the pilot
stops at the Part A betting-round boundary rather than inventing card or
physical-dealing acknowledgements.

The only formal full-hand entry point is `scripts/runtime/run_hand.py`. A
development Laptop live run uses unified registration, Part A, Part B,
SessionRuntime, event logging and recovery contracts:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --mode live --button seat_a --max-hands 20 --consent-confirmed --development-operator-face-down --registration-timeout-seconds 900
```

Registration happens once. Between hands, `C` confirms that all cards have
been returned, `S` starts the next hand, and `X` ends the session; stacks and
Button position persist in a separately checked session log. The operator
face-down switch is explicitly non-Gate evidence. Laptop now has its own
unvalidated 13-slot development geometry; calibrate and evaluate the selected
camera before claiming a Live card-perception pass. See the
[Runtime review closure](docs/reviews/2026-07-22-runtime-review-closure.md).

Before a four-participant acceptance session, run the read-only preflight,
create one ignored pseudonymous session record, execute each `FPA-00` through
`FPA-08` under the same session group, add the operator observation beside each
attempt, and generate the batch report. The complete commands and the separate
data/TCN preparation path are in
[`docs/evaluation/stage2a-prevalidation-infrastructure.md`](docs/evaluation/stage2a-prevalidation-infrastructure.md).

No physical mechanism may move from software until Stage 3 safety and hardware
gates are signed off by an operator and the robotics owner.
