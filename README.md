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

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\game\demo_stage1.py
.\.venv\Scripts\python.exe scripts\game\replay_stage1.py
.\.venv\Scripts\python.exe scripts\game\random_stage1.py --hands 10000 --seed 20260721
```

Stage 1's software oracle and tests are complete. Fixed-Limit remains a tested
candidate until the S0-07 product decision is signed; passing Stage 1 does not
silently freeze that product choice.

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

No physical mechanism may move from software until Stage 3 safety and hardware
gates are signed off by an operator and the robotics owner.
