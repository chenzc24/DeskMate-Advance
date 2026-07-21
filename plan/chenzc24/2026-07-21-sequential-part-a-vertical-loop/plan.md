# Sequential Part A Vertical Loop

## Outcome And Owned Paths

Build the Laptop-only Stage 2A vertical loop from a deterministic acting seat
through simulated rotation acknowledgement, consented session identity
verification, gesture/English-speech fusion, deterministic game legality and
the next acting seat. Own `src/poker_dealer/runtime/sequential_part_a.py`, the
bounded multimodal decision window, the combined live pilot, scoped tests and
the corresponding Stage 2A documentation.

## Dirty Read-Only Paths

Card perception, physical dealing, ledger implementation, betting semantics,
robot transports, model weights and unrelated user changes remain read-only.
Shared action/identity/game contracts may only receive backward-compatible
integration changes with tests.

## External Dependencies

Use the already pinned local YuNet, SFace, MediaPipe gesture and Vosk English
assets. Camera and microphone are local inputs. Runtime downloads, biometric
storage and audio/frame recording remain prohibited.

## Validation And Physical Motion

Test phase gates, ACK mismatch/failure, identity unknown/mismatch, multimodal
agreement/conflict/single-source timeout, illegal/stale action rejection,
accepted action focus advance and round completion. Run scoped and full tests,
then bounded camera/microphone-capable smoke where practical. `rotate_to` is
executed only by `SimulatedDealer`; no robot connection or physical motion is
authorized.

## Commit Intent

The user explicitly requested publication on 2026-07-21. Commit the owned Part
A vertical-loop paths on an `agent/` branch, push that branch and open a draft
PR; do not include ignored runtime output, model assets or identity-bearing
media.

## Completed Outcome

- Added deterministic gates for simulated matching `rotate_to` ACK, current
  session identity, the state-owned action window, game legality and next-seat
  rotation.
- Added a bounded multimodal decision window: matching gesture/speech
  candidates agree immediately and conflicts reject as ambiguous. In the
  default four-player mode, gesture-only evidence waits 500 ms, while
  speech-only evidence additionally requires a matching gesture or explicit
  operator `C` confirmation. Speech-only promotion remains available only in
  the explicitly selected two-player development fixture.
- Added the combined Laptop UI with consented enrollment, identity-gated audio
  and gesture processing, Vosk window resets and explicit Part A boundary at
  `dealing_board/settled`.
- Corrected mode isolation after acceptance. `four_player_core` is the default
  and cannot start unless A/B/C/D are all enrolled; an incomplete gallery is a
  hard start error. `two_player_pilot` must be selected explicitly, accepts
  exactly two adjacent clockwise seats and terminates after those two
  identity-action cycles instead of falling through to an unenrolled seat.
- Added an explicit `expected_seat_unenrolled` identity state, structured
  enrollment/rotation/identity/speech/fusion/game-transition logs and a
  continuous identity guard. The action window closes without changing game
  or ledger state when the verified face changes, multiple faces appear, or
  face evidence remains unavailable beyond the configurable grace period.
- Added a deterministic four-player preflop integration vector that advances
  D -> A -> B -> C and reaches `dealing_board` at state version 4. Scoped
  correction tests passed (26), full suite passed (148), and JSON parse,
  compile and diff checks passed. The final camera+microphone 10-frame smoke
  had zero missing frames and zero dropped audio blocks.
- Four-person live acceptance remains pending until four participants are
  available. Face-to-player matching is a consented, memory-only development
  pilot and is not admitted as the Core identity authority; no liveness or
  hand-to-face association is claimed.
- Physical motion remains `simulated only`; no robot was connected. Frames,
  audio and embeddings persisted: zero. Publication is limited to the owned
  source, contracts, tests, documentation and this target plan.
