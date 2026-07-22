# Actor-bound player action target plan

## Outcome

Replace the Stage 2A assumption that any visible hand or audible command belongs
to the state-selected player with an explicit, expiring session `ActorBinding`.
Gesture evidence must be associated with the verified player's tracked body, and
speech controls must support pending command confirmation/cancellation with a
session-only speaker-verification boundary. Models remain evidence producers;
the deterministic game engine remains the only game-transition authority.

## Owned paths

- `src/poker_dealer/perception/attribution/`
- `src/poker_dealer/perception/actions/mediapipe_adapter.py`
- `src/poker_dealer/perception/actions/speech.py`
- `src/poker_dealer/perception/actions/window.py`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `scripts/perception/live_sequential_part_a.py`
- `configs/perception/actor_binding_session.json`
- `configs/perception/speaker_verification_session.json`
- `docs/evaluation/stage2a-actor-binding-and-speaker-verification.md`
- `docs/evaluation/stage2a-single-player-laptop-pilot.md`
- Scoped tests under `tests/perception/attribution/`, `tests/perception/actions/`
  and `tests/runtime/`
- Model metadata only when an existing local supporting asset is activated

## Dirty read-only paths

The worktree contains pre-existing user changes and untracked Stage 2A
preparation artifacts. In particular, camera adapter/diagnostics, master/stage
plans, README, training/evaluation infrastructure and their tests are treated
as read-only unless an unavoidable integration seam is identified. No unrelated
change will be reverted, formatted or staged.

## External dependencies and open evidence

- The local MediaPipe pose assets can support a development body/hand
  association pilot, but target-camera admission remains open.
- The official Apache-2.0 Vosk `vosk-model-spk-0.4` asset is now hash-pinned and
  locally loadable as a `development` model. It is not admitted as candidate or
  release: thresholds still require held-out participant/session calibration,
  noisy-room and robot-noise evaluation, and replay-attack analysis.
- Robotics owns real `rotate_to` completion. This target may expose the ACK and
  visual-settle boundary but will not fabricate physical validation.

## Validation

- Unit tests for binding lifecycle, stale state, lease expiry, wrong player,
  ambiguous/multiple hands, hand-to-wrist association, speech confirm/cancel,
  wrong speaker and cleared in-memory galleries.
- Existing identity/action/runtime/game tests.
- Practical full `pytest` suite, JSON/manifest parsing, `git diff --check`, and
  scoped `git status --short --branch`.
- Recorded/synthetic replay before another live robot-camera run.

## Physical motion status

No unattended physical motion is authorized. All tests in this target are
software, simulator, recorded replay or operator-driven camera UI tests.

## Commit intent

Do not commit, push, create a branch, publish or open a PR unless the user asks.
