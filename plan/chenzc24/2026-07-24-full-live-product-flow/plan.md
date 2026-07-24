# Full Live Product Flow

## Outcome

Implement the confirmed Core v1 runtime decisions:

- a successful, sensor-valid `dispense_one` acknowledgement is the face-down
  evidence for a hole-card slot; no operator key/button confirmation is needed;
- robot camera profiles use state-directed view cycling with full-frame YOLO
  card detection, while the 13 logical slots remain authoritative game-state
  identities;
- runtime model assets are selectively prepared for Git LFS and model admission
  status is changed only where the repository's evidence gate is satisfied;
- the mobile web console and English announcements cover registration, dealing,
  betting, board progression, showdown, recovery, hand boundary and session end.

## Owned paths

- `src/poker_dealer/runtime/`
- `src/poker_dealer/perception/cards/geometry.py`
- `scripts/runtime/run_hand.py`
- `configs/runtime/`
- `configs/perception/card_slots_*.json`
- `configs/game/core_v1.json`
- `configs/runtime/announcements_en.json`
- `models/manifest.yaml`
- `.gitignore`
- `.gitattributes`
- scoped runtime, game and perception tests
- architecture, plan and stage documentation directly describing these contracts

## Dirty read-only / preserve

The worktree already contains uncommitted AudioRelay, face identity, announcer
and mobile-console work dated 2026-07-23. Those changes are inputs to this target
and must be extended in place without reverting or replacing unrelated behavior.
No raw media, audio, embeddings or run logs may be added to Git.

## External dependencies

- Robotics still owns physical kinematics, actuation and the concrete ACK
  transport. This target consumes only the frozen semantic command/ACK contract.
- Target camera and physical geometry validation remain pending.
- Card-recognition v2 remains under model debugging/evaluation and is not promoted
  to release by this target.

## Validation

- JSON/YAML-machine-readable parsing and model asset hash checks
- targeted game/runtime/perception tests
- practical full Python suite
- runtime CLI preflight and simulator/log checker
- `git diff --check`
- scoped `git status --short --branch`

## Physical-motion status

No unattended or real physical motion is authorized. All runtime tests use the
simulated dealer; the real adapter remains fail-closed until Robotics supplies
and validates the hardware transport.

## Commit intent

Do not commit, push, create a branch, PR or release unless the user explicitly
requests it after validation.

## Result

Completed on 2026-07-24:

- hole-card delivery now advances directly from a successful sensor-valid
  dispense ACK to `present_face_down`; the old operator confirmation switch and
  CLI flag were removed;
- robot-camera profiles now use state-directed full-frame YOLO binding, while
  the 13 card-slot IDs remain logical state/ledger identities; fixed pixel ROIs
  remain available only for the Laptop fixed-table fixture;
- the mobile web console now mirrors registration, the complete authoritative
  hand state, recovery, hand boundaries and session end, and exposes only
  confirmation/recovery controls rather than game-action buttons;
- English announcements cover the same end-to-end flow;
- selected runtime model assets are no longer blanket-ignored and have Git LFS
  attributes. Card-recognition v2 remains `candidate`; the other runnable
  perception components remain `development` until the mandatory held-out
  admission metrics exist, so this target does not mislabel runtime usability
  as a formal model release.

Validation completed:

- all changed JSON-compatible configuration and manifest files parse;
- Laptop and robot-camera `live-preflight` checks pass, including exact asset
  hashes; robot-camera reports full-frame binding with zero pixel ROIs and
  thirteen logical slots;
- the complete Python suite passes: `359 passed`;
- the mobile UI was exercised in registration, hand and recovery states at a
  phone-sized landscape viewport with no page overflow;
- `node --check` passes for the browser application;
- `git diff --check` passes; the final scoped worktree status remains dirty by
  design because no commit or push was requested.

Physical-motion status is unchanged: only the simulated dealer was exercised,
and the real robotics adapter remains fail-closed pending the hardware
transport and physical safety validation.
