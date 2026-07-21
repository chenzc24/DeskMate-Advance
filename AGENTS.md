# Poker Dealer Repository Rules

This is the independent Poker Dealer project root. The completed Baseline and
archived DeskMate are separate products: do not import their code, data,
weights, configs, package interfaces, or runtime assumptions.

## Authority And Scope

1. `docs/plans/POKER_DEALER_MASTER_PLAN.md` is the active product and delivery
   authority until a human explicitly supersedes it.
2. `docs/stages/STAGE_*.md` define the deliverables and gates for each stage;
   they may not silently expand Core scope.
3. `configs/game/core_v1.json` freezes the four-player position, state-owned
   action attention, perception evidence, table-slot lifecycle, ledger, pot,
   showdown and safety semantics. Its Fixed-Limit structure is still a candidate and its
   numeric values are defaults exactly as S0-07 states. Hardware/geometry facts
   remain pending in the decision register and Gate audit; partial software
   freeze must not be misrepresented as product or physical validation.
4. DeskMate history is read-only at the branch/commit named in
   `archive/deskmate/README.md`; it is never an active dependency.

Do not modify the containing `DeepLearning` workspace unless the user names it.
Preserve unrelated dirty and ignored files. Never commit, push, create a branch,
publish a release, or open a PR unless the user asks.

## Minimal Target Workflow

For code, data, model, protocol, hardware-integration, or multi-file changes,
create or update one `plan/<git-user>/<YYYY-MM-DD-target>/plan.md`. Record the
outcome/owned paths, dirty read-only paths, external dependencies, validation,
physical-motion status and commit intent. Avoid overlapping policy documents.

## Core Product Boundary

Core v1 is a fixed four-seat/four-player Texas Hold'em table with manual shuffle
and deck loading, automatic single-card dealing, manual card return,
state-controlled fixed-seat player-behaviour evidence, face-up card recognition,
deterministic rules/evaluation, multi-pot digital accounting, and an
authoritative digital chip ledger.
Fixed-Limit is the current candidate but remains a human product decision;
stakes, stack, cap and timeout values are defaults. Physical chip handling,
free-space card collection, 5–6 player expansion, unconstrained table vision,
autonomous shuffling and treating gesture/voice as game logic are Plus work.

Preserve this dependency direction:

`camera/action input -> owned observations -> deterministic game engine ->`
`semantic dealer command -> safety controller/robot adapter -> acknowledgement`

- Game rules and hand ranking are deterministic code, never model predictions.
- Models emit observations with confidence/evidence, never game transitions,
  motor speeds, servo angles, GPIO writes, or serial bytes.
- The game state machine is the sole `acting_seat` authority. Only its focused
  fixed seat ROI can yield an action candidate; focus changes only after a
  legal action and ledger update commit atomically with a new state version.
- Player-action model output stays `PlayerActionObservation` evidence until
  temporal/calibration confirmation and game legality checks succeed. Stale,
  non-current-seat, ambiguous, occluded or conflicting evidence changes
  neither focus nor ledger. Under S0-21 only, consented session face identity
  may verify a registered player at the already-selected seat; it never selects
  focus, transfers game state, or mutates the ledger. Embeddings are memory-only.
- The digital ledger is the only Core balance authority. Physical chips are
  not recognized and operator adjustments/rebuys require append-only audited
  events with operator identity and reason.
- The game engine requests semantic actions such as `rotate_to` and
  `dispense_one`; the robotics controller owns kinematics and actuation.
- A successful command acknowledgement is required before advancing a physical
  deal step. Timeout, jam, duplicate card, unknown vision, illegal action,
  disconnect, or state mismatch pauses the hand for human recovery.
- Unknown/low-confidence evidence is not absence and is never guessed into a
  card identity. Duplicate identities in one hand are a hard error.
- Keep camera, game, perception and mechanism simulators independently usable.
- Keep queues/windows bounded and use monotonic timestamps for durations.

## Artifact, Data And Model Rules

Git tracks source, configs, schemas, compact evaluation, manifests and model
metadata. Raw/private images and video stay in ignored `data/raw/`; derived
views in ignored `data/work/`; runs in ignored `runs/`; weights in ignored
`models/assets/`. Never commit identity-bearing media, credentials, licenses,
consent records or signed URLs. Never commit or persist face embeddings; clear
the in-memory enrollment gallery at session end. Face enrollment requires
explicit participant consent.

Identify a dataset snapshot by manifest SHA-256. Keep original bytes immutable;
split card data by physical deck design and complete capture session, and split
behaviour data by participant and complete session, before generating crops,
sequences or augmentations. Keep adjacent windows, near duplicates and all
views of one source item in one split. Card negatives include glare, blur,
occlusion, empty slots, hands, shadows, wrong orientation and unknown deck
backs. Behaviour negatives include ordinary hand motion, card/chip handling,
cancelled actions, simultaneous neighbour motion, occlusion and long no-action
recordings.

No runtime model is admitted until `models/manifest.yaml` records immutable
version, source/license, exact weight hash, dataset/view hashes, framework,
input/output contract, metrics and offline export. Model states are
`development`, `candidate`, `release`, or `fallback`; only one release and one
fallback per model ID may be active. Runtime downloads are prohibited.

## Validation And Safety

Always run `git diff --check` and scoped `git status --short --branch`.
Documentation/config work must parse machine-readable files and verify links,
IDs, labels, units, versions, gates and archive references. Python work runs
targeted tests and the practical full suite.

Card perception reports per-rank/per-suit precision, recall, F1, confusion,
unknown/rejection behaviour, duplicate-card detection, per-slot stability,
latency and target-camera results across held-out decks/sessions. Integration
uses simulator and recorded replay before hardware.

Player-action perception reports per-action precision, recall, F1, confusion,
calibration/rejection, false accepted actions per hour and hand, cross-seat
leakage, cancellation behaviour and P95 confirmation latency on held-out
participants/sessions. Accuracy alone never admits an action model.

Physical motion requires an operator, clear area, guards, low force/speed,
homing, card-present and jam sensing, current/timeout limits, watchdog, manual
stop, and recovery instructions. Protocol/mock tests come first. No target
plan may authorize unattended physical motion.
