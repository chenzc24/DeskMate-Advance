# Four-Player Live Acceptance Preparation

## Outcome And Owned Paths

Prepare the complete pre-validation Part A package without claiming live
results: repeatable four-participant acceptance, environment/model preflight,
pseudonymous consent/session and operator records, batch aggregation, immutable
action-data manifests and participant-safe splits, deterministic safety replay,
landmark derivation, an optional compact-TCN training/export implementation,
development model metadata, scoped tests and operator documentation.

## Dirty Read-Only Paths

The game rules, ledger, Part B card pipeline, robot transport, physical motion,
trained model weights and existing perception thresholds remain read-only. The
merged Part A runtime may receive only backward-compatible logging/CLI changes;
the shared model manifest may receive development-only drafts with blockers.

## External Dependencies

Tomorrow's execution needs four consenting participants, Laptop camera 0 and
the selected English microphone. Today's validation uses synthetic JSONL and
the existing simulated coordinator only. No runtime downloads are allowed.

## Validation And Physical Motion

Parse all new JSON, run analyzer unit tests, existing Part A/runtime tests and
the practical full suite. Run a no-device synthetic acceptance report. Do not
connect a robot or authorize physical motion. Do not persist frames, audio,
face images or embeddings; runtime evidence is JSONL under ignored `runs/`.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Outcome

- Added a machine-readable nine-case protocol covering incomplete enrollment,
  exact D-A-B-C progression, wrong-player rejection, multiple faces, identity
  loss, speech confirmation, multimodal conflict, long no-action activity and
  wrong-player recovery.
- Added a runner that creates a unique ignored evidence directory, invokes the
  four-player runtime with frozen case/session/hand values and automatically
  invokes the offline analyzer.
- All runtime JSON events now carry one session ID, hand ID, acceptance case
  and monotonic log timestamp. Optional JSONL output uses exclusive creation
  and never overwrites prior evidence.
- Added a strict analyzer for required, forbidden and ordered events, minimum
  timing gaps, exact state transitions, state-version continuity, persistence
  flags, robot-disconnected evidence and confirmation-latency summaries.
- Synthetic/scoped tests passed (26); the full suite passed (156). JSON,
  compile, artifact and diff checks passed.
- A bounded camera/microphone smoke produced 5/5 readable frames, zero dropped
  audio blocks and a valid tagged JSONL. Its intentionally incomplete FPA-07
  evidence was correctly rejected for missing the identity gate and 120-second
  duration, so no live four-player result is claimed.
- Added a read-only preflight that verified Python/dependency versions, all
  gesture/speech/face/hand-landmarker hashes, the protocol, model manifest,
  disk, camera 0 and a real 16 kHz/mono microphone block. The full device
  preflight passed with no overflow and no saved media.
- Added ignored pseudonymous four-seat session/consent records, per-attempt
  operator observations for handedness/environment/modality/failure category,
  and a batch aggregator that retains retries and requires machine plus manual
  evidence for all FPA-00 through FPA-08.
- Added an action source-manifest schema, immutable hash/file validation and
  deterministic participant-level train/validation/test assignment with
  session and duplicate-byte leakage checks.
- Added a 10,000-event deterministic rejection replay. All 10,010 negative,
  stale, duplicate, illegal and low-confidence observations were rejected;
  state/ledger stayed unchanged and a subsequent legal call recovered to
  version 1 / seat A.
- Added the hashed MediaPipe Hand Landmarker derivation path, ignored normalized
  landmark views, 32-frame masked windows, compact TCN config/model, optional
  PyTorch training, weighting/early stopping, held-out classification metrics,
  development checkpoint, TorchScript export and reference/export check. A
  generated blank-video source completed the manifest -> split -> landmark ->
  view hash -> TCN dry-run path. No PyTorch install, training or weight asset was
  performed.
- Added a development-only untrained TCN manifest entry, supporting asset
  metadata and a handoff template/checklist that keeps all missing live,
  dataset, metric, target-camera and physical-integration blockers explicit.
- Final full suite passed (174). All 29 machine JSON files parsed, all 16
  required artifacts existed, compile and diff checks passed.
- Physical motion remained disabled. Persisted camera frames, audio and face
  embeddings: zero. No commit or push was performed.
