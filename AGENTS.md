# DeskMate Advance Repository Rules

This directory is the independent DeskMate Advance project root. Keep the
workflow small: one source of truth per artifact type, one bounded target at a
time, and no copied datasets or checkpoints without a clear reason.

The completed Baseline repository is a separate task. Do not import its code,
data, weights, labels, configs, package interfaces, or architectural
assumptions into this repository.

## Authority And Scope

1. `DeskMate_Advance_Proposal (1).pptx` is the read-only product requirement
   and overrides derived project documents.
2. `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md` governs function scope,
   pretrained-model selection, learned extensions, event contracts, gates,
   integration, and delivery until a human explicitly supersedes it.
3. A dated target plan may bound implementation work but may not weaken the
   authorities or safety requirements above.

Do not modify the containing `DeepLearning` workspace unless the user
explicitly names it. Preserve unrelated dirty files. Never commit, push,
create a branch, publish a release, or open a PR unless the user asks.

## Minimal Target Workflow

For code, data, model, protocol, hardware-integration, or multi-file changes,
create or update one plan at:

```text
plan/<git-user>/<YYYY-MM-DD-target>/plan.md
```

State only:

- outcome and owned paths;
- dirty paths left read-only;
- external data, model, service, and hardware dependencies;
- validation and whether physical robot motion is involved;
- commit intent.

Do not create extra policy documents, experience notes, or overlapping plans
unless a human requests them. Record factual completed validation once in
`plan/log.md` when that path exists.

## Artifact Ownership

Git is the control plane; large binaries and privacy-sensitive media stay local
or in an approved artifact store.

| Artifact | Canonical location | Git policy | Lifetime |
| --- | --- | --- | --- |
| source/review/split manifests | `data/manifests/` | tracked | permanent |
| raw video, audio, images, and landmarks containing identity metadata | `data/raw/` | ignored | keep one canonical copy |
| derived sequence datasets | `data/work/<dataset-id>/` | ignored | reproducible and disposable |
| training runs | `runs/<run-id>/` | ignored | temporary workspace |
| model metadata | `models/manifest.yaml` | tracked | permanent |
| model weights and exports | `models/` or artifact store | ignored except metadata | retain only admitted versions |
| detailed runtime evidence | `artifacts/` | ignored | retain only when needed |
| compact evaluation summaries | `docs/evaluation/` | tracked | permanent |

New pipelines must not create another full copy merely to rename, verify
determinism, or hand data to a run. Generate reproducible views, compare their
manifests and hashes, then remove disposable copies.

## Dataset Rules

- Identify a dataset snapshot by its manifest SHA-256, not its directory name.
- Keep original bytes immutable and store one manifest row per source item with
  consent/license status, participant, session, device, label, and content
  hash where applicable.
- Split by participant and complete recording session before generating
  windows, landmark sequences, crops, or augmentations.
- Keep duplicate, near-duplicate, adjacent-window, and same-session relatives
  in one split. Frame-level random splitting is prohibited.
- The exact train/selection/test proportions belong in the active plan and
  resolved config; do not silently change them between runs.
- Include transition clips, missing-landmark cases, ordinary hand motion, and
  long no-event recordings as negative evidence.
- Derived views must carry parent sample ID, split, transform, extractor and
  config hashes. They are cache, not new independent data.
- Never delete a canonical source or frozen manifest until a verified second
  copy exists.
- Never commit private video/audio, face imagery, participant identifiers,
  consent records, credentials, or signed download URLs.

## Experiment And Model Rules

Use one immutable run ID:

```text
<task>-<dataset-id>-<model>-s<seed>-<YYYYMMDD-HHMM>
```

Every meaningful run records the Git commit and dirty state, resolved config,
dataset and derived-view manifest hashes, feature-extractor/version hash, base
weight hash when applicable, seed, environment, metrics, and produced
checkpoint/export hashes. A directory name alone is not provenance.

- `runs/` is scratch space, never the permanent release registry.
- Keep configs and metrics for useful comparisons. Keep resumable and selected
  checkpoints only while they remain active candidates.
- Failed-run weights are disposable after their metrics, config, and failure
  reason are recorded.
- Promote a model by adding an immutable version and SHA-256 to
  `models/manifest.yaml`.
- Release and fallback configs must resolve through the manifest, not an
  arbitrary mutable path under `runs/`.
- Model states are `development`, `candidate`, `release`, or `fallback`. Only
  one release and one fallback per model ID may be active.
- Data cleanup, checkpoint deletion, and artifact retirement require explicit
  human approval after a read-only inventory.

## Advance Architecture Invariants

- Keep the project boundary independent of the Baseline repository; no runtime
  or build dependency may point to `../project`.
- Preserve the logical flow:
  `sensor input -> pretrained feature extraction -> temporal model/logic ->
  UnifiedEvent -> controller adapter`.
- The primary learned extension is a compact temporal gesture model over
  normalized hand-landmark sequences. Additional learned models start only
  after the gates in `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md` are
  satisfied.
- Framework tensors, MediaPipe result objects, and device-specific handles must
  not cross the model/runtime boundary. Convert them to owned domain records.
- Preserve timestamps and missing-data masks. Do not infer duration from a
  fixed frame-rate assumption.
- Low-confidence, stale, conflicting, or incomplete evidence becomes
  `unknown` or no event; it is never treated as confirmed negative evidence.
- Frame predictions must pass smoothing, multi-frame confirmation, duration or
  entry/exit thresholds, and cooldown before producing a persistent event.
- A model emits semantic events and evidence, never motor speeds, servo angles,
  or direct Arduino commands.
- `suggested_action` is advisory. The controller owns prioritization, distance
  limits, obstacle handling, manual stop, watchdog, and final actuation.
- Keep a model replay path and a legal event simulator so model and controller
  can be tested independently.
- Keep queues and windows bounded. Long-running inference must not accumulate
  frames, audio, events, or logs without limit.

## Unified Event Rules

Every emitted event must conform to one versioned project schema and include,
at minimum:

- semantic event name;
- `model_level: "advanced"` for learned Advance outputs;
- calibrated confidence;
- timestamp-derived duration;
- compact supporting evidence;
- schema/model version and traceable inference context in logs.

The controller-facing event vocabulary is a project contract. Changing a
field, label, unit, default, or rejection rule requires an explicit migration
and consumer validation; do not make silent compatibility changes.

## Validation

Always run `git diff --check` and scoped `git status --short --branch`.

- Documentation/config: parse machine-readable files and check referenced
  paths, IDs, hashes, labels, units, versions, and gates.
- Python: run targeted tests and `python -m pytest -q tests` when practical.
- Model/data: record exact dataset, view, extractor, config, and checkpoint
  hashes; compare candidates on identical held-out participant/session groups.
- Classification: report per-class precision, recall, F1, confusion matrix,
  rejection/unknown behaviour, and calibration rather than accuracy alone.
- Continuous behaviour: use long negative recordings and report false trigger
  rate, detection latency, missing-data behaviour, and cooldown behaviour.
- Perception/runtime: use recorded target-camera/audio evidence and report
  validity, stale/miss rate, throughput, memory, and P95 latency; do not infer
  performance from code inspection.
- Integration: test recorded-model replay and simulated events before live
  hardware. Validate both sides of the event contract independently.
- Robot motion: protocol/mock tests first. Physical motion requires an
  operator, clear area, low speed, distance limits, collision protection,
  watchdog, and emergency/manual stop.

Downloaded data, weights, runs, videos, audio, logs, secrets, signed URLs, and
local environments must remain outside Git. The final demo must load every
required model and runtime asset offline.
