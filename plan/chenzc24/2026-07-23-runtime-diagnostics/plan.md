# Runtime Diagnostics Bundle

## Outcome

Add an opt-in diagnostics layer to the formal `run_hand.py` entry so a field
run can be debugged from startup through shutdown without changing game,
perception, Dealer or safety semantics. Each run produces one isolated,
machine-readable bundle with a manifest, lifecycle events, bounded step
metrics, captured stdout/stderr, authoritative hand/session logs and a final
summary with the first recorded failure.

## Owned Paths

- `src/poker_dealer/runtime/diagnostics.py`
- `src/poker_dealer/runtime/hand_loop.py` for an optional observation-only sink
- `src/poker_dealer/runtime/__init__.py` exports
- `scripts/runtime/run_hand.py` diagnostics CLI and lifecycle integration
- focused Runtime tests and Runtime documentation
- `configs/contracts/` only if a diagnostics artifact schema is required

## Dirty And Read-Only Paths

The worktree was clean at target start. Preserve raw/private data, ignored
historic runs, model assets, perception thresholds, DeskMate archive and all
Robotics firmware/mechanics. Existing audit logs remain authoritative and are
not replaced by diagnostics. The unrelated untracked
`plan/chenzc24/2026-07-23-pretrained-poker-labeling/` appeared while this target
was in progress and remains read-only and untouched.

## External Dependencies

No new required dependency. Process/platform data uses the Python standard
library. Diagnostics remain usable when optional performance packages are not
installed.

## Implementation Stages

1. Implement an exclusive, ignored diagnostics bundle with redaction, bounded
   hash-chained JSONL, immediate flush, stdio teeing and deterministic summary.
2. Add an optional `HandRuntimeLoop` sink for correlated step durations and
   exception context without granting diagnostics any runtime authority.
3. Add `--diagnostics` and `--diagnostics-dir` to the formal CLI; place default
   hand/session logs inside the bundle and record preflight/results/artifacts.
4. Test normal completion, event limits, redaction, caught failure, artifact
   hashes and unchanged non-diagnostics behavior.
5. Document field usage, bundle interpretation, privacy and known Gate limits.

## Validation

- Focused diagnostics, hand-loop and CLI tests.
- Practical full `pytest` suite and Python compileall.
- Parse all new JSON artifacts and validate privacy/bounds behavior.
- `git diff --check` and scoped `git status --short --branch`.

## Physical Motion Status

No physical motion is authorized or performed. `robot_hardware` remains
disabled. Validation uses Replay, mocks and the simulated Dealer only.

## Commit Intent

Do not commit, push, create a branch, release or PR unless explicitly requested.

## Completion Record

Implemented the opt-in Diagnostics bundle on the formal Runtime entry. Default
Diagnostics runs now keep authoritative hand/session logs, redacted portable
configuration snapshots, bounded hash-chained lifecycle/metric JSONL, bounded
stdio, correlated Runtime/port latency, process snapshots and automatic
first-failure context in one exclusive ignored directory. Added an independent
portable Checker that revalidates Diagnostics chains, artifact/config hashes,
privacy declarations and the original hand/session logs.

Validation on 2026-07-23 used Replay and no devices: one complete 111-step hand
produced a portable bundle and passed the Diagnostics, hand and session
Checkers. Runtime-focused tests passed 97 tests; the practical full suite passed
314 tests. Python compileall, all config JSON parsing, repository-local Markdown
links and `git diff --check` passed. No camera, microphone or physical motion was
opened. Target geometry, live participants and real Dealer gates remain open.
