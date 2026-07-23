# Non-Hardware Live Productization

## Outcome

Complete every product/runtime capability that can be validated with the
Laptop camera, recorded evidence and `SimulatedDealerAdapter`: one registration
per session, consecutive hands with persistent stacks and rotating Button,
table-clear/rebuy/end controls, operator-visible recovery choices, session
audit/checking, explicit Laptop 13-slot calibration and software-only
qualification. Do not infer the future robot transport, camera stream, motion,
ACK timing or firmware behavior.

## Owned Paths

- `src/poker_dealer/runtime/` session orchestration, controls and audit
- `src/poker_dealer/perception/cards/` non-hardware slot occupancy interface
- `scripts/runtime/` formal Laptop live/replay entry points
- Laptop-only runtime/perception configuration and schemas
- focused runtime/perception tests and integration documentation

## Read-Only / External Dependencies

- `robot_hardware`, real transport, firmware, motors, sensors and safety gates
- future robot-camera stream and target geometry
- raw/private participant media and persisted identity embeddings
- existing unrelated dirty files

## Stages

1. Audit the current single-hand CLI and session authority.
2. Add a reusable session controller for registration-once, hand loop, terminal
   settlement, table clearance, rebuy/end and recovery decisions.
3. Add append-only session JSONL plus a checker spanning all hand logs.
4. Expose the controller in formal Live and multi-hand Replay modes.
5. Separate Laptop card geometry from robot placeholders and add an owned
   occupancy/orientation source contract that can replace the development key.
6. Add multi-hand and fault-injection qualification tests.
7. Update operator documentation and run complete validation.

## Validation

- Unit tests for all session commands and illegal-phase rejection.
- Multi-hand replay proving roster reuse, stack continuity, Button rotation,
  rebuy, void and table-clear gates.
- Recovery tests for retry/reconcile/void without physical motion.
- JSON/Schema/link checks, compileall, practical full suite, `git diff --check`
  and scoped status.

## Physical Motion Status

No physical motion is authorized or implemented. All dealing remains simulated.

## Commit Intent

Do not commit, push, branch, release or open a PR unless explicitly requested.

## Outcome Record

- Continuous Replay and development Live now use one frozen roster across
  multiple hands, persistent stacks, rotating Button and bounded hand count.
- Table clear, rebuy, session end and retry/reconcile/void decisions are
  explicit audited session-boundary controls.
- The session JSONL checker reopens and independently checks every referenced
  hand log and rejects hash, continuity, Button, ledger or lifecycle mismatch.
- Laptop owns a separate unvalidated 13-slot geometry and an offline still-image
  calibration tool; robot geometry and transport remain untouched.
- A 20-hand no-device qualification completed with all hand/session checks
  passing; the full repository suite passed 309 tests.
- No camera, microphone or physical dealer was opened during qualification.
