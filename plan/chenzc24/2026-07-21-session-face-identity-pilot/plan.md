# Session Face Identity Pilot

## Outcome And Owned Paths

Implement the user-approved session-only pipeline `face detection -> aligned
embedding -> consented in-memory enrollment gallery -> player_id or unknown`.
The feature verifies the player registered to a seat after robot/state focus;
it never selects the acting seat, moves cards/chips, or changes the ledger.

Owned paths are the S0-21 product-policy amendments, the face identity config
and observation schema, `src/poker_dealer/perception/identity/`, the Laptop
identity UI, scoped tests, development model manifest entries, Stage 0/2
evaluation documentation, repository rules needed to preserve the new privacy
boundary, and this plan. Model weights stay ignored under `models/assets/`.

## Dirty Paths Left Read-Only

Betting rules, seat order, action/card observations, ledger, card perception,
robotics transports and unrelated archived work remain read-only. S0-17 stays
frozen: deterministic game state is the sole acting-seat authority.

## External Dependencies

- Official OpenCV Zoo YuNet face detector and SFace embedding model.
- OpenCV 5.0.0 FaceDetectorYN/FaceRecognizerSF and Laptop camera 0.
- Every enrolled participant must explicitly consent. Final thresholds require
  held-out participant/session evidence and cannot be copied from a benchmark.

## Validation And Physical Motion

Validate model hashes/load, exact-one-face enrollment, consent requirement,
unique player/seat registration, normalized embeddings, cosine threshold and
margin rejection, unknown/ambiguous/multiple-face behavior, memory-only clear,
schema output, state/ledger isolation, full tests and bounded camera UI smoke.
No frames or embeddings are persisted. No robot connection or physical motion
is authorized.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Outcome

- Registered S0-21 as an explicit product amendment while preserving S0-17
  state-machine seat authority and game/ledger isolation.
- Pinned and hash-verified official OpenCV Zoo YuNet and SFace development
  assets; runtime downloads remain prohibited.
- Implemented exact-one-face detection/alignment/embedding, consent-gated
  four-player in-memory enrollment, cosine threshold/margin rejection,
  temporal confirmation and `player_id`/unknown/mismatch observations.
- Added a non-recording Laptop UI and session cleanup on `X` or process exit.
- Passed 18 scoped identity/contract tests and the full 130-test suite. A
  20-frame camera-0 smoke completed with zero missing reads, 31.19 ms mean
  inference latency, no saved frames and no persisted embeddings.
- Physical-motion status remains `not connected / not authorized`. No commit,
  push, branch or release was created.
