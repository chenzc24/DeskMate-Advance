# Model Admission Status — 2026-07-24

## Decision

The runtime-selected MediaPipe gesture/pose, Vosk English/speaker, YuNet and
SFace assets have been exercised successfully in Laptop and robot-camera
development flows. They are now selectively visible to source control and
configured for Git LFS so an offline checkout can package the runtime without
downloads.

This packaging decision is not a model-release decision. The repository has no
held-out participant/session report containing all required per-action
precision/recall/F1, confusion, rejection/calibration, false accepted actions
per hour and hand, cross-seat leakage, cancellation behavior and P95
confirmation latency. Face and speaker verification also lack held-out
false-match/false-non-match and attack/noise coverage. Their manifest state
therefore remains `development`.

The two card families remain source-controlled:

- LGD gen3 stays a development baseline.
- Poker Dealer v2 stays `candidate` while target-camera card debugging and
  held-out deck/session evaluation continue.

No asset was marked `release`, and no admission blocker or metric was invented.
