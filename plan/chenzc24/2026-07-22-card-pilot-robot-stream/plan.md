# Card Pilot Robot Stream Input

- Outcome and owned paths: allow `scripts/perception/live_card_pilot.py` to
  consume the existing Raspberry Pi HTTP(S) MJPEG stream through the owned
  `OpenCVCamera` boundary while preserving local-camera compatibility. Owned
  paths are the card pilot script, its Stage 2B usage documentation, focused
  CLI tests and this plan.
- Dirty paths left read-only: all game/ledger contracts, model code and
  weights, camera adapter internals, other perception runtimes, datasets,
  manifests and unrelated plans.
- External dependencies: the operator-provided private-network Raspberry Pi
  MJPEG endpoint, the already ignored and hash-verified LGD ONNX/class assets,
  OpenCV FFmpeg support and network reachability. The endpoint is never stored
  in output or committed configuration.
- Validation and physical motion: run focused CLI tests, the card pilot tests,
  the practical full Python suite, JSON parsing, `git diff --check` and scoped
  Git status. Perform only a bounded, non-recording headless stream smoke when
  the endpoint is reachable. No robot-control connection or physical motion is
  involved or authorized.
- Commit intent: do not commit or push unless the user explicitly requests it.

## Completed Outcome

- Added `--stream-url`, bounded open/read timeout options and explicit
  local-index/backend conflict rejection to the live card pilot while keeping
  its prior DirectShow defaults unchanged.
- Reused the existing FFmpeg-backed latest-frame camera boundary; no new
  decoder, queue or frame persistence path was introduced.
- Focused card/CLI tests passed (`12 passed`) and the practical full suite
  passed (`232 passed`). CLI help and `git diff --check` passed.
- The private-network Raspberry Pi endpoint negotiated 640x480 at a reported
  25 FPS. A bounded headless run read 20/20 frames with zero missing reads;
  the empty fixture ROI produced 20 safe `unknown/no_detection` observations.
- No frames were saved, no game state was changed, no robot-control endpoint
  was contacted and no physical motion was authorized.
