# Part A A2 Feature And Benchmark Evidence

Date: 2026-07-20

Status: **tooling validated; target-camera evidence unavailable; Gate A2 open**

## Implemented Boundary

Pose features now preserve:

- model ID/version, asset/config SHA-256, source/frame/timestamp and
  `dropped_before`;
- per-landmark missing mask from finite coordinates, visibility and presence;
- shoulder/hip-relative normalized landmarks and normalization scale;
- shoulder tilt, torso lean from image vertical and nose-to-shoulder offset;
- upper-body motion per true elapsed second, with no interpolation across
  missing or overlong gaps.

Face features now preserve:

- normalized face bounding-box center, width, height and area as a relative
  screen-distance proxy, never centimeters;
- raw XYZ Euler decomposition and translation from the optional 4 x 4 matrix;
- left/right blink blendshape evidence and valid-eye elapsed time;
- separate geometry, rotation and blink states, so a valid face does not imply
  that matrix or blink evidence exists.

The raw rotation axes and signs require target-camera neutral calibration.
MediaPipe documents the matrix as a canonical-face-to-detected-face transform,
primarily intended for face effects; Part A therefore does not label the raw
axes as product-level yaw/pitch/roll before calibration. The official task
returns optional blendshapes and transformation matrices, while Pose returns
33 normalized and world landmarks with visibility evidence:
[Face result](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult),
[Face options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions),
[Pose landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

## Recorded Benchmark Contract

`scripts/ergonomics/benchmark_recordings.py`:

1. accepts a tracked JSONL source manifest;
2. verifies video and timestamp-sidecar SHA-256 before decoding;
3. requires pseudonymous participant/session, device, scenario, split,
   consent and license status;
4. rejects duplicate content crossing splits;
5. requires contiguous frame indices and capture timestamps that increase at
   millisecond resolution;
6. sends identical frames and timestamps through Pose Full, Pose Lite and
   Face;
7. keeps bounded latency reservoirs and a per-recording frame limit;
8. writes detailed results only under ignored `artifacts/`;
9. records Git dirty state, environment, config/model hashes, valid/missing/
   error rates, feature availability, dropped counts, invalid duration,
   initialization, P50/P95/P99 latency, throughput and sampled memory.

No duration is reconstructed from nominal FPS. The timestamp sidecar is part
of the source evidence and has its own content hash.

## Validation Performed

```text
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe scripts/ergonomics/smoke_part_a.py
.venv\Scripts\python.exe scripts/ergonomics/benchmark_recordings.py \
  --manifest <local-manifest> --output artifacts/ergonomics/a2-benchmark.json
```

- Part A feature/benchmark focused tests cover geometry, translation/scale
  invariance, motion versus elapsed time, missing and long gaps, matrix
  decomposition, partial blink evidence, manifest hashes/splits, timestamp
  rejection and bounded percentile aggregation.
- A six-frame synthetic black-video replay completed end to end with matching
  video/sidecar length. Full, Lite and Face each reported six `missing`, zero
  `error`, as expected; JSON serialization contained no NaN.
- The synthetic run validates the harness only. It contains no person and is
  not accuracy, validity or candidate-selection evidence.

## Gate A2 Still Requires

- final robotics camera device/transport/resolution/FPS/color/timestamp
  contract;
- consented target-camera sessions covering neutral posture, lean/slouch,
  near/far, left/right/up/down, blink, glasses, low light, occlusion and long
  ordinary negative behaviour;
- participant/session split review before derived features are generated;
- two repeated replays for determinism;
- Full/Lite paired valid-output, stability and P95 comparison;
- Face geometry/matrix/blendshape availability by condition;
- human-frozen acceptance lines for valid rate, miss/false-trigger rate and
  latency.

Until those inputs exist, Pose Full remains the development primary candidate,
Pose Lite remains the development fallback candidate, and Face remains a
development candidate. No release promotion or fine-tuning decision is made.
