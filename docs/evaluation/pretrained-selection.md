# Stage 1 Pretrained Component Selection

Status: **ACTIVE / no component is released or finally selected**
Started: 2026-07-20
Primary runtime target: laptop
Final camera source: `pending_robotics`

## Purpose

Select one reproducible pretrained perception path for each required DeskMate
function using target-input evidence, runtime metrics, and explicit failure
behaviour. Public benchmark rankings or a successful single-frame demo are not
selection evidence.

Stage 1 may proceed while Gate 0 is partially frozen, but final selection is
blocked where the missing camera contract or acceptance thresholds materially
affect the result.

## Frozen Inputs From Stage 0

- Laptop is the P0 production model runtime.
- `phone_usage_detected` means evidence of active phone use, not phone presence
  alone.
- Static/reminder durations and cooldowns are configurable interfaces.
- Static, phone, and gesture semantic events may be active in parallel; the
  model/event layer assigns no priority between them.
- Laptop microphone input is available.
- Gesture-to-command mapping is intentionally deferred; the visual classes
  remain the proposed `wave`, `swipe`, `circle`, and `no_gesture` set.

## Unfrozen Inputs Kept Visible

| Item | Status | Stage 1 treatment |
| --- | --- | --- |
| final robot camera device and transport | `pending_robotics` | use local laptop camera only for exploratory smoke/benchmark work; do not freeze results |
| acknowledgement versus condition-cleared lifecycle | pending discussion | do not embed final event lifecycle in perception adapters |
| model/communication/controller failure safe state | pending discussion | perception must expose missing/stale/disconnected input; controller behaviour remains open |
| false-trigger, miss, and response-latency acceptance lines | pending discussion | record raw metrics and distributions; do not label a candidate accepted |
| gesture-to-command mapping | deferred by design | benchmark landmark quality only; keep action mapping outside the model |

## Runtime Inventory

The following is exploratory machine evidence, not a portable environment
contract:

- CPU: Intel Core i9-14900HX, 24 cores / 32 logical processors.
- GPU devices include Intel UHD Graphics and NVIDIA GeForce RTX 4070 Laptop
  GPU. P0 benchmarks must report CPU results; GPU acceleration is optional and
  must not be assumed.
- Project virtual environment: Python 3.13.1.
- Installed in the project environment: NumPy 2.5.1, OpenCV Python 4.13.0.92,
  pytest 9.1.1.
- Not installed at inventory time: MediaPipe, PyYAML, psutil, sounddevice.
- Available exploratory camera: HP True Vision FHD Camera. DroidCam devices are
  present but are not the default source.
- Available audio endpoints include the laptop microphone array and DroidCam
  Audio. The exact P0 microphone device index remains a runtime configuration.

A read-only pip dry run on Python 3.13.1 resolved `mediapipe==0.10.35`, but it
would also install `opencv-contrib-python==5.0.0.93` while the current camera
target uses `opencv-python==4.13.0.92`. Do not install MediaPipe until the
project chooses one OpenCV package family and verifies the existing camera
adapter/tests against that resolved environment.

## Official API Findings

- MediaPipe Tasks officially supports desktop Windows and Python 3.9 or later,
  and loads model assets from an explicit local path. See the
  [official Python setup guide](https://developers.google.com/edge/mediapipe/solutions/setup_python).
- Pose Landmarker supports image, video, and live-stream modes and produces 33
  normalized and world pose landmarks. See the
  [official Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
- Hand Landmarker produces handedness plus 21 normalized and world landmarks,
  which matches the planned `T × 21 × 3` gesture input. See the
  [official Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
- Face Landmarker produces face landmarks and can optionally produce
  blendshapes and a facial transformation matrix. See the
  [official Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).
- MediaPipe Object Detector supports image, video, and live-stream modes; the
  official Python example uses an EfficientDet-Lite0 TFLite asset. See the
  [official Object Detector guide](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python).
- MediaPipe live-stream task APIs may ignore new input frames when a task is
  busy. The project must therefore keep monotonic timestamps, bounded queues,
  and explicit missing/stale accounting rather than assuming one result per
  camera frame.

## Initial Candidate Ledger

| Component ID | Functions | Candidate | Status | Required output | Current blocker |
| --- | --- | --- | --- | --- | --- |
| `pose_landmarker` | sitting/static, posture, motion | MediaPipe Pose Landmarker | `shortlisted` | 33 landmarks, visibility/presence, optional world coordinates | target camera evidence and model variant/asset not frozen |
| `face_landmarker` | screen distance, head direction, blink evidence | MediaPipe Face Landmarker | `shortlisted` | face landmarks, optional blendshapes and transformation matrix | camera distance/angle evidence and enabled optional outputs not frozen |
| `hand_landmarker` | dynamic gesture features, phone-use fusion | MediaPipe Hand Landmarker | `shortlisted_priority` | handedness, 21 landmarks, world landmarks, timestamp | OpenCV/MediaPipe environment resolution and target camera evidence |
| `phone_detector` | phone evidence for active-use fusion | MediaPipe Object Detector with EfficientDet-Lite0 candidate | `shortlisted_unverified` | phone box, class, confidence, timestamp | phone label metadata, small-target/occlusion recall and target camera unknown |
| `audio_level` | environmental noise | RMS / relative SPL program logic | `required_non_model` | timestamped level and validity | microphone device/config and calibration procedure |
| `identity` | personalized greeting | YuNet + SFace | `deferred_p2` | embedding and similarity with quality gate | outside current P0/P1 scope |
| `audio_classification` | sound type | YAMNet candidate | `deferred_p2` | class probabilities | no frozen product need |

No candidate above is a release decision. `shortlisted` means only that the
candidate has the required API shape and is allowed into target-input
benchmarking.

## Function Coverage Check

| PPT function | Stage 1 source | Learned later? |
| --- | --- | --- |
| `static_too_long` | Pose landmarks + timestamped motion features | only if rule path fails |
| `bad_posture` | Pose landmarks + derived angles | conditional posture model |
| `screen_too_close` | Face landmarks/transformation + calibration | no core training planned |
| `low_blink_rate` | Face eye landmarks/blendshapes + valid-time window | no core training planned |
| `phone_usage_detected` | Phone detector + Hand + optional head direction + focus state | detector fine-tune only if evidence justifies it |
| `gesture_detected` | Hand landmarks | TCN is the primary learned extension |
| brightness events | image statistics | no |
| `noise_too_high` | audio RMS/relative SPL | no |
| identity events | YuNet/SFace | deferred P2 |
| `off_task_behaviour` | fused events/features | deferred conditional model |

## Benchmark Evidence Contract

Every candidate run must use the same recorded session manifest and report:

- Git commit and dirty state;
- config and model asset SHA-256;
- source ID, resolution, nominal and observed frame rate;
- processed frames, produced observations, missing/stale/dropped counts;
- initialization time;
- per-frame P50/P95 latency and full-loop FPS;
- CPU and memory;
- valid-output rate;
- scenario-specific misses and false positives;
- offline/no-network model loading;
- failure behaviour on empty, disconnected, malformed, or stale input.

Landmarker-specific checks:

- Pose: seated upper-body visibility, shoulder/neck stability, partial-body
  crops, side view, low light.
- Face: user-to-camera distance, side angle, glasses, low light, blink evidence,
  head-direction stability.
- Hand: left/right consistency, fast movement, leave/re-enter, occlusion,
  landmark jitter and gaps.
- Phone: small targets, partial occlusion, hand overlap, phone lying unused on
  the desk, screen-like distractors and class-label coverage.

## Stage 1 Work Batches

### S1-A — Environment and input inventory

Status: **in progress**.

- Laptop runtime and project environment inventoried.
- Existing uncommitted laptop camera adapter is treated as a separate,
  read-only exploratory input target.
- Final robotics camera remains pending.
- MediaPipe/OpenCV dependency-family conflict identified before installation.

Exit: one project environment choice, one exploratory source, and explicit
final-camera pending status.

### S1-B — Candidate asset manifest

Status: **not started**.

- Resolve the OpenCV package family with the camera work owner.
- Add MediaPipe as an explicit project dependency only after compatibility is
  demonstrated.
- Select exact Pose/Face/Hand model variants.
- Verify EfficientDet-Lite0 metadata includes the required phone category.
- Download assets once, record source/license/hash, and prohibit runtime
  download.

Exit: local immutable assets and metadata registered without declaring model
acceptance.

### S1-C — Common benchmark harness

Status: **not started**.

- Build project-owned observation adapters; MediaPipe objects do not cross the
  boundary.
- Support recorded-video mode first for deterministic comparison.
- Add live mode after recorded evidence is stable.
- Record missing/dropped results caused by asynchronous task backpressure.

Exit: all candidates produce the same benchmark record shape.

### S1-D — Exploratory laptop-camera evidence

Status: **blocked on S1-B/S1-C**.

- Use HP True Vision only as an explicitly exploratory source.
- Capture no private media into Git.
- Report performance and validity without claiming it represents the final
  robot camera.

Exit: smoke evidence for APIs and runtime scheduling.

### S1-E — Target-camera selection evidence

Status: **blocked on robotics decision**.

- Freeze device/transport/resolution/FPS/color/timestamp contract.
- Build a session manifest covering distance, angle, lighting, occlusion and
  normal negative behaviour.
- Run the common harness and compare evidence against acceptance lines once
  those lines are frozen.

Exit: component selection record with release/fallback decisions.

## Current Stage 1 Decision

Proceed with API-compatible shortlists and environment preparation. Do not yet:

- install MediaPipe into the current project environment;
- download or promote model assets;
- freeze a phone detector;
- treat the laptop camera as the final robot source;
- declare Gate 1 passed.

The next executable decision is resolving the OpenCV package family with the
existing laptop-camera target, followed by a recorded-input benchmark skeleton
for MediaPipe Hand Landmarker as the highest-priority feature extractor.
