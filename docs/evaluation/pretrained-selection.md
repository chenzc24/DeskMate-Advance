# Stage 1 Pretrained Component Selection

Status: **INITIAL SELECTION COMPLETE / TARGET VALIDATION PENDING**

Updated: 2026-07-20

Primary runtime: laptop, CPU baseline

Final camera: `pending_robotics`

## Decision boundary

This record separates three meanings that must not be conflated:

- **initial selection**: the API, local asset, output shape and architecture role are suitable;
- **development asset admission**: exact bytes are hashed and can run offline in the project environment;
- **release/fallback selection**: target-camera validity, scenario errors and runtime gates have passed.

The first two are complete for Pose, Face, Hand and Phone. The third is not.

## Frozen Stage 0 inputs

- Laptop is the formal model runtime.
- `phone_usage_detected` means active phone use, not phone visibility alone.
- Durations and cooldowns are configurable.
- Static, phone and gesture events are parallel; the model layer gives them no priority.
- Laptop microphone input is available.
- Gesture classes remain `wave`, `swipe`, `circle`, `no_gesture` and `unknown`; their controller commands are deliberately deferred.

Still unfrozen: final camera contract, acknowledged-versus-cleared lifecycle, failure safe-state ownership and quantitative acceptance lines.

## Environment resolution

The project environment previously had `opencv-python==4.13.0.92`; installing MediaPipe would introduce a second OpenCV wheel family. A fresh temporary Python 3.13 environment was first used to verify the complete proposed set. The project environment was then changed to:

| Dependency | Resolved version |
| --- | --- |
| Python | 3.13.1; project range `>=3.11,<3.14` |
| MediaPipe | 0.10.35 |
| OpenCV distribution | `opencv-contrib-python==5.0.0.93` only |
| `cv2` | 5.0.0 |
| NumPy | 2.5.1 |
| sounddevice | 0.5.5 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |

Post-change evidence:

- `opencv-python` is absent, so two wheel families cannot shadow the same `cv2` namespace;
- MediaPipe, OpenCV, NumPy, sounddevice and psutil import together;
- the existing camera boundary passed all 11 tests unchanged;
- all selected assets load from local paths without runtime download.

This resolves the current laptop environment risk, not every future platform risk. A Python, OS, architecture or dependency update must repeat the clean-environment and camera tests.

## Initial model choices

| Component | Primary candidate | Fallback / later branch | Reason | Release blocker |
| --- | --- | --- | --- | --- |
| Pose | MediaPipe Pose Landmarker Full | Pose Lite | Keep the stronger starting variant for seated/partial-body validity; compare Lite only on identical recordings if CPU budget requires it | final camera, validity, P95, Full-vs-Lite comparison |
| Face | MediaPipe Face Landmarker | disable optional outputs if unneeded | One bundle provides landmarks plus optional blendshapes and transformation matrix for blink/head/distance evidence | calibration, glasses/angle/low-light evidence, optional-output CPU cost |
| Hand | MediaPipe Hand Landmarker Full | invalid/unknown when landmarks are missing | Provides handedness and 21 normalized/world landmarks; it must be frozen before formal gesture data extraction | fast-motion and occlusion gap rates, one/two-hand requirement, final camera |
| Phone | MediaPipe Object Detector, EfficientDet-Lite0 int8 | compare Lite2 or fine-tune only after failure evidence | Official label map contains `cell phone`; Lite0 is the low-complexity starting detector | small-target/occlusion recall, idle-phone negatives, final camera |
| Noise | RMS / relative SPL | YAMNet deferred P2 | Requirement is level, not sound class | microphone selection and calibration policy |
| Brightness | frame statistics | none | No learned model is needed | target-camera exposure behaviour |
| Identity | YuNet/SFace deferred P2 | none | Adds privacy/data scope and is not core P0/P1 | explicit product need and consent path |
| Off-task | rule fusion first; learned model deferred | conditional MLP after data gate | Avoids an ill-defined early multimodal label | frozen label protocol and sufficient continuous sessions |

Official API facts used for admission:

- [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python) documents local model asset paths and Windows Python setup.
- [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) provides normalized and world pose landmarks in image/video/live-stream modes.
- [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) provides handedness and 21 normalized/world landmarks.
- [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) can output landmarks, blendshapes and transformation matrices.
- [Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python) supports EfficientDet-Lite task assets and category filtering.

## Offline asset admission

The binary files live under ignored `models/assets/`. `models/manifest.yaml` is the tracked source of their URL, size, version role and SHA-256. Because the official URLs contain `latest`, the local hash—not the URL text—is the immutable identity.

The MediaPipe framework repository is Apache-2.0, but the reviewed task pages do not explicitly state a separate license for each hosted model bundle. The manifest therefore records model-asset licensing as pending and restricts the assets to local project evaluation until a human verifies the applicable terms. The framework license is not silently applied to separately hosted model bytes.

| Asset | Bytes | SHA-256 prefix | Role |
| --- | ---: | --- | --- |
| Pose Full | 9,398,198 | `4eaa5eb7a983` | primary Gate 1 evaluation path |
| Pose Lite | 5,777,746 | `59929e1d1ee9` | fallback Gate 1 evaluation path |
| Hand | 7,819,105 | `fbc2a30080c3` | primary Gate 1 evaluation path |
| Face | 3,758,596 | `64184e229b26` | primary Gate 1 evaluation path |
| EfficientDet-Lite0 int8 | 4,602,795 | `0720bf247bd7` | primary Gate 1 evaluation path |
| COCO label map | 661 | `f8803ef79001` | supporting metadata; `cell phone` verified |

All model entries remain `development`; none is marked `candidate`, `release` or `fallback`. Promotion occurs only after the relevant workstream gate and target-camera validation, by the single integration owner.

## Synthetic smoke evidence

`scripts/evaluation/smoke_mediapipe_tasks.py` verifies every local asset's size and SHA-256, initializes the task and performs one inference on a 640×480 black RGB image. The last execution produced zero observations for all tasks at the configured phone threshold and exited successfully.

One-shot timings are deliberately not preserved as performance claims: task startup warms delegates/caches, a black image does not exercise landmark tracking, and independent task calls do not represent the shared full loop. Performance selection requires recorded target-camera sessions with repeated P50/P95 measurements.

## Runtime architecture decision

Use separate Pose, Face, Hand and Phone task lanes rather than binding all consumers to one monolithic graph. Each lane has capacity-one latest-frame semantics, an independent configurable cadence and explicit skipped/stale counters. Convert BGR to RGB once, then fan out the shared immutable view.

Initial benchmark cadences are Hand 15 Hz, Pose 10 Hz, Face 10 Hz and Phone 5 Hz. They are starting parameters only. Synchronous VIDEO mode is the deterministic benchmark path; bounded LIVE_STREAM mode is the runtime path. Both must emit the same project-owned observation records, never MediaPipe objects.

The detailed data flow, fusion rules and tradeoffs are in `docs/architecture/perception-architecture.md`.

## Function coverage

| PPT function | Selected source | Learned extension |
| --- | --- | --- |
| `static_too_long` | Pose + timestamped movement | only if rules fail across target conditions |
| `bad_posture` | Pose + calibrated angles | conditional posture model |
| `screen_too_close` | Face + camera calibration | none currently |
| `low_blink_rate` | Face eye evidence + valid-time window | none currently |
| `phone_usage_detected` | Phone + Hand + time; optional head support | detector fine-tune only after measured failure |
| `gesture_detected` | Hand landmark sequences | compact TCN, primary learned extension |
| bright/dark | image statistics | none |
| `noise_too_high` | RMS / relative SPL | none |
| identity events | deferred | YuNet/SFace P2 |
| `off_task_behaviour` | deferred rule fusion | conditional model P2 |

## Remaining Gate 1 evidence

The final robot camera must first freeze device, transport, resolution, observed FPS, color and timestamp behaviour. One participant/session-aware recording manifest must then cover seated posture, partial crops, distance, angle, glasses, low light, fast hands, occlusion, phone in-use, phone idle on desk and screen-like distractors.

Every candidate comparison must report:

- source/model/config hashes and dirty Git state;
- valid, missing, stale and skipped rates;
- initialization, per-module P50/P95, full-loop FPS, CPU and memory;
- scenario misses and false positives;
- offline cold start and malformed/disconnected/stale input behaviour.

Gate 1 remains blocked until quantitative error and latency acceptance lines are frozen and the target-camera evidence meets them.

## Stage 1 batch status

| Batch | Status | Meaning |
| --- | --- | --- |
| S1-A Environment/input inventory | complete for laptop | final robotics camera remains pending |
| S1-B Development asset manifest | complete | exact evaluation assets admitted, no candidate/release promotion |
| S1-C Common benchmark harness | partial | asset smoke exists; project observation adapters and recorded benchmark remain |
| S1-D Exploratory laptop evidence | ready, optional | cannot substitute for target-camera evidence |
| S1-E Target-camera selection | blocked | waiting for robotics camera contract and acceptance lines |

The next implementation target is the project-owned observation adapters plus deterministic recorded-video benchmark. Formal gesture dataset extraction begins only after the Hand extractor/config passes the target-camera gate and is frozen.
