# Part A Laptop Camera Live Probe

Date: 2026-07-20

Status: **A3 semantic-state UI validated; not target-camera evidence**

## Command

Interactive, non-recording preview:

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py
```

Use `Q` or `Esc` to exit. The window displays:

- Pose valid/missing/error, cached age, latency and missing rate;
- shoulder tilt, torso lean and normalized upper-body motion;
- Face geometry/rotation/blink sub-states, face-area proxy, uncalibrated raw
  rotation XYZ and left/right blink scores;
- luminance statistics and age;
- a five-second neutral-calibration banner and explicit failure reasons;
- eight independent `normal/warning/unknown` semantic states and their phases;
- optional latest-only microphone dBFS when `--enable-audio` is supplied;
- pose skeleton, face box and effective capture FPS.

Use `C` to discard scalar calibration values and restart the neutral window.
The UI labels all event thresholds as development defaults.

Bounded headless smoke:

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py `
  --headless --max-frames 30 --duration-seconds 5
```

Optional microphone level (no audio recording):

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py `
  --enable-audio --audio-device-index 1
```

Headless mode is rejected unless a frame or duration bound is supplied. The
probe contains no `VideoWriter`, `imwrite`, screenshot or sample serialization
path. It prints only compact aggregate metrics and the latest non-landmark
evidence.

## Runtime Design

- Shared `OpenCVCamera` remains unchanged and owns the laptop device.
- Camera capture is synchronous with a capacity-one backend preference; the
  probe creates no frame queue.
- Pose and Face default to independent 10 Hz cadence over the latest frame.
- Full-frame luminance percentiles run at 2 Hz to avoid consuming each display
  frame's latency budget.
- Repeated or backward camera timestamps are rejected instead of moving the
  model scheduler backward.
- Cached model output has an explicit age and becomes stale after 500 ms.
- Model latency samples use a bounded reservoir.
- Missing, error and stale remain distinct and do not become normal negative
  evidence.
- Model assets and config hashes are verified before the camera is opened.

## Real Laptop Smoke Evidence

Device: camera index 0, DirectShow, 1280 x 720, requested 30 FPS.

Model: Pose Full plus Face Landmarker.

Bound: 20 successful frames or 4 seconds.

Media retained: none.

Observed result:

| Metric | Result |
| --- | ---: |
| successful camera frames | 20 |
| camera missing/disconnected | 0 / 0 |
| adapter-reported drops | 0 |
| effective capture FPS | 28.21 |
| Pose runs / valid / error | 6 / 6 / 0 |
| Pose P50 / P95 | 18.97 / 29.72 ms |
| Face runs / valid / error | 6 / 6 / 0 |
| Face P50 / P95 | 12.72 / 15.96 ms |
| luminance runs | 2 |

The latest compact evidence included a shoulder tilt of approximately -1.45
degrees, raw face-area proxy, raw rotation and both blink scores. Only about
39.4% of Pose landmarks met the configured visibility/presence gate, and torso
lean was unavailable because the laptop framing did not provide valid hips.
This is useful failure evidence: the current view can exercise face and
upper-body functions, but `bad_posture` cannot depend exclusively on visible
hips without changing camera framing or adding a calibrated upper-body
fallback.

The run is too short and uncontrolled for accuracy, false-trigger, calibration
or Pose Full/Lite selection claims. It proves only that the live pipeline is
connected, bounded, offline and capable of producing real Part A evidence.

## A3 Bounded Camera And Microphone Smoke

A later seven-second headless run processed 204 frames at 29.45 effective FPS,
with 55/55 valid Pose and Face runs. P95 inference latency was 23.76 ms for Pose
Full and 17.84 ms for Face. Microphone index 1 produced a valid latest dBFS
level; no samples were retained.

The view supplied 40 valid Face calibration samples and ten luminance samples,
but zero valid shoulder samples. Calibration therefore ended as
`not_ready: pose_samples:0/20`, and all profile-dependent states remained
`unknown`. This is the intended safe failure path and is not positive accuracy
evidence. See `ergonomics-a3-foundation.md` for the complete A3 status.
