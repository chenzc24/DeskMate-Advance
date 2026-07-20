# Part A A1 Foundation Evidence

Date: 2026-07-20
Status: **foundation validated; target-camera selection pending**

## Scope Validated

- Part A owns Pose, Face, luminance and audio-level observations under the
  `ergonomics` namespace.
- Pose and Face adapters consume the shared `FramePacket` without modifying
  shared camera/domain code.
- MediaPipe results are converted to immutable project records before leaving
  the adapter.
- VIDEO mode consumes timestamp milliseconds derived from monotonic capture
  nanoseconds and rejects non-increasing timestamps.
- Missing detections and inference/timestamp errors remain distinct from valid
  negative evidence.
- Luminance reports Rec. 709 mean, median, P10 and P90 without applying
  unfrozen thresholds.
- Audio reports normalized RMS and dBFS with explicit missing/error handling;
  it does not claim calibrated SPL.

## Commands And Results

```text
.venv\Scripts\python.exe scripts/ergonomics/smoke_part_a.py
.venv\Scripts\python.exe -m pytest -q tests
```

Results:

- Pose Full and Face assets initialized offline through the Part A VIDEO-mode
  adapters.
- A synthetic black frame correctly produced `missing`, not a confirmed
  negative person/face state.
- Synthetic luminance and silence produced valid raw statistics.
- Full repository test suite: `19 passed`.
- Part A configuration parsed successfully.

The one-frame synthetic timings are smoke diagnostics only. They are not
latency benchmarks and cannot choose Pose Full versus Lite.

## Not Yet Validated

- Final robotics camera, transport, resolution, frame rate and timestamp path.
- Pose Full versus Lite valid-output and P95 comparison on seated users.
- Face validity for distance, side angle, glasses, blink, low light and
  occlusion.
- Microphone device selection and any relative-SPL calibration.
- Product thresholds, temporal rules, false-trigger limits and event latency.
- Recorded replay determinism on a frozen participant/session manifest.

No model is promoted by this evidence. Gate A1 remains open until recorded
target-input evidence is available; A2 feature design may proceed against
explicitly exploratory local recordings without treating them as selection
evidence.
