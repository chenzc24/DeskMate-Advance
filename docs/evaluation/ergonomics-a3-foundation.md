# Part A A3 Calibrated Semantic-State Foundation

Date: 2026-07-20

Status: **implementation validated; A3 release gate remains open**

## Delivered Capability

The existing Pose, Face, luminance and audio-level evidence now feeds eight
independent semantic-state channels:

| Channel | Primary evidence | Calibration dependency |
| --- | --- | --- |
| `static_too_long` | normalized upper-body motion and true timestamps | no |
| `bad_posture` | shoulder/torso deltas | neutral shoulder and optional torso |
| `screen_too_close` | face-area ratio, not absolute distance | neutral face area |
| `head_off_center` | raw matrix X/Y deltas | neutral head rotation |
| `low_blink_rate` | complete open-closed-open blinks over valid eye time | no |
| `environment_too_dark` | mean luminance | no |
| `environment_too_bright` | P90 luminance | no |
| `noise_too_high` | latest valid dBFS | no |

Every channel exposes `normal`, `warning`, or `unknown`, plus the explanatory
phase `idle`, `entering`, `active`, `exiting`, or `cooldown`. Channels are
updated independently; one active state cannot overwrite another. Confirmed
states also expose an `active_duration_ms` derived from timestamps for the
later event adapter.

The configuration at `configs/ergonomics/events.json` contains explicit entry,
exit, duration and cooldown values. Its status is
`development_defaults_not_acceptance_thresholds`: these numbers are tunable
defaults for implementation testing, not frozen product thresholds.

## Safety And Missing-Evidence Semantics

- Durations use monotonic nanosecond timestamps, never assumed frame counts.
- Camera misses, reported drops, or gaps over 500 ms inject `unknown`; they
  cannot finish entry/exit confirmation, and active duration pauses across the
  unobserved interval.
- `unknown` cannot confirm entry or clear a confirmed warning.
- Pose/Face stale states, temporal gaps, incomplete landmarks, invalid audio,
  insufficient blink coverage and failed calibration remain `unknown`.
- Numeric entry/exit hysteresis is separate from temporal confirmation.
- Partial posture coverage may confirm a visible shoulder problem, but it
  cannot claim a complete normal posture when torso evidence is unavailable.
- Screen distance is reported only as a ratio to the calibrated face-area
  proxy; no centimetre estimate is made.
- Head output is `head_off_center`. Left/right/up/down signs remain unfrozen.
- Audio is optional and latest-only. The background poller retains scalar
  RMS/dBFS, status and one error string; it does not retain sample windows.
- No semantic channel emits motor values or changes controller priority.

## Neutral Calibration

The live probe automatically opens one fixed five-second development
calibration window. The user should sit in the intended neutral posture, keep
the head and screen distance neutral, keep both shoulders visible, and keep the
eyes open normally. Only fresh valid scalar features are retained. Robust
medians form the profile; frames, landmarks, face images and audio samples are
not retained.

Calibration requires 20 valid shoulder samples, 20 valid Face geometry/head
samples and five valid luminance samples. A missing torso is recorded as
partial coverage rather than blocking the entire profile. Failed calibration
is frozen as `not_ready` with explicit reasons and must be restarted with `C`.

## Commands

Interactive camera states, with microphone disabled:

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py
```

Enable latest-only microphone level calculation explicitly:

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py `
  --enable-audio --audio-device-index 1
```

Bounded headless validation:

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/live_part_a.py `
  --headless --duration-seconds 7 --max-frames 220 `
  --enable-audio --audio-device-index 1
```

## Validation

Full repository tests after A3 integration: **106 passed**.

The tests cover strict timestamp ordering, three-valued evidence, entry/exit,
cooldown, numeric hysteresis, independent simultaneous states, robust
calibration, blink completeness and valid-time gating, stale evidence,
latest-only audio polling, resource cleanup and configuration validation.

A bounded real laptop run used camera index 0 through DirectShow at 1280x720,
Pose Full, Face Landmarker and microphone index 1. It retained no media or
audio.

| Metric | Result |
| --- | ---: |
| successful camera frames | 204 |
| capture duration / effective FPS | 6.89 s / 29.45 |
| Pose runs / valid / P95 | 55 / 55 / 23.76 ms |
| Face runs / valid / P95 | 55 / 55 / 17.84 ms |
| audio state | valid, latest dBFS available |
| calibration Face / luminance samples | 40 / 10 |
| calibration shoulder samples | 0 |
| calibration result | `not_ready: pose_samples:0/20` |

The real run is useful negative-path evidence: the user was close enough for
Face but the current framing did not expose valid shoulders, so calibration
correctly refused to create a profile. Consequently calibrated posture,
distance and head states remained `unknown`; the implementation did not invent
a neutral baseline. This run does not validate function accuracy.

After discontinuity and cleanup hardening, a second bounded three-second run
processed 86 frames at 29.39 effective FPS with no reported camera drop. The
audio poller ended with `poller_stopped: true`, valid scalar dBFS, and no
cleanup error. This regression run again retained no media or audio.

## Remaining Gate Work

A3 is not a release and its gate is not closed. It still requires the target
robotics camera contract, recorded replay fixtures, long negative sessions,
positive scenarios for each channel, event-candidate fixture validation, false
trigger rate, detection latency, threshold tuning and the final
acknowledgement/condition-cleared lifecycle. The current laptop path validates
wiring and safe rejection only. The script's duration/frame bounds are checked
between synchronous camera reads; unattended deployment still needs an outer
watchdog against a blocked Windows camera driver.
