# Robot MJPEG Camera Input

## Outcome And Owned Paths

Add a backward-compatible network-camera mode for the robot MJPEG endpoint.
The camera boundary will convert the stream into the existing immutable
`FramePacket` contract, retain only the latest frame in memory, expose bounded
open/read timeouts and explicit disconnect semantics, and make the mode
available to the camera preview, Part A sequential runtime and four-player
acceptance runner.

Owned paths are the camera adapter/tests and the three CLI/documentation entry
points needed to select the stream. Existing identity, gesture, speech, game,
ledger and robot-command logic remain unchanged.

## Dirty Read-Only Paths

Preserve all pre-existing uncommitted Part A preparation changes. In
particular, do not rewrite the acceptance protocol, model/training work,
master-plan decisions, game rules, Part B paths or model assets. Changes to the
already-dirty sequential Part A runtime must be limited to camera selection.

## External Dependencies

The operator-provided endpoint is
`http://100.80.46.54:5000/video_feed`. It is an unauthenticated HTTP MJPEG
development stream reachable on the current private network. Availability,
camera mounting, focus, exposure, robot-settle timing and card readability are
external facts and are not frozen by this software change.

## Validation And Physical Motion

Run camera and CLI unit tests, the practical full test suite, JSON parsing where
applicable, `git diff --check` and scoped `git status`. Perform a bounded,
non-recording live decode and a headless Part A smoke using the robot stream.
Do not save frames, audio or embeddings. Do not connect to a robot-control
endpoint or authorize physical motion; the Part A rotation adapter remains the
simulated dealer.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Outcome

- Added validated HTTP(S) MJPEG selection without changing the local camera
  index path. Embedded URL credentials are rejected and the endpoint itself is
  not persisted by the runtime.
- The FFmpeg-backed reader uses 5 s open and 2 s read defaults, a daemon reader
  and an in-memory latest-frame buffer of one. Overwritten frames are reported
  through `FramePacket.dropped_before`; failures become explicit
  `missing/disconnected` observations.
- Added `--stream-url` to the non-recording preview, sequential Part A runtime
  and four-player acceptance runner. The Part A summary now reports total
  dropped camera frames.
- The real endpoint negotiated 640x480 at a reported 25 FPS. A full Part A
  headless run consumed 60 frames with zero missing reads and eight deliberate
  latest-buffer drops; no frames, audio or embeddings were saved.
- Camera/CLI tests passed and the final practical suite passed: 183 tests.
  Compile and `git diff --check` passed.
- Physical motion remained disabled. The runtime still reports
  `rotation_adapter=simulated_dealer_only` and
  `physical_robot_connected=false`.
- No commit or push was performed.

## Reconnect Correction

Two interactive runs later showed that the server stayed reachable while an
individual MJPEG client connection stopped decoding after roughly 55-61
seconds. The initial implementation incorrectly promoted three decode failures
to permanent camera disconnection. The network reader now performs up to five
bounded reopen attempts with a 250 ms backoff, emits temporary missing evidence
while reopening and increments a reconnect counter after recovery. Part A logs
`camera_read_status` and `camera_reconnected`; registration, game focus and the
ledger remain unchanged during the gap. Persistent reopen failure is still a
hard disconnect and safe exit.

The corrected adapter passed a real 80-second headless Part A run across the
previous failure window: 2,294 frames, zero missing reads, 94 intentional
latest-buffer drops and no reconnect required in that particular run. A
deterministic fake-stream test forced three decode failures, reopened a second
capture, preserved the source/frame sequence contract and delivered the next
frame with reconnect count one. The final full suite passed: 186 tests; compile
and diff checks passed.
