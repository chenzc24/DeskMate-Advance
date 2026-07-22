# Camera input module

This package is the single runtime entry point for live camera frames. The
target laptop may use DirectShow camera index `0`, while the robot development
camera may expose an HTTP(S) MJPEG stream. The final table-camera source and
geometry must still be established by the Stage 0 hardware inventory.

## Public model-facing API

Import only from the package root:

```python
from poker_dealer.io.camera import (
    CameraConfig,
    CameraReadStatus,
    OpenCVCamera,
)

config = CameraConfig(
    device_index=0,
    source_id="table_camera",
    backend="dshow",
    width=1280,
    height=720,
    fps=30,
)

with OpenCVCamera(config) as camera:
    result = camera.read()
    if result.status is CameraReadStatus.OK:
        packet = result.frame
        image_bgr = packet.image
        timestamp_ns = packet.captured_at_ns
```

The image is an owned, read-only BGR NumPy array. Convert it to RGB inside the
perception adapter when a model requires RGB. Use `captured_at_ns` for duration;
never infer elapsed time from FPS. Treat `missing` and `disconnected` as unknown
input, not as confirmed negative evidence.

For an MJPEG source, use a network config instead of a local camera index:

```python
config = CameraConfig(
    stream_url="http://robot.local:5000/video_feed",
    source_id="robot_mjpeg_stream",
    backend="auto",
    width=None,
    height=None,
    fps=None,
    open_timeout_ms=5000,
    read_timeout_ms=2000,
)
```

The network reader uses FFmpeg timeouts and a background latest-frame buffer of
exactly one frame. Frames overwritten before the model consumes them are
reported through `FramePacket.dropped_before`; they are never queued for later
action inference. Network timestamps are local monotonic receive times because
the MJPEG endpoint does not provide capture timestamps. Stream URLs must be
absolute HTTP(S) URLs and may not contain embedded credentials.

Three consecutive decode failures close only the current MJPEG connection.
The reader makes up to five bounded reopen attempts with a 250 ms backoff while
returning `missing` observations to perception. A successful reopen preserves
the frame contract and increments `network_reconnects`; only exhausted reopen
attempts produce `disconnected`. The caller must keep identity/action state
unchanged during `missing` input.

## Operator tools

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\camera\probe_camera.py --indexes 0
.\.venv\Scripts\python.exe scripts\camera\preview_camera.py --index 0
.\.venv\Scripts\python.exe scripts\camera\preview_camera.py --stream-url http://100.80.46.54:5000/video_feed
```

The preview does not record or save images. Close it before starting a model
that needs exclusive camera access.
