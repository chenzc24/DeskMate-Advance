# Camera input module

This package is the single runtime entry point for live camera frames. On the
target laptop may use DirectShow camera index `0`; the actual table-camera index
must be established by the Stage 0 hardware inventory.

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

## Operator tools

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\camera\probe_camera.py --indexes 0
.\.venv\Scripts\python.exe scripts\camera\preview_camera.py --index 0
```

The preview does not record or save images. Close it before starting a model
that needs exclusive camera access.
