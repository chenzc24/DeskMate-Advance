# Microphone input module

This package is the single runtime entry point for live microphone blocks. The
current laptop uses input device index `1`, the Intel Smart Sound microphone
array. DroidCam Audio is not used.

## Public model-facing API

```python
from deskmate_advance.perception.audio import (
    AudioReadStatus,
    MicrophoneConfig,
    SoundDeviceMicrophone,
)

config = MicrophoneConfig(
    device_index=1,
    source_id="intel_smart_sound_microphone_array",
    sample_rate_hz=16_000,
    channel_count=1,
    block_duration_ms=100,
)

with SoundDeviceMicrophone(config) as microphone:
    result = microphone.read()
    if result.status is AudioReadStatus.OK:
        packet = result.packet
        samples = packet.samples
        timestamp_ns = packet.captured_at_ns
```

Samples are an owned, read-only `float32` NumPy array with shape
`(sample_count, channel_count)`. Use `captured_at_ns` for event timing. A
`degraded` result contains an overflow-marked packet; `missing` and
`disconnected` contain no packet. These states become unknown input and must not
be interpreted as confirmed silence.

The synchronous adapter allocates exactly one configured block per read and
does not create an accumulating background queue.

## Device probe and smoke read

The default command lists devices and checks the configured format without
capturing audio:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\probe_microphone.py
```

An explicit smoke test reads one 100 ms block, prints only RMS and metadata, and
does not save samples:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\probe_microphone.py --read-once
```
