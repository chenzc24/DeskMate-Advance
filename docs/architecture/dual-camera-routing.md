# Dual-Camera Perception Routing

The live runtime may use two explicitly owned camera roles:

- `player_camera`: registration, face verification, visual settle, pose,
  gesture and player-action evidence.
- `camera` (the table camera): card delivery confirmation, card recognition and
  table-clear views.

`camera` remains the required profile field for backwards compatibility.
`player_camera` is optional. When it is absent, both roles resolve to the same
camera and the runtime preserves the original single-camera behavior.

The active robot-camera AudioRelay profile uses:

```json
{
  "camera": {
    "kind": "mjpeg",
    "source_id": "robot_mjpeg_camera",
    "stream_endpoint": "robot_camera"
  },
  "player_camera": {
    "kind": "local",
    "source_id": "droidcam_player_camera",
    "device_index": 1,
    "backend": "msmf",
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```

The table stream URL remains centralized in
`configs/runtime/network_endpoints.json`. DroidCam is consumed through its
Windows virtual-camera device because its direct `/video` endpoint is exclusive
while the DroidCam Windows client is active.

## State-Owned Route Selection

The runtime loop, not a model, chooses the route:

| Runtime activity | Camera route |
| --- | --- |
| Player registration | player |
| Player visual settle | player |
| Session face verification | player |
| Pose, gesture and speech-bound action | player |
| Hole-card delivery confirmation | table |
| Board-card recognition | table |
| Between-hand table-clear view | table |

The selected camera still returns the existing immutable `FramePacket`
contract. Perception adapters receive a frame and emit evidence exactly as
before; they cannot select a camera, focus a seat, or mutate game state.

## Failure And Privacy

- Failure of the camera required by the current state pauses the hand through
  the existing camera-disconnect recovery path.
- Both camera resources are locked before a live session opens.
- The mobile console shows only the currently selected route.
- Diagnostic metadata may record route, source ID, timing and failure reason;
  frames and face embeddings are not persisted.
- A configured player camera does not validate physical geometry. Camera
  placement and player/hand coverage remain target-hardware calibration work.

Physical chip recognition is not a Core ledger authority. Any future table
camera chip observation may provide operator evidence only; audited digital
ledger events remain authoritative.
