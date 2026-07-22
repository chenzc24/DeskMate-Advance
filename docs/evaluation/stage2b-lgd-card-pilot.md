# Stage 2B LGD Card Recognition Pilot

Status: `development feasibility only / not admitted / target-camera evidence open`

## Purpose And Boundary

This pilot pins the public `sroot/lgd-cards-gen3` YOLO11s ONNX asset as a fast
52-class card-recognition baseline. It consumes one project-owned BGR image
already cropped to a fixed card-slot ROI. The model can suggest one rank+suit
identity, but only the project temporal promoter can emit the frozen
`CardObservation` contract after repeated consistent evidence.

No detection or low confidence becomes `unknown`. A strong conflicting class
also becomes `unknown`; multiple same-class corner-pip detections are accepted
as one raw identity. The adapter never selects a slot, advances a hand, changes
the ledger or controls hardware.

The detector's boxes are corner-level evidence, not card records. Two printed
corners of one physical card may therefore remain visible as two boxes, while
the fixed-slot adapter emits only one card identity for temporal confirmation.
Future multi-card operation must run one bounded ROI per physical table slot;
the promoter also rejects the same 52-card identity appearing in two slots.

## Pinned Asset

- Repository: `https://huggingface.co/sroot/lgd-cards-gen3`
- Revision: `b2e9e89cc0138a70fc3ac5661922f99b4e3ae135`
- ONNX bytes: `38233687`
- ONNX SHA-256:
  `8b767cdfed2c8e954a9134013ac3d2f2c53be048768d559675be01277a8a8fd1`
- Class sidecar bytes: `316`
- Class sidecar SHA-256:
  `8a2d7e9dacf245aca5ef5a402cb404def919e9994e9142644d80c6d6248ee038`
- License: `AGPL-3.0`; running the ONNX with OpenCV does not change the weight
  license. Distribution obligations require review before any release.
- Local runtime: OpenCV DNN `5.0.0`, CPU, output shape `1x56x8400`.

The upstream model card reports recall `0.847` and precision proxy `0.771` on
a private 225-frame proof-of-concept holdout. Those self-reported numbers are
not Poker Dealer target-camera, held-out-deck or Gate 2B evidence.

## Offline Use

The model and class mapping must already exist under the ignored
`models/assets/card_recognition/lgd-cards-gen3/` directory. Runtime downloads
are prohibited and both files are size/hash verified before model load.

Run a single local image without temporal promotion:

```powershell
.\.venv\Scripts\python.exe scripts\perception\smoke_card_model.py `
  --image C:\absolute\path\to\card.jpg
```

To exercise the three-frame project confirmation boundary on the same still
image:

```powershell
.\.venv\Scripts\python.exe scripts\perception\smoke_card_model.py `
  --image C:\absolute\path\to\card.jpg `
  --slot board_flop_1 `
  --repeat 3
```

`--repeat` is only a deterministic still-image smoke mechanism. It is not live
temporal evidence and cannot support a Gate claim.

## Laptop Camera Use

The live pilot reads one camera, crops the central fixed ROI, runs the same
offline ONNX adapter and applies the project temporal promoter. It displays
only observations and never writes frames, mutates game state or connects to
the robot. Place one face-up card fully inside the green ROI, keep it still for
at least three frames, and press `Q` or `Esc` to stop:

```powershell
.\.venv\Scripts\python.exe scripts\perception\live_card_pilot.py
```

This laptop currently exposes the working DirectShow camera as device `1`,
which is the pilot config default. Override camera selection when required:

```powershell
.\.venv\Scripts\python.exe scripts\perception\live_card_pilot.py `
  --index 0 `
  --backend msmf
```

For a bounded, non-visual pipeline check, add `--headless --max-frames 20`.
The default live session is capped at 300 seconds even if no exit key is
pressed.

## Raspberry Pi MJPEG Use

The same pilot can consume the existing Raspberry Pi HTTP(S) MJPEG stream
through the project camera boundary. The stream URL and local camera index are
mutually exclusive; network streams use OpenCV FFmpeg with bounded open/read
timeouts, latest-frame buffering and reconnect handling:

```powershell
.\.venv\Scripts\python.exe scripts\perception\live_card_pilot.py `
  --stream-url http://100.80.46.54:5000/video_feed `
  --max-seconds 300
```

For a short non-visual connectivity and recognition smoke:

```powershell
.\.venv\Scripts\python.exe scripts\perception\live_card_pilot.py `
  --stream-url http://100.80.46.54:5000/video_feed `
  --headless `
  --max-frames 20 `
  --emit-all
```

The endpoint is used only at runtime and is not echoed in the summary. The
central normalized ROI remains a development fixture until target-table slot
geometry is measured; receiving Raspberry Pi frames does not close Gate 2B or
authorize robot motion.

## Local Validation

- Asset and class hashes verified; all 52 codes map uniquely into the project
  `Rank` and `Suit` enums.
- Synthetic blank image: `unknown`, `no_detection`, no guessed card.
- Public A-spades PNG pipeline smoke: predicted `A/spades` at
  `0.8228160738945007`; the model returned two same-identity corner detections.
- Three simulated stable observations progressed through
  `face_up_unconfirmed`, `face_up_unconfirmed`, `confirmed` and the final JSON
  validated against `card_observation.schema.json`.
- Low confidence, conflicting identities, duplicate confirmed identities and
  non-monotonic timestamps are rejected by targeted tests.
- A bounded DirectShow camera smoke at device `1` read 20/20 frames at
  1280x720 with no persistence. With no card deliberately presented, all 20
  observations were `unknown/no_detection`; model inference averaged 59.44 ms
  with P95 66.83 ms on this laptop CPU.
- A bounded Raspberry Pi MJPEG smoke read 20/20 frames at 640x480 with a
  reported 25 FPS and zero missing reads. With no card in the central fixture
  ROI, all 20 observations remained `unknown/no_detection`; inference averaged
  83.28 ms with P95 90.00 ms. No frames were saved and no robot-control
  endpoint was contacted.

The public sample and empty live-camera run are only functional smoke inputs.
No target-camera frame was saved and no private media, dataset snapshot or
accuracy matrix was created.

## Open Evidence And Next Decision

Before considering fine-tuning or model admission, test representative local
images from the intended camera and physical decks. Record per-rank/per-suit
failures and unknown behaviour. This detector remains a convenient baseline,
not the master plan's preferred final fixed-ROI dual-head classifier.

Gate 2B remains open because there is no target-camera thirteen-slot/four-
orientation replay, held-out physical deck/session split, calibrated threshold,
per-rank/per-suit report, long replay or offline deployment endurance result.
No camera image was persisted, and no robot connection or physical motion
occurred.
