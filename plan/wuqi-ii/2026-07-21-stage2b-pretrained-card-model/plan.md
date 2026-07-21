# Stage 2B Pretrained Card Model Integration

## Outcome And Owned Paths

Integrate the pinned `sroot/lgd-cards-gen3` 52-class ONNX detector as a
development-only fixed-ROI card-recognition baseline with a bounded,
non-recording Laptop camera UI. The local adapter must verify the immutable
asset, run fully offline, reject missing/low-confidence or conflicting evidence
as `unknown`, and emit the existing project-owned `CardObservation` without
advancing game state.

Owned paths are `configs/perception/cards_lgd_pilot.json`,
`src/poker_dealer/perception/cards/`, `scripts/perception/*card*`,
`tests/perception/cards/`, `docs/evaluation/stage2b-*`, the development entry in
`models/manifest.yaml`, and this plan. The ONNX asset stays ignored under
`models/assets/card_recognition/lgd-cards-gen3/`.

## Dirty Paths Left Read-Only

Stage 0/1 contracts and game logic, action and identity perception, Robotics,
table geometry, archived DeskMate content, private media, unrelated plans and
all existing model assets remain read-only. The shared model manifest is
updated only with the pinned development asset metadata required by repository
policy; no model is promoted to candidate or release.

## External Dependencies

- Hugging Face model `sroot/lgd-cards-gen3` pinned to revision
  `b2e9e89cc0138a70fc3ac5661922f99b4e3ae135`.
- `model.onnx` and `model.classes.json`, downloaded once during setup and
  verified by SHA-256 before every load; runtime downloads remain prohibited.
- AGPL-3.0 model weights inherited from Ultralytics YOLO11. The adapter uses
  the repository's existing OpenCV DNN runtime and records the license without
  claiming that ONNX changes the weight license.
- The model card reports a private 225-frame proof-of-concept holdout with
  recall `0.847` and precision proxy `0.771`; these are upstream development
  numbers, not Poker Dealer target-camera metrics.

## Validation And Physical Motion

Validate config parsing, pinned asset hashes, 52-class rank/suit mapping,
letterbox preprocessing, bounded YOLO output decoding, NMS/ambiguity rejection,
same-card corner deduplication, blank-image rejection, schema-compatible
`CardObservation`, deterministic local-image CLI output, fixed-ROI cropping,
missing/disconnected camera handling, bounded headless/live-camera execution
and practical full tests. Run
`git diff --check` and scoped `git status --short --branch`. A bounded Laptop
camera run is authorized only for non-recording feasibility evidence; no frame
or video persistence, robot connection or physical motion is authorized.

## Commit Intent

Do not commit, push or merge unless the user explicitly requests it after
reviewing the local test result.

## Completed Validation

- Pinned and verified the ignored ONNX asset at
  `8b767cdfed2c8e954a9134013ac3d2f2c53be048768d559675be01277a8a8fd1`
  and its 52-class sidecar at
  `8a2d7e9dacf245aca5ef5a402cb404def919e9994e9142644d80c6d6248ee038`.
- OpenCV DNN 5.0.0 loaded the model fully offline and produced the expected
  bounded `1x56x8400` output. A blank frame returned `unknown/no_detection`.
- A public A-spades functional sample returned `A/spades` at
  `0.8228160738945007`; three simulated stable frames produced a schema-valid
  `confirmed` observation. This is a pipeline smoke result, not an accuracy
  claim or target-camera evidence.
- Stage 2B card pilot tests: `9 passed`; JSON parsing, Python compilation and
  `git diff --check` passed.
- Practical full suite: `133 passed, 3 skipped, 2 failed`. Both failures are
  the pre-existing Stage 2A face-identity tests whose ignored YuNet/SFace
  assets are absent locally; no card-pilot test failed.
- A bounded non-recording DirectShow smoke on camera `1` read 20/20 frames at
  1280x720. The deliberately empty ROI produced 20 `unknown/no_detection`
  observations; inference mean was 59.44 ms and P95 was 66.83 ms. No frames
  were saved and no game state was mutated.
- No robot connection, physical motion, commit, push or merge occurs.
