# Live chip denomination template integration

- Status: replacing the first live integration with the requested
  track-owned best-frame pipeline; target-camera retest pending.
- Objective: attach the fixed-design `1`/`5`/`10`/`20` template matcher to
  YOLO11 live chip boxes, display only accepted denomination evidence and
  compute a non-authoritative visible total. Rejected or stale evidence remains
  unknown.
- Owned paths:
  - `chip_recognition_workspace/live_chip_yolo11.py`
  - `chip_recognition_workspace/chip_live_value.py`
  - `chip_recognition_workspace/chip_value_tracker.py`
  - `chip_recognition_workspace/chip_best_frame.py`
  - `chip_recognition_workspace/test_chip_live_value.py`
  - `chip_recognition_workspace/test_chip_value_tracker.py`
  - `chip_recognition_workspace/test_chip_best_frame.py`
  - this plan file
- Dirty/read-only paths:
  - `data/raw/chip_templates/`, `data/chips/` and all card/face data are
    read-only;
  - `data/work/chips/2026-07-23-template-matching/` is the ignored derived
    template library and remains externally generated;
  - the existing YOLO localization checkpoint and unrelated dirty files remain
    read-only.
- External dependencies: project `.venv`, existing YOLO11 localization
  checkpoint, OpenCV, NumPy, Torch/Ultralytics and the already-built offline
  template library. No OCR or runtime download is used.
- Validation: compile/help checks; focused value-adapter and template tests;
  bounded recorded-image recognition; practical full suite; live Raspberry Pi
  validation is explicitly deferred because the user requested no launch after
  modification.
- Physical-motion status: perception display only. It does not mutate the
  digital ledger, advance game state, connect to robot control, send GPIO or
  serial commands, or authorize physical motion.
- Commit intent: no commit, push, branch or PR unless the user explicitly asks.
- Current change target:
  - run YOLO localization first and assign a persistent `track_id` before
    denomination work;
  - retain a bounded best raw frame for each track using projected size,
    sharpness, detector confidence, box aspect and glare;
  - measure the fixed-design outer-ring colour on the original elliptical chip
    before perspective normalization;
  - rectify only the selected frame, match the central printed number, fuse
    independent colour/number evidence conservatively, then feed only distinct
    source-frame observations into temporal confirmation;
  - retain unknown/rejection output for weak, conflicting, too-small or overly
    flat evidence. No result is authoritative game or ledger state.
- Outcome:
  - the live runner no longer initializes or calls RapidOCR;
  - accepted YOLO boxes are rectified and classified asynchronously every five
    frames, while stale, failed or low-confidence evidence displays `?`;
  - each detection exposes denomination, fused template score, margin, ellipse
    quality and rejection reason; the visible total is explicitly
    non-authoritative and changes neither game state nor ledger;
  - four recorded smoke frames covering `1`, `5`, `10` and `20` were all
    localized, rectified and classified correctly, with match scores
    `0.673068`, `0.944471`, `0.964503` and `0.615437`;
  - focused tests pass (`26 passed`) and compile/help checks pass;
  - the practical suite remains `292 passed, 4 skipped, 4 failed`; the four
    failures are the pre-existing missing YuNet face-identity asset outside the
    owned paths.
  - follow-up hardening adds a fixed-design red/yellow-ring correction for
    accepted `1` candidates and rejects `5` candidates without that ring;
  - projected chip faces with a minor axis below 42 px return `too_far`, and
    faces with ellipse aspect ratio below 0.38 return `too_flat`;
  - cached value evidence is limited to eight frames and requires IoU 0.55
    before attaching to a current YOLO box, reducing stale cross-chip values;
  - on all 66 labelled development captures the hardened gate accepted 47 and
    all 47 were correct; nine were rejected as `too_flat`, nine by the template
    threshold and one by ellipse quality. This is a safety/precision gate, not
    evidence that information lost in the 640x480 source can be recovered.
  - the revised warm-colour rule covers red/orange/yellow HSV shifts and Lab
    redness; an ambiguous `1`/`5` colour region now returns unknown;
  - each YOLO box receives an IoU-associated track ID; a denomination is shown
    only after five of seven distinct value batches agree, reused cache frames
    do not add votes, and switching requires three consecutive alternative
    results above score 0.70;
  - two consecutive `too_far` or `too_flat` batches temporarily hide a
    confirmed value without forgetting the track;
  - rectification crops are capped at 224 px before GrabCut; across the 66
    development captures per-chip P50/P95 value latency is 80.771/135.002 ms;
  - the hardened development result remains 47 accepted, 47 correct, with no
    accepted `1`/`5` confusion. Target-camera retest is still required because
    the current Raspberry Pi stream and lighting are not an independent split.
  - requested v2 ordering is now implemented as `YOLO -> track_id -> bounded
    best frame -> raw elliptical outer-ring colour -> perspective correction
    -> central digit shape -> multi-frame confirmation`;
  - v2 development replay on all 66 labelled captures accepts 48 and all 48
    are correct: `1` accepts 11/18, `5` accepts 11/15, `10` accepts 12/16 and
    `20` accepts 14/17. No accepted `1`/`5` confusion occurs; rejected evidence
    stays unknown;
  - a 30-frame Raspberry Pi MJPEG smoke run completed at 24.61 FPS with no
    missing reads or value-engine exceptions. No chip was in that view, so this
    verifies connectivity/control flow only, not target-camera denomination
    quality;
  - v2 focused tests pass (`34 passed`);
    the practical suite remains `292 passed, 4 skipped, 4 failed`, with the
    same four unrelated failures caused by the missing YuNet face asset.
