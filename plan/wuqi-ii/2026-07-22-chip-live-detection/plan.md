# Chip live detection

- Status: YOLO + OCR laptop-camera implementation complete; operator live-value test pending.
- Objective: run the completed YOLO11n single-class chip-localization model on a local Windows camera by default, with live boxes, confidence, visible count and bounded diagnostics; retain the Raspberry Pi MJPEG input as an explicit optional source.
- Owned paths:
  - `chip_recognition_workspace/live_chip_yolo11.py`
  - `chip_recognition_workspace/chip_ocr.py`
  - `chip_recognition_workspace/requirements.txt`
  - this plan file
- Dirty/read-only paths:
  - all other files under `chip_recognition_workspace/` are existing untracked experiments and remain read-only;
  - `runs/chip_localization/yolo11n_public_target_v2/weights/best.pt` is the ignored trained input and must not be modified;
  - chip datasets, card assets and unrelated user changes remain read-only.
- External dependencies: project `.venv`, Ultralytics 8.4.104, Torch 2.13.0+cu130, OpenCV 5.0.0, RapidOCR 3.9.2, ONNX Runtime 1.27.0 and a Windows DirectShow/MSMF camera; the MJPEG endpoint is optional. RapidOCR assets are verified local package files and runtime downloads are not used.
- Validation: compile/help checks, verify the checkpoint class map is the expected single `poker_chip` class, run bounded blank-image inference, and perform a bounded local-camera smoke without saving frames.
- Physical-motion status: vision-only development pilot; it does not mutate the ledger, connect to robot control, save frames or authorize physical motion.
- Commit intent: keep the chip workspace and training outputs uncommitted unless the user explicitly requests publication.
- Validation outcome: Python compilation and CLI help passed; the completed
  checkpoint loaded on CUDA with class map `{0: "poker_chip"}` and produced zero
  detections on a synthetic blank frame. A bounded attempt against
  `http://100.80.46.54:5000/video_feed` timed out while opening the stream, so
  target-camera frames were not available for the live smoke. Camera failures
  are reported as bounded `camera_error` output and do not touch game state.
- Local-camera update: DirectShow camera index `0` was unavailable. Index `1`
  successfully delivered three consecutive 1280x720 frames at a negotiated
  30 FPS with zero missing reads; the first CUDA inference took about 3 seconds
  while subsequent inference took 14-16 ms. The default is therefore index `1`
  and an explicit synthetic warm-up now occurs before opening the camera.
- DroidCam update: Windows enumerates `DroidCam Video` before the USB webcam.
  DroidCam camera index `0` successfully delivered three 1280x720 frames at
  30 FPS with zero missing reads through MSMF. The development runner now
  defaults to index `0` with backend `msmf`; the USB camera remains available
  through `--camera-index 1 --backend dshow`.
- OCR extension: each YOLO chip crop is enlarged and read by RapidOCR, with a
  thresholded fallback. Only exact `1`, `5`, `10` or `20` results above the
  configured confidence are accepted; unrelated text remains unknown and does
  not contribute to the visible total. OCR runs asynchronously so the camera
  display and YOLO detection loop remain responsive.
- OCR validation: pinned packages installed without changing OpenCV 5.0;
  packaged OCR model files were present and validated locally. A synthetic `20`
  crop was accepted as denomination 20 at confidence `0.99998`. A 60-frame
  DroidCam smoke completed at about 18 effective FPS with zero missing reads
  and zero OCR errors, but no chip was visible during that bounded capture, so
  live denomination evidence still requires the operator to present a chip.
