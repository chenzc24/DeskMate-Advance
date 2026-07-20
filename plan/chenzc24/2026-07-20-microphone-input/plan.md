# Laptop microphone input target

- Outcome and owned paths: add a timestamped, bounded, synchronous microphone
  adapter for the laptop's Intel Smart Sound microphone array, an audio domain
  packet, a device probe, module documentation, and focused tests. Owned paths
  are `src/deskmate_advance/domain/audio.py`, the audio exports added to
  `src/deskmate_advance/domain/__init__.py`,
  `src/deskmate_advance/perception/audio/`, the audio exports added to
  `src/deskmate_advance/perception/__init__.py`,
  `scripts/runtime/probe_microphone.py`, `tests/perception/test_audio.py`, and
  this plan.
- Dirty paths left read-only: `.gitignore`, `pyproject.toml`, `configs/`,
  `docs/`, `models/`, `scripts/evaluation/`, the camera implementation and its
  tests, and all pre-existing `plan/` targets. The existing project dependency
  `sounddevice==0.5.5` is reused without editing dependency resolution.
- External dependencies: the Intel Smart Sound microphone array and Windows
  audio permissions, the installed `sounddevice`/PortAudio runtime, Python
  3.11+, and NumPy. DroidCam Audio is explicitly outside this target.
- Validation: fake-stream unit tests cover successful, overflowed, malformed,
  failed, and closed reads; the device probe does not capture audio; one live
  100 ms smoke read reports metadata and RMS only and never saves samples. No
  controller integration or physical robot motion is involved.
- Commit intent: do not commit, push, create a branch, or open a PR.
