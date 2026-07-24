# Android AudioRelay Runtime Integration

## Outcome

Connect the already-tested Android AudioRelay link to the formal
`scripts/runtime/run_hand.py` entry point:

- receive Android microphone audio through AudioRelay's Windows virtual
  microphone at its native sample rate, continuously resample to Vosk's
  required 16 kHz without changing game authority;
- send committed-event announcements through AudioRelay's Windows virtual
  speakers;
- load a versioned `en-US` announcement catalog instead of relying on
  code-only prompt strings, and select an English Windows voice when available;
- suppress and flush microphone evidence while announcements are audible so
  dealer speech cannot become player-action evidence;
- retain the existing local-microphone and console/no-announcer fallbacks.

## Owned Paths

- `configs/runtime/laptop_audiorelay.json`
- `configs/runtime/robot_camera_audiorelay.json`
- `configs/runtime/announcements_en.json`
- `docs/architecture/audiorelay-runtime.md`
- `scripts/runtime/run_hand.py`
- `src/poker_dealer/runtime/announcer.py`
- `src/poker_dealer/runtime/audio_input.py`
- `src/poker_dealer/runtime/live_perception.py`
- `src/poker_dealer/runtime/profile.py`
- `src/poker_dealer/perception/identity/__init__.py`
- `src/poker_dealer/perception/identity/gallery.py`
- `src/poker_dealer/perception/identity/opencv_adapter.py`
- `tests/runtime/test_audio_input.py`
- `tests/runtime/test_announcer.py`
- `tests/runtime/test_run_hand_cli.py`
- `tests/runtime/test_registration_ui.py`
- `tests/runtime/test_registration_runtime.py`
- `tests/perception/identity/test_session_face_identity.py`

The formal registration component is also exposed through a
`registration-smoke` mode. It combines the Raspberry Pi MJPEG camera with the
AudioRelay microphone and English announcements, then stops after freezing the
four-player roster without creating a game hand or emitting a dealer command.
Registration evidence records transition-only camera health events for missing
frames, recovery duration, background reconnect epochs, long frame gaps and
terminal disconnects. These records contain no frame pixels.

The registration preview uses a fixed 1280x720 operator dashboard: an isolated
aspect-preserving camera panel with model-derived face boxes, a stage-aware
sidebar for face capture/TTS playback/voice phrases, four-role progress and a
fixed keyboard legend. Its resizable keep-ratio window starts within the
available desktop area and scales cleanly when maximized. Operator instructions
and announcements name the physical `E` key instead of the abstract confirm
intent.

Before the operator presses `E`, a throttled detector-only preview draws all
current face boxes and gives one-face guidance. This preview computes no face
embedding; consent-gated enrollment and memory-only embeddings still begin only
after the explicit `E` control.

A duplicate face at a later role is a retryable registration rejection rather
than a process error. The already-registered role remains intact, the attempted
role remains focused, candidate references are discarded, and UI/TTS instruct
the operator to change players and press `E` again. The audit event contains
only roles, seats, scalar similarity/threshold metadata and privacy flags.

Additional narrowly scoped runtime exports may be updated if the new types need
to be imported from `poker_dealer.runtime`.

## Dirty Read-Only Paths

The following pre-existing user changes are not owned by this target and must
be preserved:

- `README.md`
- `pyproject.toml`
- diagnostics changes already present in `scripts/runtime/run_hand.py`
- `.gitattributes`, `models/manifest.yaml`, the card-finetune target plan and
  any later unrelated user changes
- diagnostics exports in `src/poker_dealer/runtime/__init__.py`
- diagnostics instrumentation in `src/poker_dealer/runtime/hand_loop.py`
- all pre-existing untracked diagnostics, data, training and plan paths shown
  by `git status`

Where this target must touch `scripts/runtime/run_hand.py`, changes are applied
on top of the diagnostics work without removing or rewriting it.

## External Dependencies

- AudioRelay Desktop `0.27.5`, already installed locally.
- AudioRelay Android `0.26.1`, already connected over the local Wi-Fi.
- Windows virtual endpoints:
  `Virtual Mic (AudioRelay Wave)` and
  `Virtual Speakers (Virtual Speakers for AudioRelay)`.
- Existing `speech-pilot` optional dependencies (`vosk`, `sounddevice`).

No runtime downloads, credentials, audio recordings or speaker embeddings are
added to the repository.

## Validation

- Parse the new runtime JSON profile.
- Parse and validate the English announcement catalog, required keys and
  placeholders.
- Unit-test announcement lifecycle, priority queue behavior and playback
  suppression/tail guard.
- Unit-test that suppressed or stale AudioRelay PCM never reaches Vosk.
- Verify the AudioRelay virtual microphone supports the configured
  native-rate mono/int16 input and the resampler emits 16 kHz PCM.
- Run formal `live-preflight` with the AudioRelay profile.
- Run targeted runtime tests and the practical full pytest suite.
- Run `git diff --check` and scoped `git status --short --branch`.

Hardware/Wi-Fi soak testing remains an operator-run follow-up and is not
represented as completed by software tests.

Completed software validation:

- AudioRelay profile parsed through the named `laptop_audiorelay` selector.
- The local `Virtual Mic (AudioRelay Wave)` accepted native 44.1 kHz mono
  int16 input; a real two-second callback capture produced exactly 32,000
  resampled 16 kHz frames without PortAudio status errors.
- Formal `live-preflight` passed with immutable perception assets and runtime
  downloads/audio persistence disabled.
- Announcement, committed-event mapping, microphone suppression and CLI tests
  passed.
- English catalog `poker-dealer-en-us@1.2.0` parsed with 47 validated entries;
  installed `Microsoft Zira Desktop` is the first voice preference.
- The practical full pytest suite passed with 340 tests.

## Physical-Motion Status

No physical motion is authorized. The laptop profile continues to use the
simulated dealer. Voice and announcements remain evidence/feedback adapters and
never write motor commands, ledger state or game transitions.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.
