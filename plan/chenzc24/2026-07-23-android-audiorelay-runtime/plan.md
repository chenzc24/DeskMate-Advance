# Android AudioRelay Runtime Integration

## Outcome

Connect the already-tested Android AudioRelay link to the formal
`scripts/runtime/run_hand.py` entry point:

- receive Android microphone audio through AudioRelay's Windows virtual
  microphone without changing Vosk or game authority;
- send committed-event announcements through AudioRelay's Windows virtual
  speakers;
- suppress and flush microphone evidence while announcements are audible so
  dealer speech cannot become player-action evidence;
- retain the existing local-microphone and console/no-announcer fallbacks.

## Owned Paths

- `configs/runtime/laptop_audiorelay.json`
- `docs/architecture/audiorelay-runtime.md`
- `scripts/runtime/run_hand.py`
- `src/poker_dealer/runtime/announcer.py`
- `src/poker_dealer/runtime/live_perception.py`
- `tests/runtime/test_announcer.py`
- `tests/runtime/test_run_hand_cli.py`

Additional narrowly scoped runtime exports may be updated if the new types need
to be imported from `poker_dealer.runtime`.

## Dirty Read-Only Paths

The following pre-existing user changes are not owned by this target and must
be preserved:

- `README.md`
- `pyproject.toml`
- diagnostics changes already present in `scripts/runtime/run_hand.py`
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
- Unit-test announcement lifecycle, priority queue behavior and playback
  suppression/tail guard.
- Unit-test that suppressed or stale AudioRelay PCM never reaches Vosk.
- Verify the AudioRelay virtual microphone supports the configured
  16 kHz/mono/int16 input.
- Run formal `live-preflight` with the AudioRelay profile.
- Run targeted runtime tests and the practical full pytest suite.
- Run `git diff --check` and scoped `git status --short --branch`.

Hardware/Wi-Fi soak testing remains an operator-run follow-up and is not
represented as completed by software tests.

## Physical-Motion Status

No physical motion is authorized. The laptop profile continues to use the
simulated dealer. Voice and announcements remain evidence/feedback adapters and
never write motor commands, ledger state or game transitions.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.
