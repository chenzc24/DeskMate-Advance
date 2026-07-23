# AudioRelay live runtime

This is the development-only Android audio bridge for the Poker Dealer live
runtime. It does not change the Core game authority: speech remains attributed
evidence, and only the deterministic game engine can accept a legal action.

## Audio routes

```text
Poker Dealer committed event
  -> Windows System.Speech
  -> Windows default playback device / AudioRelay stream
  -> Android speaker

Android microphone
  -> AudioRelay
  -> Virtual Mic (AudioRelay Wave)
  -> LivePerceptionSession
  -> Vosk + enrolled-speaker verification + confirmation
  -> deterministic game engine legality checks
```

The runtime uses half-duplex protection. From the moment a Windows announcement
is queued until 350 ms after it completes, microphone blocks are discarded and
the recognizer window is reset. This prevents the dealer announcement returning
through the Android microphone and becoming player evidence. Adjust the tail
only after an echo test.

## One-time Windows and Android setup

1. Put the PC and Android phone on the same ordinary Wi-Fi network.
2. Start the AudioRelay link in both directions and confirm that the Android app
   is connected to the PC.
3. In Windows/AudioRelay, route computer playback to the Android speaker. The
   Windows TTS adapter uses the current default playback route.
4. Enable microphone forwarding from Android and keep
   `Virtual Mic (AudioRelay Wave)` available as a Windows recording device.
5. Disable Android battery optimization for AudioRelay, keep the phone powered,
   and prevent Wi-Fi sleep during a session.

The tested local setup used PC `10.241.149.250`, Android `10.241.149.7`,
AudioRelay PC `0.27.5`, and Android server `0.26.1`. Addresses and device indices
are not stored in the profile because DHCP and Windows audio enumeration can
change them.

## Preflight

With the AudioRelay connection active:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile configs/runtime/laptop_audiorelay.json `
  --mode live-preflight
```

If the virtual microphone name does not resolve on a particular Windows host,
list devices with `python -m sounddevice` and temporarily override it:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile configs/runtime/laptop_audiorelay.json `
  --mode live-preflight `
  --speech-device 38
```

Index `38` was valid only on the tested PC; prefer the profile's stable device
name.

## Live development run

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile configs/runtime/laptop_audiorelay.json `
  --mode live `
  --button seat_a `
  --consent-confirmed `
  --development-operator-face-down `
  --announcer windows `
  --diagnostics
```

`--announcer console` verifies event-to-text wiring without audible output.
`--announcer none` is the default. `--announcement-tail-guard-ms 350` controls
the post-playback microphone guard.

Face and speaker enrollment require explicit participant consent. Audio and
speaker embeddings are not written by this bridge. The current live command is
a non-Gate development run because face-down card orientation still uses the
operator confirmation fallback, and it never authorizes unattended physical
motion.

## Acceptance check

Before a real table session, verify all of the following:

- A test Windows announcement is audible on the Android phone.
- Speech into the Android phone reaches the configured virtual microphone.
- Dealer speech during and immediately after playback creates no action
  candidate.
- Each enrolled player's command is attributed to that player and still needs
  the configured confirmation/legal-action checks.
- Disconnecting AudioRelay causes missing/unknown speech evidence, never a
  guessed action or ledger change.
