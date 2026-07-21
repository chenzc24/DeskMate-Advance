# Stage 2A Multimodal Action Pilot

## Outcome And Owned Paths

Replace the unreliable `Closed_Fist -> bet` development mapping with the
already-supported `Victory -> bet` gesture, and add a bounded offline English
command pilot for the same five poker action semantics plus `cancel/confirm` controls.
Gesture and speech remain independent evidence sources. A conservative fusion
helper may agree, pass through one source, or reject a conflict, but only the
game engine can accept a legal action and advance the acting seat.

Owned paths are the two action pilot configs, `src/poker_dealer/perception/actions/`,
`scripts/perception/`, `tests/perception/actions/`, action pilot documentation,
the two development entries in `models/manifest.yaml`, `pyproject.toml`, and
this plan. The only Stage 1 engine edit routes the already-frozen
`voice_adapter` audit source from the model version; promotion thresholds and
state transitions stay unchanged. Downloaded ASR assets stay ignored under
`models/assets/`.

## Dirty Paths Left Read-Only

All archived DeskMate changes, Stage 0/1 contracts and engine internals, card
perception, robotics, hardware protocols, unrelated plans and private media
remain read-only. Apart from the narrow existing-enum audit-source routing
above, Stage 1 stays read-only. The frozen action schema is reused without
adding transcript or vendor-specific fields.

## External Dependencies

- Vosk 0.3.45 Windows runtime and the Apache-2.0
  `vosk-model-small-en-us-0.15` development asset.
- SoundDevice 0.5.5 and the Laptop microphone.
- The state-owned listening window is the only Core seat attribution. The
  omnidirectional Laptop microphone does not prove who spoke.

## Validation And Physical Motion

Validate both configs, immutable asset hashes, all five command mappings,
control/rejection/cooldown behavior, fusion agreement/conflict, schema and game
authority, silent offline decoding, microphone availability, bounded live
audio capture, gesture UI startup, targeted tests and the practical full suite.
No frames or audio are saved. No robot connection or physical motion is
authorized.

## Commit Intent

Do not commit or push unless the user explicitly requests it.

## Completed Validation

- `Closed_Fist -> bet` was replaced by `Victory -> bet`; the five-action
  mapping and unknown/release/cooldown tests pass.
- The user selected English recognition. The final development runtime uses
  `vosk-model-small-en-us-0.15`, archive SHA-256
  `30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498`
  and extracted tree SHA-256
  `57929637421baa20ff74ffb194f48e7c4a5bd0c09eac1c79a3c305ddf32db038`.
- All seven words (`fold/check/call/bet/raise/cancel/confirm`) initialize in
  the constrained grammar without missing-vocabulary warnings.
- Targeted action/fusion tests: 23 passed. Practical full suite: 116 passed.
- Laptop microphone device 1 captured 16 bounded audio blocks over 5.50
  seconds with zero dropped blocks and zero saved bytes. No person spoke during
  the smoke test, so the live seven-word matrix remains open.
- No robot connection or physical motion occurred.
