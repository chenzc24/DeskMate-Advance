# Mobile Web Console Target Plan

## Outcome

Provide a Windows-hosted, phone-friendly registration console that mirrors the
shared Raspberry Pi camera frame and registration state, draws the same
pre-capture face detections, and submits semantic E/S/X/Q operator intents from
an Android touch screen. Audio remains on the existing AudioRelay path, with
real input-level visibility and browser-local mirroring of the same rendered
English prompts.

The mobile presentation is a fixed single-viewport dashboard optimized for a
roughly 6.73-inch phone in landscape, with a compact portrait fallback. Voice
enrollment presents one word per sample in the fixed order `CHECK`, `CALL`,
`RAISE`.

## Owned Paths

- `pyproject.toml`
- `scripts/runtime/run_hand.py`
- `configs/contracts/network_endpoints.schema.json`
- `configs/contracts/runtime_profile.schema.json`
- `configs/runtime/network_endpoints.json`
- `configs/runtime/robot_camera.json`
- `configs/runtime/robot_camera_audiorelay.json`
- `configs/runtime/robot_hardware.json`
- `src/poker_dealer/domain/controls.py`
- `src/poker_dealer/runtime/__init__.py`
- `src/poker_dealer/runtime/live_perception.py`
- `src/poker_dealer/runtime/mobile_web_console.py`
- `src/poker_dealer/runtime/mobile_web_assets/`
- `src/poker_dealer/runtime/network.py`
- `tests/domain/test_roles_and_controls.py`
- `tests/runtime/test_mobile_web_console.py`
- `tests/runtime/test_network_endpoints.py`
- `tests/runtime/test_registration_runtime.py`
- `tests/runtime/test_runtime_profiles.py`
- `tests/runtime/test_run_hand_cli.py`
- `docs/architecture/mobile-web-console.md`
- `docs/architecture/network-endpoints.md`

## Dirty Read-Only Paths

Preserve all unrelated existing changes, especially the AudioRelay profile,
announcement catalog, identity enrollment, diagnostics, and registration UI
work already present in the worktree. Only make narrowly required integration
edits where owned paths overlap that work.

## External Dependencies

- `aiohttp` is an optional runtime dependency for the local HTTP and WebSocket
  service.
- Android Chrome reaches the configured Windows endpoint through trusted local
  Wi-Fi or an optional private Tailnet.
- Changeable mobile-web and robot-camera network values are resolved from the
  single validated `configs/runtime/network_endpoints.json` file.
- AudioRelay continues to provide the Android microphone and speaker channels;
  the web page does not request browser microphone access.

## Validation

- Unit-test bounded/idempotent remote command handling, stale-view rejection,
  controller ownership, state serialization, prompt mirroring, actual audio
  level reporting, and composite control/event adapters.
- Exercise the local HTTP page, MJPEG endpoint, and WebSocket handshake.
- Verify the responsive page at 852x393 and 800x360 landscape CSS-pixel
  viewports, plus the 393x852 portrait fallback, with viewport-sized scroll
  bounds and every control visible.
- Run targeted runtime/domain/CLI tests, then the practical full test suite.
- Parse affected JSON configurations, run `git diff --check`, and inspect scoped
  `git status --short --branch`.
- Verify named robot profiles resolve their camera URL from the shared network
  configuration and CLI network overrides remain explicit.
- Confirm no camera frames, face images, embeddings, or microphone samples are
  written by the web console.

## Physical Motion Status

No physical dealer adapter is opened and no motion is authorized. Web controls
remain observations consumed by the deterministic registration/runtime
boundaries.

## Commit Intent

Do not commit, push, create a branch, or publish a public deployment unless the
user explicitly requests it.
