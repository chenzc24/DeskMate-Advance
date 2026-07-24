# Mobile Web Console

## Scope

The mobile console is the full-session operator display, not a registration-only
page. It mirrors the same authoritative state used by the Windows runtime:

- face and memory-only voice registration;
- hole-card delivery progress;
- pre-flop, flop, turn and river acting seat, legal actions, commitments and pot;
- state-bound board cards and logical target;
- showdown, awards and final stacks;
- pause/recovery, table clearance, rebuy selection, next hand and session end.

It does not provide fold/check/call/bet/raise buttons. Player actions still come
from the current state-selected player's English voice and/or hand gesture.
Touch controls are semantic registration, speech-confirm/cancel and operator
recovery/session controls only.

```text
Camera -> shared FramePacket -> perception/runtime -> deterministic engine
             |                                      |
             +-- memory-only JPEG ------------------+--> mobile state view

mobile semantic control -> bounded/versioned queue -> owning runtime/controller
committed event -> English announcement -> Windows TTS + mirrored phone TTS
```

## Start

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web-console]"

.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile robot_camera_audiorelay `
  --mode live `
  --button seat_a `
  --consent-confirmed `
  --announcer windows `
  --web-console `
  --headless
```

The default endpoint is `http://127.0.0.1:8765/`. Omit `--headless` to retain
the simultaneous OpenCV/keyboard fallback.

For private phone access, keep loopback binding and use Tailnet-only serving:

```powershell
tailscale serve --bg 8765
tailscale serve status
```

Do not use a public Funnel. Direct `--web-host 0.0.0.0` binding requires a
restrictive Windows firewall and Tailnet ACL.

## State and control semantics

- The first browser is controller; later browsers are read-only viewers.
- Every command carries a unique ID and the rendered `view_version`; stale
  commands fail closed and the last 256 results are deduplicated.
- The control queue is bounded. `queued` means delivered to the runtime, not
  accepted by game logic.
- Registration exposes capture, start and clear.
- During `AWAITING_ACTION`, the phone exposes confirm/cancel for a pending spoken
  candidate. It never creates a poker action itself.
- Recovery exposes retry, void and conflict-slot selection/reconciliation.
- Between hands it exposes table-clear confirmation, low-stack selection/rebuy,
  next hand and session end.
- Destructive clear/void/exit controls require a confirmation dialog.
- Reconnect receives a complete current snapshot and never replays old commands.

## Announcements

The Windows announcer and phone prompt use the same rendered English catalog.
Only committed engine/runtime/session facts are announced. Registration, blind
posting, turn start, accepted action, street start, dealing, card failure,
showdown, awards, recovery, table clearance, next hand and session completion
are covered. The repeat button replays the latest rendered prompt; model
candidates are not announced as facts.

## Privacy and authority

Frames are JPEG-encoded in memory and never written by this service. Audio stays
on AudioRelay; the browser requests no microphone. Face and speaker embeddings
remain session-memory-only. The page cannot call the game reducer, change the
ledger, select the acting seat, emit a dealer command or authorize physical
motion.
