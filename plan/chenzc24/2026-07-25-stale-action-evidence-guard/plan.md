# Stale Action Evidence Guard

## Outcome

Prevent a speech or gesture candidate captured before the current actor
binding became valid from terminating the live process. Reject the stale
candidate as evidence, keep the authoritative acting seat and ledger
unchanged, and continue waiting for fresh evidence.

## Owned Paths

- `src/poker_dealer/runtime/hand_loop.py`
- Scoped hand-loop regression tests.

## Dirty Read-Only Paths

- Preserve the in-progress Flop batch/timeout work and phone-front profile.

## External Dependencies

- None.

## Validation

- Reproduced an otherwise context-matching action whose observation timestamp
  precedes `binding.verified_at_ns`.
- The regression proves it is recorded as
  `player_action_observation_rejected/actor_binding_time_mismatch`, ignored,
  and followed by a normally settled hand.
- Targeted hand-loop/actor-binding tests: 10 passed.
- Practical full Python suite: 428 passed.

## Physical-Motion Status

No physical motion is authorized.

## Commit Intent

Do not commit or push.
