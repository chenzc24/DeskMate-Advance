# Mobile Stereo Screen-Distance Preplan

## Outcome And Owned Paths

Define a provisional, test-gated architecture for estimating a person's
distance to the screen while DeskMate and its two cameras move. Keep the
proposal explicitly unfrozen until camera geometry and recorded evidence pass
their gates.

Owned paths:

- `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md`
- `docs/plans/PART_A_ERGONOMICS_PLAN.md`
- `plan/chenzc24/2026-07-20-mobile-stereo-distance-preplan/plan.md`

## Dirty Paths Left Read-Only

All existing source, configuration, tests, evaluation documents, data,
artifacts, and other target plans remain read-only. In particular, this target
does not change the current single-camera laptop probe or its development
distance rule.

## External Dependencies

- Robotics confirmation of camera models, rigid baseline, mounting pose,
  synchronization support, resolution, FPS, color format and timestamps.
- A mounting geometry in which the person's face and at least three
  non-collinear screen reference points remain in both cameras' common field of
  view over the intended DeskMate motion envelope.
- A versioned stereo calibration artifact and a screen-plane reference method,
  provisionally bezel fiducials or equally testable screen features.
- Human decisions for the face/eye reference point, operational range,
  distance-error tolerance, valid-result rate and event thresholds.
- Consented target-camera recordings and independent physical ground-truth
  distance measurements; private media remains outside Git.

## Validation And Physical Motion

Validate documentation consistency, then use static recorded pairs before any
moving test. Later implementation must test synchronization, calibration,
triangulation, screen-plane fit, distance error, stale/unknown behavior,
throughput and P95 latency. Moving-camera validation must compare against an
independent distance reference and include blur, occlusion and lost-marker
cases.

This documentation target involves no physical robot motion. Any later powered
motion requires an operator, clear area, low speed, distance limits, collision
protection, watchdog and emergency/manual stop.

## Commit Intent

Do not commit, push, create a branch or open a pull request unless the user
explicitly requests it.
