# Robotics Handoff Contract

## Ownership

Deep Learning/Runtime owns only semantic intent, command correlation, evidence
validation and game pause/advance decisions. Robotics owns all physical
implementation: transport/firmware, target calibration, kinematics, motors,
feeder, sensors, homing, interlock, E-stop, watchdog and recovery procedures.

Dependency direction remains:

```text
HandRuntime -> DealerCommand -> Robotics controller -> DealerAck -> HandRuntime
```

Robotics must never call the game reducer or ledger directly. Runtime must
never send angles, PWM, servo values, GPIO or unversioned transport bytes.

## Frozen semantic requests

- `home`
- `rotate_to(target_slot)`
- `dispense_one`
- `stop`
- `get_status`
- `reset_fault`

The nine logical targets are four player locations and five Board locations.
Their millimetre/angle realization belongs to Robotics calibration and is not
stored in game or model code.

## Required real ACK evidence

A terminal ACK must correlate to the exact command ID/type/target and include
protocol/device state version plus available sensor evidence:

- homed;
- at target;
- deck present;
- exactly one exit pulse for successful dispense;
- interlock closed;
- emergency stop inactive;
- explicit terminal status and error code/reason on failure.

Elapsed time alone cannot produce a successful ACK. Unknown, lost, malformed,
duplicate-with-conflict or mismatched ACKs never advance a hand.

## Robotics delivery required before enabling `robot_hardware`

1. Transport or SDK plus a frozen protocol version and framing/CRC rules.
2. Firmware version and command-deduplication behavior.
3. Nine-target calibration identifier and measured positioning distribution.
4. Sensor/error-code mapping and heartbeat/disconnect behavior.
5. Measured P95/P99 latency for timeout configuration.
6. Homing, interlock, E-stop, watchdog and jam/double-feed evidence.
7. Operator runbook and Stage 3 safety release.

The runtime-side real Adapter may be supplied by Robotics or implemented as a
thin jointly reviewed protocol translation. It cannot contain motion planning
or synthesize missing sensor evidence.

## Current status

`robot_hardware` is declared with `adapter=real`, `physical_motion=true` and
`enabled=false`. Its `UnavailableDealerAdapter` rejects `open()` and every
command. Laptop, robot-camera and Replay paths use an explicitly non-physical
`SimulatedDealerAdapter`; they cannot silently change this hardware status.
