"""High-volume deterministic safety replay for Part A observation gates."""

from __future__ import annotations

from collections import Counter
from typing import Any

from poker_dealer.domain import ActionEvidenceState, PlayerActionType, Seat
from poker_dealer.game import HandEngine, SimulatedActionPerception, state_to_dict


def run_action_safety_replay(no_action_events: int = 10_000) -> dict[str, Any]:
    if no_action_events <= 0:
        raise ValueError("no_action_events must be positive")
    engine = HandEngine.start("stage2a-safety-replay", Seat.A)
    simulator = SimulatedActionPerception()
    initial = state_to_dict(engine.snapshot())
    reasons: Counter[str] = Counter()
    accepted = 0

    def apply(observation: object) -> None:
        nonlocal accepted
        result = engine.apply_observation(observation)  # type: ignore[arg-type]
        reasons[result.reason] += 1
        accepted += int(result.accepted)

    for _ in range(no_action_events):
        apply(simulator.emit(engine.state, ActionEvidenceState.NO_ACTION))
    for evidence_state in (
        ActionEvidenceState.AMBIGUOUS,
        ActionEvidenceState.OCCLUDED,
        ActionEvidenceState.UNKNOWN,
        ActionEvidenceState.OUT_OF_ROI,
    ):
        apply(simulator.emit(engine.state, evidence_state))
    apply(
        simulator.emit(
            engine.state,
            ActionEvidenceState.CANDIDATE,
            action=PlayerActionType.CALL,
            focus_seat=Seat.A,
            confidence=0.99,
        )
    )
    apply(
        simulator.emit(
            engine.state,
            ActionEvidenceState.CANDIDATE,
            action=PlayerActionType.CALL,
            confidence=0.99,
            expected_state_version=99,
        )
    )
    apply(
        simulator.emit(
            engine.state,
            ActionEvidenceState.CANDIDATE,
            action=PlayerActionType.CALL,
            confidence=0.10,
        )
    )
    apply(
        simulator.emit(
            engine.state,
            ActionEvidenceState.CANDIDATE,
            action=PlayerActionType.CHECK,
            confidence=0.99,
        )
    )
    duplicate = simulator.emit(engine.state, ActionEvidenceState.NO_ACTION)
    apply(duplicate)
    apply(duplicate)

    before_recovery = state_to_dict(engine.snapshot())
    unchanged = before_recovery == initial
    recovery = engine.apply_observation(
        simulator.emit(
            engine.state,
            ActionEvidenceState.CANDIDATE,
            action=PlayerActionType.CALL,
            confidence=0.99,
        )
    )
    recovery_ok = (
        recovery.accepted
        and engine.state.state_version == 1
        and engine.state.acting_seat is Seat.A
    )
    return {
        "schema_version": "1.0",
        "replay_id": "stage2a-action-safety-replay-v1",
        "result": (
            "PASS" if accepted == 0 and unchanged and recovery_ok else "FAIL"
        ),
        "no_action_events": no_action_events,
        "rejected_before_recovery": sum(reasons.values()),
        "accepted_before_recovery": accepted,
        "rejection_reasons": dict(sorted(reasons.items())),
        "state_and_ledger_unchanged_before_recovery": unchanged,
        "recovery_action_accepted": recovery.accepted,
        "recovery_state_version": engine.state.state_version,
        "recovery_next_seat": (
            engine.state.acting_seat.value if engine.state.acting_seat else None
        ),
        "physical_robot_connected": False,
    }
