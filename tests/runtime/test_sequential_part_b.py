from __future__ import annotations

from poker_dealer.domain import (
    CardIdentity,
    HandPhase,
    ObservationStatus,
    Rank,
    Seat,
    Suit,
)
from poker_dealer.game import (
    EventLog,
    FixedLimitRules,
    HandEngine,
    SimulatedCardPerception,
    SimulatedDealer,
    SlotLifecycle,
)
from poker_dealer.runtime import HandRuntime, PartBPhase, SequentialPartBCoordinator


def test_hole_delivery_requires_ack_and_face_down_evidence_for_all_eight() -> None:
    engine = HandEngine.setup_session("hole-delivery", Seat.A)
    engine.begin_hand("begin")
    coordinator = SequentialPartBCoordinator(engine)
    perception = SimulatedCardPerception()
    dealer = SimulatedDealer()
    dealer.homed = True

    for index in range(8):
        now = (index + 1) * 10_000_000
        rotation = coordinator.request_rotation(now)
        rotation_ack = dealer.execute(rotation, now + 1)
        assert coordinator.accept_rotation_ack(rotation_ack)
        assert coordinator.accept_rotation_ack(rotation_ack)  # duplicate is idempotent
        dispense = coordinator.request_dispense(now + 2)
        dispense_ack = dealer.execute(dispense, now + 3)
        assert coordinator.accept_dispense_ack(dispense_ack)
        step = coordinator.current_step
        assert step is not None and len(step.vision_slots) == 1
        result = coordinator.accept_card_observation(
            perception.emit(
                step.vision_slots[0],
                ObservationStatus.FACE_DOWN,
                observed_at_ns=now + 4,
            )
        )
        assert result.accepted

    assert coordinator.phase is PartBPhase.COMPLETE
    assert engine.state.phase is HandPhase.AWAITING_ACTION
    assert engine.state.acting_seat is Seat.D
    assert dealer.dispensed_cards == 8


def test_board_visual_unknown_holds_and_timeout_pauses_authoritative_hand() -> None:
    engine = HandEngine.start("board-timeout", Seat.A)
    for index, action in enumerate(("call", "call", "call", "check")):
        from poker_dealer.domain import PlayerActionType
        from poker_dealer.game import ActionRequest

        state = engine.state
        result = engine.apply_action(
            ActionRequest(
                f"action-{index}",
                state.hand_id,
                state.state_version,
                state.acting_seat,  # type: ignore[arg-type]
                PlayerActionType(action),
            )
        )
        assert result.accepted

    coordinator = SequentialPartBCoordinator(engine, visual_timeout_ms=5)
    dealer = SimulatedDealer()
    dealer.homed = True
    rotation = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(rotation, 2))
    dispense = coordinator.request_dispense(3)
    assert coordinator.accept_dispense_ack(dealer.execute(dispense, 4))
    step = coordinator.current_step
    assert step is not None
    unknown = SimulatedCardPerception().emit(
        step.vision_slots[0], ObservationStatus.UNKNOWN, observed_at_ns=5
    )
    assert not coordinator.accept_card_observation(unknown).accepted
    assert engine.state.phase is HandPhase.DEALING_BOARD

    assert coordinator.check_timeout(4 + 5_000_000)
    assert coordinator.phase is PartBPhase.RECOVERY_REQUIRED
    assert engine.state.phase is HandPhase.PAUSED_RECOVERY
    assert engine.state.paused_reason == "card_visual_timeout"


def test_same_slot_card_change_is_a_hard_conflict() -> None:
    engine = HandEngine.start("same-slot-conflict", Seat.A)
    from poker_dealer.domain import PlayerActionType, VisionSlot
    from poker_dealer.game import ActionRequest

    for index, action in enumerate(
        (
            PlayerActionType.CALL,
            PlayerActionType.CALL,
            PlayerActionType.CALL,
            PlayerActionType.CHECK,
        )
    ):
        state = engine.state
        assert engine.apply_action(
            ActionRequest(
                f"action-{index}",
                state.hand_id,
                state.state_version,
                state.acting_seat,  # type: ignore[arg-type]
                action,
            )
        ).accepted

    simulator = SimulatedCardPerception()
    first = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.CONFIRMED,
        card=CardIdentity(Rank.ACE, Suit.SPADES),
        confidence=0.99,
        observed_at_ns=1,
    )
    assert engine.apply_card_observation(first).accepted
    version = engine.state.state_version
    repeat = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.CONFIRMED,
        card=CardIdentity(Rank.ACE, Suit.SPADES),
        confidence=0.99,
        observed_at_ns=2,
    )
    assert engine.apply_card_observation(repeat).reason == "card_already_confirmed"
    assert engine.state.state_version == version
    changed = simulator.emit(
        VisionSlot.BOARD_FLOP_1,
        ObservationStatus.CONFIRMED,
        card=CardIdentity(Rank.KING, Suit.SPADES),
        confidence=0.99,
        observed_at_ns=3,
    )
    result = engine.apply_card_observation(changed)
    assert not result.accepted
    assert result.reason == "slot_card_identity_changed"
    assert engine.state.phase is HandPhase.PAUSED_RECOVERY


def test_restart_after_dispense_ack_waits_for_visual_without_redealing() -> None:
    engine = HandEngine.setup_session("resume-visual", Seat.A)
    engine.begin_hand("begin")
    coordinator = SequentialPartBCoordinator(engine)
    dealer = SimulatedDealer()
    dealer.homed = True
    rotation = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(rotation, 2))
    dispense = coordinator.request_dispense(3)
    assert coordinator.accept_dispense_ack(dealer.execute(dispense, 4))
    step = coordinator.current_step
    assert step is not None
    slot = step.vision_slots[0]
    assert engine.state.slot_states[slot] is SlotLifecycle.DELIVERY_PENDING

    recovered = HandEngine.from_log(
        FixedLimitRules(), EventLog.from_jsonl(engine.log.to_jsonl())
    )
    resumed = SequentialPartBCoordinator(recovered)
    assert resumed.phase is PartBPhase.WAITING_VISUAL_CONFIRMATION
    assert resumed.current_step is not None
    assert resumed.current_step.vision_slots == (slot,)
    observation = SimulatedCardPerception().emit(
        slot, ObservationStatus.FACE_DOWN, observed_at_ns=5
    )
    assert resumed.accept_card_observation(observation).accepted
    assert resumed.phase is PartBPhase.WAITING_ROTATION_ACK
    assert dealer.dispensed_cards == 1


def test_restart_with_unresolved_command_pauses_instead_of_guessing() -> None:
    engine = HandEngine.setup_session("resume-pending", Seat.A)
    engine.begin_hand("begin")
    coordinator = SequentialPartBCoordinator(engine)
    coordinator.request_rotation(1)
    recovered = HandEngine.from_log(
        FixedLimitRules(), EventLog.from_jsonl(engine.log.to_jsonl())
    )
    runtime = HandRuntime(recovered, "session")
    assert runtime.phase is HandPhase.PAUSED_RECOVERY
    assert runtime.engine.state.paused_reason == "recovered_with_pending_dealer_command"
