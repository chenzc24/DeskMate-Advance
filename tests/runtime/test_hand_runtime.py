from __future__ import annotations

from poker_dealer.domain import (
    ActionEvidenceState,
    CardIdentity,
    HandPhase,
    ObservationStatus,
    PlayerActionObservation,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    TableRole,
    VisionSlot,
)
from poker_dealer.game import SimulatedCardPerception, SimulatedDealer, SlotLifecycle
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState
from poker_dealer.runtime import (
    FrozenSessionRoster,
    HandRuntime,
    PartBMode,
    PartBPhase,
    RegisteredParticipant,
)


def test_one_runtime_drives_a_complete_no_burn_hand() -> None:
    roster = FrozenSessionRoster(
        "session",
        Seat.A,
        1,
        (
            RegisteredParticipant("player-a", Seat.A, TableRole.BUTTON, 5),
            RegisteredParticipant("player-b", Seat.B, TableRole.SMALL_BLIND, 5),
            RegisteredParticipant("player-c", Seat.C, TableRole.BIG_BLIND, 5),
            RegisteredParticipant("player-d", Seat.D, TableRole.UNDER_THE_GUN, 5),
        ),
    )
    runtime = HandRuntime.from_roster(
        hand_id="complete-live-sim",
        roster=roster,
        require_actor_binding=False,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    perception = SimulatedCardPerception()
    clock = 0

    cards = {
        VisionSlot.BOARD_FLOP_1: CardIdentity(Rank.ACE, Suit.SPADES),
        VisionSlot.BOARD_FLOP_2: CardIdentity(Rank.KING, Suit.HEARTS),
        VisionSlot.BOARD_FLOP_3: CardIdentity(Rank.QUEEN, Suit.DIAMONDS),
        VisionSlot.BOARD_TURN: CardIdentity(Rank.JACK, Suit.CLUBS),
        VisionSlot.BOARD_RIVER: CardIdentity(Rank.TWO, Suit.SPADES),
        VisionSlot.SEAT_A_HOLE_1: CardIdentity(Rank.THREE, Suit.CLUBS),
        VisionSlot.SEAT_A_HOLE_2: CardIdentity(Rank.FOUR, Suit.CLUBS),
        VisionSlot.SEAT_B_HOLE_1: CardIdentity(Rank.FIVE, Suit.CLUBS),
        VisionSlot.SEAT_B_HOLE_2: CardIdentity(Rank.SIX, Suit.CLUBS),
        VisionSlot.SEAT_C_HOLE_1: CardIdentity(Rank.SEVEN, Suit.CLUBS),
        VisionSlot.SEAT_C_HOLE_2: CardIdentity(Rank.EIGHT, Suit.CLUBS),
        VisionSlot.SEAT_D_HOLE_1: CardIdentity(Rank.NINE, Suit.CLUBS),
        VisionSlot.SEAT_D_HOLE_2: CardIdentity(Rank.TEN, Suit.CLUBS),
    }

    def tick() -> int:
        nonlocal clock
        clock += 1_000_000
        return clock

    def drive_part_b() -> None:
        while runtime.part_b is not None:
            coordinator = runtime.part_b
            if coordinator.phase is PartBPhase.WAITING_ROTATION_ACK:
                command = runtime.request_rotation(tick())
                assert runtime.accept_rotation_ack(dealer.execute(command, tick()))
                continue
            if coordinator.phase is PartBPhase.WAITING_DISPENSE_ACK:
                command = runtime.request_dispense(tick())
                assert runtime.accept_dispense_ack(dealer.execute(command, tick()))
                continue
            assert coordinator.phase is PartBPhase.WAITING_VISUAL_CONFIRMATION
            step = coordinator.current_step
            assert step is not None
            for slot in step.vision_slots:
                expected = (
                    SlotLifecycle.PRESENT_FACE_DOWN
                    if coordinator.mode is PartBMode.HOLE_DEAL
                    else SlotLifecycle.CONFIRMED
                )
                if runtime.engine.state.slot_states[slot] is expected:
                    continue
                if coordinator.mode is PartBMode.HOLE_DEAL:
                    observation = perception.emit(
                        slot,
                        ObservationStatus.FACE_DOWN,
                        observed_at_ns=tick(),
                    )
                else:
                    observation = perception.emit(
                        slot,
                        ObservationStatus.CONFIRMED,
                        card=cards[slot],
                        confidence=0.999,
                        observed_at_ns=tick(),
                    )
                assert runtime.accept_card_observation(observation).accepted

    def drive_part_a() -> None:
        while runtime.part_a is not None:
            command = runtime.request_rotation(tick())
            assert runtime.accept_rotation_ack(dealer.execute(command, tick()))
            coordinator = runtime.part_a
            assert coordinator is not None and coordinator.focus_seat is not None
            seat = coordinator.focus_seat
            identity_time = tick()
            assert runtime.accept_identity(
                FaceIdentityObservation(
                    observation_id=f"identity:{runtime.engine.state.state_version}",
                    session_id="session",
                    expected_state_version=runtime.engine.state.state_version,
                    observed_at_ns=identity_time,
                    focus_seat=seat,
                    identity_state=FaceIdentityState.MATCHED,
                    player_id=f"player:{seat.value}",
                    registered_seat=seat,
                    similarity=0.95,
                    second_best_similarity=0.1,
                    stable_frames=5,
                    stable_duration_ms=300,
                    model_version="face@test",
                    policy_version="test",
                )
            )
            action = (
                PlayerActionType.CHECK
                if PlayerActionType.CHECK in runtime.engine.state.legal_actions
                else PlayerActionType.CALL
            )
            observed_at = tick()
            outcome = runtime.accept_action(
                PlayerActionObservation(
                    observation_id=f"action:{runtime.engine.state.state_version}",
                    hand_id=runtime.engine.state.hand_id,
                    expected_state_version=runtime.engine.state.state_version,
                    window_started_at_ns=identity_time,
                    observed_at_ns=observed_at,
                    focus_seat=seat,
                    evidence_state=ActionEvidenceState.CANDIDATE,
                    candidate_action=action,
                    confidence=0.99,
                    stable_duration_ms=300,
                    stable_frames=5,
                    model_version="simulated-action@test",
                    calibration_version="test",
                )
            )
            assert outcome.accepted

    transitions = 0
    while runtime.phase is not HandPhase.SETTLED:
        transitions += 1
        assert transitions < 20
        if runtime.part_b is not None:
            drive_part_b()
        elif runtime.part_a is not None:
            drive_part_a()
        else:
            raise AssertionError(f"runtime has no active lane in {runtime.phase.value}")

    assert dealer.dispensed_cards == 13
    assert len(runtime.engine.state.board) == 5
    assert len(runtime.engine.state.hole_cards) == 4
    assert runtime.last_showdown_ranks is not None
    assert runtime.engine.state.total_units() == 320
    command_events = [
        event
        for event in runtime.engine.log.events
        if event.kind == "dealer_command_issued"
    ]
    ack_events = [
        event
        for event in runtime.engine.log.events
        if event.kind == "dealer_ack_received"
    ]
    assert len(command_events) == len(ack_events) == 46
    assert runtime.engine.state.pending_command_id is None


def test_production_defaults_are_fail_closed_and_rotation_timeout_pauses() -> None:
    runtime = HandRuntime.new_hand(
        hand_id="timeout",
        session_id="session",
        button=Seat.A,
        command_timeout_ms=5,
    )
    command = runtime.request_rotation(100)
    assert runtime.check_timeout(100 + 5_000_000)
    assert runtime.phase is HandPhase.PAUSED_RECOVERY
    assert runtime.engine.state.paused_reason == "dealer_command_timeout"
    assert command.command_id.startswith("part-b:timeout:")
