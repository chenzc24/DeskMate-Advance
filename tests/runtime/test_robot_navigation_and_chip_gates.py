from __future__ import annotations

from dataclasses import replace

from poker_dealer.domain import (
    ActionEvidenceState,
    ChipAmountScope,
    ChipCount,
    ChipObservation,
    ChipObservationStatus,
    NavigationAckStatus,
    PlayerActionObservation,
    PlayerActionType,
    RobotInputKind,
    RobotPoseNode,
    RobotWorkflowNode,
    Seat,
)
from poker_dealer.game import HandEngine
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState
from poker_dealer.robotics.dealer import SimulatedDealerAdapter
from poker_dealer.robotics.navigation import SimulatedNavigationAdapter
from poker_dealer.runtime import HandRuntime, PartAPhase
from poker_dealer.runtime.event_log import RuntimeEventWriter
from poker_dealer.runtime.hand_loop import HandRuntimeLoop


class _PhysicalNavigationAdapter(SimulatedNavigationAdapter):
    """Test-only adapter: physical timing semantics without motor I/O."""

    physical_motion = True

    def health(self):
        return replace(super().health(), physical_motion=True)


def _matched_identity(runtime: HandRuntime, observed_at_ns: int) -> FaceIdentityObservation:
    assert runtime.part_a is not None and runtime.part_a.focus_seat is not None
    seat = runtime.part_a.focus_seat
    return FaceIdentityObservation(
        observation_id=f"identity:{seat.value}",
        session_id=runtime.session_id,
        expected_state_version=runtime.engine.state.state_version,
        observed_at_ns=observed_at_ns,
        focus_seat=seat,
        identity_state=FaceIdentityState.MATCHED,
        player_id=f"player-{seat.value}",
        registered_seat=seat,
        similarity=0.95,
        second_best_similarity=0.1,
        stable_frames=5,
        stable_duration_ms=300,
        model_version="identity@test",
        policy_version="test",
    )


def _raise_observation(runtime: HandRuntime, observed_at_ns: int) -> PlayerActionObservation:
    assert runtime.part_a is not None and runtime.part_a.focus_seat is not None
    seat = runtime.part_a.focus_seat
    return PlayerActionObservation(
        observation_id="raise-with-chip-gate",
        hand_id=runtime.engine.state.hand_id,
        expected_state_version=runtime.engine.state.state_version,
        window_started_at_ns=observed_at_ns - 300_000_000,
        observed_at_ns=observed_at_ns,
        focus_seat=seat,
        evidence_state=ActionEvidenceState.CANDIDATE,
        candidate_action=PlayerActionType.RAISE,
        confidence=0.99,
        stable_duration_ms=300,
        stable_frames=5,
        model_version="action@test",
        calibration_version="test",
    )


def test_navigation_ack_replaces_rotation_ack_without_changing_game_state() -> None:
    runtime = HandRuntime(
        HandEngine.start("navigation-gate", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    navigation = SimulatedNavigationAdapter(
        initial_pose=RobotPoseNode.INIT_TO_END
    )
    navigation.open()
    health = navigation.health()
    command = runtime.request_navigation(
        1,
        start_pose=health.pose,
        expected_pose_version=health.pose_version,
    )
    assert command.target_pose is RobotPoseNode.INIT_UTG
    assert runtime.part_a is not None
    assert runtime.part_a.phase is PartAPhase.WAITING_NAVIGATION_ACK
    ack = navigation.execute(command, 2)
    assert ack.status is NavigationAckStatus.SUCCEEDED
    assert runtime.accept_navigation_ack(ack)
    assert runtime.part_a is not None
    assert runtime.part_a.phase is PartAPhase.VERIFYING_IDENTITY
    assert runtime.engine.state.state_version == command.expected_state_version


def test_hole_navigation_uses_line_facing_button_deal_pose() -> None:
    runtime = HandRuntime.new_hand(
        hand_id="hole-route-pose",
        session_id="session",
        button=Seat.A,
    )
    navigation = SimulatedNavigationAdapter(
        initial_pose=RobotPoseNode.INIT_TO_END
    )
    navigation.open()
    health = navigation.health()
    command = runtime.request_navigation(
        1,
        start_pose=health.pose,
        expected_pose_version=health.pose_version,
    )
    assert command.target_pose is RobotPoseNode.INIT_TO_END


def test_hand_loop_waits_2500_ms_between_physical_navigation_commands(
    tmp_path,
) -> None:
    runtime = HandRuntime(
        HandEngine.start("physical-navigation-gap", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    navigation = _PhysicalNavigationAdapter()
    navigation.open()
    dealer = SimulatedDealerAdapter("simulated-dealer")
    dealer.open()
    clock_value = [100_000_000]

    with RuntimeEventWriter(tmp_path / "events.jsonl") as writer:
        loop = HandRuntimeLoop(
            runtime,
            dealer,
            identity_source=object(),
            action_source=object(),
            card_source=object(),
            navigation_port=navigation,
            event_writer=writer,
            clock_ns=lambda: clock_value[0],
        )
        assert loop._execute_navigation(0)
        assert navigation.health().pose_version == 1

        assert runtime.accept_identity(_matched_identity(runtime, 200_000_000))
        outcome = runtime.accept_action(_raise_observation(runtime, 500_000_000))
        assert outcome.accepted
        assert runtime.part_a is not None
        assert runtime.part_a.phase is PartAPhase.WAITING_ROTATION_ACK

        clock_value[0] = 2_599_999_999
        assert not loop._execute_navigation(clock_value[0])
        assert loop.navigation_cooldown_remaining_ms(clock_value[0]) == 1
        assert navigation.health().pose_version == 1
        assert runtime.part_a.pending_navigation is None

        clock_value[0] = 2_600_000_000
        assert loop.navigation_cooldown_remaining_ms(clock_value[0]) == 0
        assert loop._execute_navigation(clock_value[0])
        assert navigation.health().pose_version == 2
        assert runtime.part_a is not None
        assert runtime.part_a.pending_navigation is None

    dealer.close()
    navigation.close()


def test_raise_waits_for_matching_chip_observation_before_commit() -> None:
    runtime = HandRuntime(
        HandEngine.start("chip-gate", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
        require_chip_observation=True,
    )
    navigation = SimulatedNavigationAdapter()
    navigation.open()
    health = navigation.health()
    command = runtime.request_navigation(
        1,
        start_pose=health.pose,
        expected_pose_version=health.pose_version,
    )
    assert runtime.accept_navigation_ack(navigation.execute(command, 2))
    assert runtime.accept_identity(_matched_identity(runtime, 3))

    before_version = runtime.engine.state.state_version
    action = _raise_observation(runtime, 400_000_000)
    outcome = runtime.accept_action(action)
    assert not outcome.accepted
    assert outcome.reason == "chip_observation_required"
    assert runtime.engine.state.state_version == before_version
    requirement = runtime.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_CHIP_OBSERVATION
    assert requirement.accepted_inputs == (RobotInputKind.CHIP_OBSERVATION,)

    seat = action.focus_seat
    mismatch = ChipObservation(
        observation_id="chips-mismatch",
        hand_id=action.hand_id,
        expected_state_version=before_version,
        focus_seat=seat,
        observed_at_ns=500_000_000,
        status=ChipObservationStatus.CONFIRMED,
        amount_scope=ChipAmountScope.NEW_CONTRIBUTION,
        chip_counts=(ChipCount(2, 1),),
        total_units=2,
        confidence=0.99,
        stable_frames=5,
        model_version="chip@test",
        calibration_version="test",
    )
    mismatch_outcome = runtime.accept_chip_observation(mismatch)
    assert not mismatch_outcome.accepted
    assert mismatch_outcome.reason == "chip_amount_mismatch"
    assert runtime.engine.state.state_version == before_version

    confirmed = ChipObservation(
        observation_id="chips-confirmed",
        hand_id=action.hand_id,
        expected_state_version=before_version,
        focus_seat=seat,
        observed_at_ns=600_000_000,
        status=ChipObservationStatus.CONFIRMED,
        amount_scope=ChipAmountScope.NEW_CONTRIBUTION,
        chip_counts=(ChipCount(2, 2),),
        total_units=4,
        confidence=0.99,
        stable_frames=5,
        model_version="chip@test",
        calibration_version="test",
    )
    committed = runtime.accept_chip_observation(confirmed)
    assert committed.accepted
    assert runtime.engine.state.state_version == before_version + 1
