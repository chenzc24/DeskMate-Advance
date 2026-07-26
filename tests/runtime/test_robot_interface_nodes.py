from __future__ import annotations

from poker_dealer.domain import (
    DealerTargetSlot,
    FaceEnrollmentObservation,
    FaceEnrollmentStatus,
    HandPhase,
    RobotInputKind,
    RobotWorkflowNode,
    Seat,
)
from poker_dealer.game import CoreGameConfig, HandEngine, SimulatedDealer
from poker_dealer.runtime import (
    HandRuntime,
    PartAPhase,
    RegistrationRuntime,
    SessionRuntime,
    default_replay_roster,
)


def test_hole_deal_exposes_navigation_or_legacy_target_ack() -> None:
    runtime = HandRuntime.new_hand(
        hand_id="interface-hole",
        session_id="session",
        button=Seat.A,
    )
    requirement = runtime.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_TARGET_ACK
    assert requirement.target_slot is DealerTargetSlot.SEAT_A
    assert requirement.accepted_inputs == (
        RobotInputKind.NAVIGATION_ACK,
        RobotInputKind.LEGACY_DEALER_ACK,
    )


def test_part_a_exposes_identity_action_and_recovery_input_nodes() -> None:
    runtime = HandRuntime(
        HandEngine.start("interface-action", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = runtime.request_rotation(1)
    assert runtime.accept_rotation_ack(dealer.execute(command, 2))

    requirement = runtime.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_FACE_IDENTITY
    assert requirement.accepted_inputs == (RobotInputKind.FACE_IDENTITY,)

    runtime.engine.pause("pause-interface", "operator_test")
    runtime.sync()
    requirement = runtime.robot_requirement()
    assert runtime.phase is HandPhase.PAUSED_RECOVERY
    assert requirement.node is RobotWorkflowNode.WAITING_OPERATOR_CONTROL
    assert requirement.accepted_inputs == (RobotInputKind.OPERATOR_CONTROL,)


def test_part_a_legacy_node_remains_available_for_existing_runtime() -> None:
    runtime = HandRuntime(
        HandEngine.start("legacy-node", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    assert runtime.part_a is not None
    assert runtime.part_a.phase is PartAPhase.WAITING_ROTATION_ACK
    assert runtime.robot_requirement().node is RobotWorkflowNode.WAITING_TARGET_ACK


def test_registration_exposes_control_then_typed_face_capture_gate() -> None:
    runtime = RegistrationRuntime("registration-session", Seat.A)
    requirement = runtime.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_REGISTRATION_CONTROL
    assert requirement.accepted_inputs == (RobotInputKind.OPERATOR_CONTROL,)

    from poker_dealer.domain import (
        ControlIntent,
        ControlObservation,
        ControlSource,
    )

    started = runtime.accept_control(
        ControlObservation(
            "registration-control-1",
            ControlIntent.CONFIRM,
            ControlSource.SIMULATOR,
            1,
            "test",
            1,
        )
    )
    assert started.accepted
    requirement = runtime.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_FACE_ENROLLMENT
    assert requirement.accepted_inputs == (
        RobotInputKind.FACE_ENROLLMENT,
        RobotInputKind.OPERATOR_CONTROL,
    )
    observation = FaceEnrollmentObservation(
        observation_id="registration-face-1",
        session_id=runtime.session_id,
        expected_roster_version=runtime.roster_version,
        focus_seat=runtime.focus_seat,
        observed_at_ns=2,
        status=FaceEnrollmentStatus.CONFIRMED,
        sample_count=8,
        stable_frames=8,
        model_version="face@test",
    )
    assert requirement.accepts(observation)
    assert runtime.accept_face_enrollment(observation).accepted


def test_session_between_hands_exposes_operator_control_gate() -> None:
    session = SessionRuntime(
        default_replay_roster(),
        CoreGameConfig.from_json("configs/game/core_v1.json"),
    )
    requirement = session.robot_requirement()
    assert requirement.node is RobotWorkflowNode.WAITING_SESSION_CONTROL
    assert requirement.accepted_inputs == (RobotInputKind.OPERATOR_CONTROL,)
