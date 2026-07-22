from __future__ import annotations

from dataclasses import replace

import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    DealerAck,
    DealerAckStatus,
    DealerCommandType,
    DealerDeviceState,
    DealerErrorCode,
    DealerSensorEvidence,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import HandEngine, SimulatedDealer
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState
from poker_dealer.perception.attribution import (
    ActorBindingLease,
    AttributedActionCandidate,
)
from poker_dealer.runtime import PartAPhase, SequentialPartACoordinator


def identity(
    coordinator: SequentialPartACoordinator,
    state: FaceIdentityState,
    *,
    player_id: str | None = None,
    registered_seat: Seat | None = None,
) -> FaceIdentityObservation:
    seat = coordinator.focus_seat
    assert seat is not None
    return FaceIdentityObservation(
        observation_id=f"identity:{state.value}",
        session_id=coordinator.session_id,
        expected_state_version=coordinator.engine.state.state_version,
        observed_at_ns=1_000_000_000,
        focus_seat=seat,
        identity_state=state,
        player_id=player_id,
        registered_seat=registered_seat,
        similarity=0.9 if player_id else None,
        second_best_similarity=None,
        stable_frames=5,
        stable_duration_ms=300,
        model_version="identity@test",
        policy_version="s0-21-session-face-v1",
    )


def action(coordinator: SequentialPartACoordinator) -> PlayerActionObservation:
    seat = coordinator.focus_seat
    assert seat is not None
    return PlayerActionObservation(
        observation_id=f"action:{coordinator.engine.state.state_version}",
        hand_id=coordinator.engine.state.hand_id,
        expected_state_version=coordinator.engine.state.state_version,
        window_started_at_ns=1_000_000_000,
        observed_at_ns=1_400_000_000,
        focus_seat=seat,
        evidence_state=ActionEvidenceState.CANDIDATE,
        candidate_action=PlayerActionType.CALL,
        confidence=0.97,
        stable_duration_ms=400,
        stable_frames=5,
        model_version="multimodal-action-fusion@1.0-development",
        calibration_version="test",
        quality_flags=("multimodal_agreement",),
    )


def ready_coordinator() -> tuple[SequentialPartACoordinator, SimulatedDealer]:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("vertical", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    home = coordinator.request_rotation(1)
    assert home.command is DealerCommandType.ROTATE_TO
    dealer.homed = True
    ack = dealer.execute(home, 2)
    assert coordinator.accept_rotation_ack(ack)
    return coordinator, dealer


def test_rotation_identity_action_and_next_seat_close_loop() -> None:
    coordinator, _dealer = ready_coordinator()
    assert coordinator.focus_seat is Seat.D
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY
    assert not coordinator.accept_identity(identity(coordinator, FaceIdentityState.UNKNOWN))
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY
    assert coordinator.accept_identity(
        identity(
            coordinator,
            FaceIdentityState.MATCHED,
            player_id="player_d",
            registered_seat=Seat.D,
        )
    )
    assert coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION
    result = coordinator.accept_action(action(coordinator))
    assert result.accepted
    assert result.next_seat is Seat.A
    assert coordinator.phase is PartAPhase.WAITING_ROTATION_ACK
    assert coordinator.engine.log.events[-1].payload["source"] == "multimodal_adapter"


def test_action_is_blocked_until_identity_is_verified() -> None:
    coordinator, _dealer = ready_coordinator()
    result = coordinator.accept_action(action(coordinator))
    assert not result.accepted
    assert result.reason == "identity_not_verified"
    assert coordinator.engine.state.state_version == 0


def test_wrong_identity_context_and_seat_mismatch_hold_focus() -> None:
    coordinator, _dealer = ready_coordinator()
    mismatch = identity(
        coordinator,
        FaceIdentityState.SEAT_MISMATCH,
        player_id="player_a",
        registered_seat=Seat.A,
    )
    assert not coordinator.accept_identity(mismatch)
    assert coordinator.focus_seat is Seat.D
    stale = replace(mismatch, expected_state_version=99)
    assert not coordinator.accept_identity(stale)
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY


def test_rotation_ack_target_mismatch_enters_recovery() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("vertical", Seat.A), "session"
    )
    command = coordinator.request_rotation(1)
    bad_ack = DealerAck(
        command_id="wrong",
        command=DealerCommandType.ROTATE_TO,
        target_slot=command.target_slot,
        status=DealerAckStatus.FAILED,
        observed_at_ns=2,
        device_state=DealerDeviceState.FAULT,
        device_state_version=1,
        sensor_evidence=DealerSensorEvidence(False, False, True, 0, True, False),
        error_code=DealerErrorCode.POSITION_TIMEOUT,
        reason="test",
    )
    assert not coordinator.accept_rotation_ack(bad_ack)
    assert coordinator.phase is PartAPhase.RECOVERY_REQUIRED
    assert coordinator.engine.state.phase.value == "paused_recovery"
    assert coordinator.pending_rotation is None


def test_duplicate_rotation_ack_is_idempotent() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("duplicate-ack", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    ack = dealer.execute(command, 2)
    assert coordinator.accept_rotation_ack(ack)
    assert coordinator.accept_rotation_ack(ack)
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY


def test_open_action_window_timeout_pauses_authoritative_hand() -> None:
    coordinator, _dealer = ready_coordinator()
    assert coordinator.accept_identity(
        identity(
            coordinator,
            FaceIdentityState.MATCHED,
            player_id="player_d",
            registered_seat=Seat.D,
        )
    )
    assert coordinator.check_timeout(31_000_000_000)
    assert coordinator.phase is PartAPhase.RECOVERY_REQUIRED
    assert coordinator.engine.state.phase.value == "paused_recovery"
    assert coordinator.engine.state.paused_reason == "player_action_timeout"


def test_illegal_action_does_not_advance_focus() -> None:
    coordinator, _dealer = ready_coordinator()
    coordinator.accept_identity(
        identity(
            coordinator,
            FaceIdentityState.MATCHED,
            player_id="player_d",
            registered_seat=Seat.D,
        )
    )
    illegal = replace(action(coordinator), candidate_action=PlayerActionType.CHECK)
    result = coordinator.accept_action(illegal)
    assert not result.accepted
    assert result.reason == "illegal_action"
    assert coordinator.focus_seat is Seat.D
    assert coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION


def test_identity_revocation_closes_action_window_without_state_change() -> None:
    coordinator, _dealer = ready_coordinator()
    coordinator.accept_identity(
        identity(
            coordinator,
            FaceIdentityState.MATCHED,
            player_id="player_d",
            registered_seat=Seat.D,
        )
    )
    version = coordinator.engine.state.state_version
    coordinator.revoke_identity("face_missing_beyond_grace")
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY
    assert coordinator.verified_player_id is None
    assert coordinator.engine.state.state_version == version
    assert coordinator.focus_seat is Seat.D


def test_four_player_preflop_coordinator_closes_d_a_b_c_in_order() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("four-player-vertical", Seat.A),
        "session",
        require_actor_binding=False,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    expected = (Seat.D, Seat.A, Seat.B, Seat.C)
    actions = (
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CALL,
        PlayerActionType.CHECK,
    )
    for index, (seat, player_action) in enumerate(zip(expected, actions), 1):
        assert coordinator.focus_seat is seat
        command = coordinator.request_rotation(index * 10)
        assert command.target_slot.value == seat.value  # type: ignore[union-attr]
        assert coordinator.accept_rotation_ack(
            dealer.execute(command, index * 10 + 1)
        )
        assert coordinator.accept_identity(
            identity(
                coordinator,
                FaceIdentityState.MATCHED,
                player_id=f"player_{seat.value[-1]}",
                registered_seat=seat,
            )
        )
        item = replace(
            action(coordinator),
            observation_id=f"four-player-action:{index}",
            candidate_action=player_action,
        )
        outcome = coordinator.accept_action(item)
        assert outcome.accepted
        assert coordinator.engine.state.state_version == index
    assert coordinator.phase is PartAPhase.ROUND_COMPLETE
    assert coordinator.engine.state.phase.value == "dealing_board"
    assert coordinator.focus_seat is None


def test_strict_coordinator_requires_matching_actor_binding() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("strict-vertical", Seat.A),
        "session",
        require_actor_binding=True,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(command, 2))
    matched = identity(
        coordinator,
        FaceIdentityState.MATCHED,
        player_id="player_d",
        registered_seat=Seat.D,
    )
    assert coordinator.accept_identity(matched)
    raw = action(coordinator)
    assert coordinator.accept_action(raw).reason == "attributed_action_required"
    assert coordinator.engine.state.state_version == 0

    lease = ActorBindingLease(lease_ms=2000)
    binding = lease.open(
        matched, hand_id=coordinator.engine.state.hand_id, person_track_id="person:1"
    )
    coordinator.bind_actor(binding)
    bound = AttributedActionCandidate(raw, binding, "pose_wrist", 0.9)
    outcome = coordinator.accept_attributed_action(bound)
    assert outcome.accepted
    assert coordinator.engine.state.state_version == 1
    assert coordinator.active_actor_binding is None


def test_strict_coordinator_rejects_another_binding_id() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("strict-mismatch", Seat.A),
        "session",
        require_actor_binding=True,
        require_visual_settle=False,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(command, 2))
    matched = identity(
        coordinator,
        FaceIdentityState.MATCHED,
        player_id="player_d",
        registered_seat=Seat.D,
    )
    assert coordinator.accept_identity(matched)
    lease = ActorBindingLease(lease_ms=2000)
    first = lease.open(
        matched, hand_id=coordinator.engine.state.hand_id, person_track_id="person:1"
    )
    coordinator.bind_actor(first)
    second = ActorBindingLease(lease_ms=2000).open(
        matched, hand_id=coordinator.engine.state.hand_id, person_track_id="person:2"
    )
    outcome = coordinator.accept_attributed_action(
        AttributedActionCandidate(action(coordinator), second, "pose_wrist", 0.9)
    )
    assert not outcome.accepted
    assert outcome.reason == "actor_binding_mismatch"
    assert coordinator.engine.state.state_version == 0


def test_visual_settle_gate_is_required_when_enabled() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("settle", Seat.A),
        "session",
        require_visual_settle=True,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(command, 2))
    assert coordinator.phase is PartAPhase.WAITING_VISUAL_SETTLE
    with pytest.raises(ValueError, match="identity evidence"):
        coordinator.accept_identity(identity(coordinator, FaceIdentityState.UNKNOWN))
    coordinator.accept_visual_settle()
    assert coordinator.phase is PartAPhase.VERIFYING_IDENTITY


def test_visual_settle_timeout_pauses_hand() -> None:
    coordinator = SequentialPartACoordinator(
        HandEngine.start("settle-timeout", Seat.A),
        "session",
        require_actor_binding=True,
        require_visual_settle=True,
        visual_settle_timeout_ms=5,
    )
    dealer = SimulatedDealer()
    dealer.homed = True
    command = coordinator.request_rotation(1)
    assert coordinator.accept_rotation_ack(dealer.execute(command, 2))
    assert coordinator.check_timeout(2 + 5_000_000)
    assert coordinator.engine.state.phase.value == "paused_recovery"
    assert coordinator.engine.state.paused_reason == "visual_settle_timeout"
