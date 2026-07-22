from __future__ import annotations

from dataclasses import replace

import pytest

from poker_dealer.domain import (
    ActionEvidenceState,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
)
from poker_dealer.perception.attribution import (
    ActorBindingLease,
    ActorBindingState,
    AttributedActionCandidate,
)
from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState


def identity(
    state: FaceIdentityState = FaceIdentityState.MATCHED,
    *,
    observed_at_ns: int = 1_000_000_000,
    player_id: str | None = "player_d",
    registered_seat: Seat | None = Seat.D,
) -> FaceIdentityObservation:
    identified = state in {FaceIdentityState.MATCHED, FaceIdentityState.SEAT_MISMATCH}
    return FaceIdentityObservation(
        observation_id=f"identity:{state}:{observed_at_ns}",
        session_id="session",
        expected_state_version=3,
        observed_at_ns=observed_at_ns,
        focus_seat=Seat.D,
        identity_state=state,
        player_id=player_id if identified else None,
        registered_seat=registered_seat if identified else None,
        similarity=0.8 if identified else None,
        second_best_similarity=None,
        stable_frames=5,
        stable_duration_ms=300,
        model_version="identity@test",
        policy_version="test",
    )


def action(observed_at_ns: int) -> PlayerActionObservation:
    return PlayerActionObservation(
        observation_id=f"action:{observed_at_ns}",
        hand_id="hand",
        expected_state_version=3,
        window_started_at_ns=observed_at_ns - 300_000_000,
        observed_at_ns=observed_at_ns,
        focus_seat=Seat.D,
        evidence_state=ActionEvidenceState.CANDIDATE,
        candidate_action=PlayerActionType.CALL,
        confidence=0.95,
        stable_duration_ms=300,
        stable_frames=5,
        model_version="player-action-mediapipe@test",
        calibration_version="test",
    )


def test_missing_face_keeps_lease_until_expiry_and_match_refreshes() -> None:
    lease = ActorBindingLease(lease_ms=2500)
    opened = lease.open(identity(), hand_id="hand", person_track_id="person:1")
    no_face = identity(
        FaceIdentityState.NO_FACE,
        observed_at_ns=2_000_000_000,
        player_id=None,
        registered_seat=None,
    )
    assert lease.observe_identity(no_face)
    assert lease.binding == opened
    refreshed = identity(observed_at_ns=2_400_000_000)
    assert lease.observe_identity(refreshed)
    assert lease.binding is not None
    assert lease.binding.valid_until_ns == 4_900_000_000


def test_wrong_registered_player_revokes_immediately() -> None:
    lease = ActorBindingLease()
    lease.open(identity(), hand_id="hand", person_track_id="person:1")
    mismatch = identity(
        FaceIdentityState.SEAT_MISMATCH,
        observed_at_ns=1_100_000_000,
        player_id="player_a",
        registered_seat=Seat.A,
    )
    assert not lease.observe_identity(mismatch)
    assert lease.state is ActorBindingState.REVOKED
    assert lease.binding is None


def test_expiry_and_camera_epoch_fail_closed() -> None:
    lease = ActorBindingLease(lease_ms=1000)
    lease.open(
        identity(), hand_id="hand", person_track_id="person:1", camera_epoch=2
    )
    assert not lease.is_valid_at(1_500_000_000, camera_epoch=3)
    assert lease.last_reason == "camera_epoch_changed"

    lease.open(identity(), hand_id="hand", person_track_id="person:2")
    assert not lease.is_valid_at(2_100_000_000)
    assert lease.state is ActorBindingState.EXPIRED


def test_attributed_candidate_rejects_context_or_expired_binding() -> None:
    lease = ActorBindingLease(lease_ms=2500)
    binding = lease.open(identity(), hand_id="hand", person_track_id="person:1")
    candidate = AttributedActionCandidate(
        action(1_400_000_000), binding, "pose_wrist", 0.9
    )
    assert candidate.binding.player_id == "player_d"
    with pytest.raises(ValueError, match="does not match"):
        AttributedActionCandidate(
            replace(action(1_400_000_000), focus_seat=Seat.A),
            binding,
            "pose_wrist",
            0.9,
        )
    with pytest.raises(ValueError, match="outside"):
        AttributedActionCandidate(
            action(4_000_000_000), binding, "pose_wrist", 0.9
        )
