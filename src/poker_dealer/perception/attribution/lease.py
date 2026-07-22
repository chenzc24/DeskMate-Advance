"""Fail-closed lifecycle for expiring session actor bindings."""

from __future__ import annotations

from dataclasses import replace

from poker_dealer.perception.identity import FaceIdentityObservation, FaceIdentityState

from .domain import ActorBinding, ActorBindingState


class ActorBindingLease:
    def __init__(self, *, lease_ms: int = 2500) -> None:
        if lease_ms <= 0:
            raise ValueError("actor binding lease must be positive")
        self.lease_ms = lease_ms
        self._binding: ActorBinding | None = None
        self._state = ActorBindingState.REVOKED
        self._last_reason = "not_opened"
        self._sequence = 0

    @property
    def binding(self) -> ActorBinding | None:
        return self._binding

    @property
    def state(self) -> ActorBindingState:
        return self._state

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def open(
        self,
        observation: FaceIdentityObservation,
        *,
        hand_id: str,
        person_track_id: str,
        camera_epoch: int = 0,
    ) -> ActorBinding:
        if observation.identity_state is not FaceIdentityState.MATCHED:
            raise ValueError("actor binding requires a matched face observation")
        assert observation.player_id is not None and observation.similarity is not None
        self._sequence += 1
        binding = ActorBinding(
            binding_id=(
                f"actor:{observation.session_id}:{hand_id}:"
                f"{observation.expected_state_version}:{person_track_id}:{self._sequence}"
            ),
            session_id=observation.session_id,
            hand_id=hand_id,
            expected_state_version=observation.expected_state_version,
            focus_seat=observation.focus_seat,
            player_id=observation.player_id,
            person_track_id=person_track_id,
            verified_at_ns=observation.observed_at_ns,
            valid_until_ns=(
                observation.observed_at_ns + self.lease_ms * 1_000_000
            ),
            identity_confidence=max(0.0, min(1.0, observation.similarity)),
            camera_epoch=camera_epoch,
        )
        self._binding = binding
        self._state = ActorBindingState.ACTIVE
        self._last_reason = "matched_identity_opened"
        return binding

    def observe_identity(self, observation: FaceIdentityObservation) -> bool:
        binding = self._binding
        if binding is None or self._state is not ActorBindingState.ACTIVE:
            return False
        context_matches = (
            observation.session_id == binding.session_id
            and observation.expected_state_version == binding.expected_state_version
            and observation.focus_seat is binding.focus_seat
        )
        if not context_matches:
            self.revoke("identity_context_changed")
            return False
        if observation.identity_state is FaceIdentityState.MATCHED:
            if observation.player_id != binding.player_id:
                self.revoke("different_registered_player")
                return False
            assert observation.similarity is not None
            self._binding = replace(
                binding,
                valid_until_ns=observation.observed_at_ns + self.lease_ms * 1_000_000,
                identity_confidence=max(0.0, min(1.0, observation.similarity)),
            )
            self._last_reason = "matched_identity_refreshed"
            return True
        if observation.identity_state is FaceIdentityState.SEAT_MISMATCH:
            self.revoke("different_registered_player")
            return False
        # Missing, low-quality and multiple-face frames do not revoke a binding.
        # The lease expires unless the expected player is seen again.
        self._last_reason = f"identity_{observation.identity_state.value}_within_lease"
        return self.is_valid_at(observation.observed_at_ns)

    def is_valid_at(self, observed_at_ns: int, *, camera_epoch: int | None = None) -> bool:
        binding = self._binding
        if binding is None or self._state is not ActorBindingState.ACTIVE:
            return False
        if camera_epoch is not None and camera_epoch != binding.camera_epoch:
            self.revoke("camera_epoch_changed")
            return False
        if not binding.is_valid_at(observed_at_ns):
            self._state = ActorBindingState.EXPIRED
            self._last_reason = "identity_lease_expired"
            return False
        return True

    def revoke(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("actor binding revocation reason is required")
        self._binding = None
        self._state = ActorBindingState.REVOKED
        self._last_reason = reason

    def clear(self) -> None:
        self.revoke("session_cleared")
