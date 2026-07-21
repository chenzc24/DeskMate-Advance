"""Temporal confirmation for session-only player identity matches."""

from __future__ import annotations

from .config import FaceIdentityConfig
from .domain import (
    FaceIdentityContext,
    FaceIdentityObservation,
    FaceIdentityState,
)
from .gallery import FaceMatchResult


class FaceIdentityTemporalAdapter:
    def __init__(self, config: FaceIdentityConfig) -> None:
        self.config = config
        self._last_timestamp_ns: int | None = None
        self._pending_player_id: str | None = None
        self._pending_started_ns: int | None = None
        self._pending_frames = 0
        self._sequence = 0

    def process(
        self,
        match: FaceMatchResult,
        observed_at_ns: int,
        context: FaceIdentityContext,
    ) -> FaceIdentityObservation:
        if observed_at_ns < 0:
            raise ValueError("identity timestamp must be non-negative")
        if self._last_timestamp_ns is not None and observed_at_ns < self._last_timestamp_ns:
            raise ValueError("identity timestamps must be monotonic")
        self._last_timestamp_ns = observed_at_ns
        self._sequence += 1
        state = match.state
        player_id = None
        registered_seat = None
        stable_frames = 1
        stable_duration_ms = 0
        flags = list(match.quality_flags)

        if match.state is FaceIdentityState.MATCHED:
            assert match.player_id is not None and match.registered_seat is not None
            if match.player_id != self._pending_player_id:
                self._pending_player_id = match.player_id
                self._pending_started_ns = observed_at_ns
                self._pending_frames = 1
            else:
                self._pending_frames += 1
            stable_frames = self._pending_frames
            started_ns = (
                observed_at_ns
                if self._pending_started_ns is None
                else self._pending_started_ns
            )
            stable_duration_ms = (observed_at_ns - started_ns) // 1_000_000
            confirmed = (
                stable_frames >= self.config.minimum_stable_frames
                and stable_duration_ms >= self.config.minimum_stable_duration_ms
            )
            if confirmed:
                player_id = match.player_id
                registered_seat = match.registered_seat
                if registered_seat is context.focus_seat:
                    state = FaceIdentityState.MATCHED
                    flags.append("session_identity_verified")
                else:
                    state = FaceIdentityState.SEAT_MISMATCH
                    flags.append("registered_player_at_wrong_focus_seat")
            else:
                state = FaceIdentityState.IDENTITY_START
                flags.append("identity_temporal_confirmation_pending")
        else:
            self._pending_player_id = None
            self._pending_started_ns = None
            self._pending_frames = 0

        return FaceIdentityObservation(
            observation_id=(
                f"face-identity:{context.session_id}:{self._sequence}:{observed_at_ns}"
            ),
            session_id=context.session_id,
            expected_state_version=context.expected_state_version,
            observed_at_ns=observed_at_ns,
            focus_seat=context.focus_seat,
            identity_state=state,
            player_id=player_id,
            registered_seat=registered_seat,
            similarity=match.similarity,
            second_best_similarity=match.second_best_similarity,
            stable_frames=stable_frames,
            stable_duration_ms=int(stable_duration_ms),
            model_version=self.config.model_version,
            policy_version=self.config.policy_version,
            quality_flags=tuple(dict.fromkeys(flags)),
        )
