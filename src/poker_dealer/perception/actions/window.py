"""Bounded decision window for gesture and speech action candidates."""

from __future__ import annotations

from dataclasses import replace

from poker_dealer.domain import ActionEvidenceState, PlayerActionObservation

from .fusion import action_observation_source, fuse_action_observations


class MultimodalActionWindow:
    """Wait briefly for a second modality, then emit one fused observation."""

    def __init__(
        self,
        *,
        decision_wait_ms: int = 500,
        max_skew_ms: int = 1500,
        allow_speech_single_source: bool = True,
    ) -> None:
        if decision_wait_ms < 0 or max_skew_ms < 0:
            raise ValueError("multimodal window durations must be non-negative")
        if decision_wait_ms > max_skew_ms:
            raise ValueError("decision wait cannot exceed maximum fusion skew")
        self.decision_wait_ms = decision_wait_ms
        self.max_skew_ms = max_skew_ms
        self.allow_speech_single_source = allow_speech_single_source
        self._context: tuple[str, int, object] | None = None
        self._candidates: dict[str, PlayerActionObservation] = {}

    def add(
        self, observation: PlayerActionObservation
    ) -> PlayerActionObservation | None:
        self._prune(observation.observed_at_ns)
        context = (
            observation.hand_id,
            observation.expected_state_version,
            observation.focus_seat,
        )
        if self._context is None:
            self._context = context
        elif context != self._context:
            raise ValueError("multimodal window context changed without reset")
        if observation.evidence_state is not ActionEvidenceState.CANDIDATE:
            return None
        self._candidates[action_observation_source(observation)] = observation
        if len(self._candidates) < 2:
            return None
        fused = fuse_action_observations(
            tuple(self._candidates.values()), max_skew_ms=self.max_skew_ms
        )
        self.clear()
        return fused

    def poll(self, observed_at_ns: int) -> PlayerActionObservation | None:
        if observed_at_ns < 0:
            raise ValueError("multimodal poll timestamp must be non-negative")
        self._prune(observed_at_ns)
        if not self._candidates:
            return None
        first_ns = min(item.observed_at_ns for item in self._candidates.values())
        if observed_at_ns - first_ns < self.decision_wait_ms * 1_000_000:
            return None
        if (
            not self.allow_speech_single_source
            and set(self._candidates) == {"speech"}
        ):
            return None
        fused = fuse_action_observations(
            tuple(self._candidates.values()), max_skew_ms=self.max_skew_ms
        )
        self.clear()
        return fused

    @property
    def pending_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._candidates))

    def confirm_pending_speech(
        self, observed_at_ns: int
    ) -> PlayerActionObservation | None:
        """Promote one pending speech candidate after explicit UI confirmation."""

        if observed_at_ns < 0:
            raise ValueError("speech confirmation timestamp must be non-negative")
        self._prune(observed_at_ns)
        if set(self._candidates) != {"speech"}:
            return None
        fused = fuse_action_observations(
            tuple(self._candidates.values()), max_skew_ms=self.max_skew_ms
        )
        fused = replace(
            fused,
            observation_id=f"{fused.observation_id}:ui-confirmed",
            quality_flags=tuple(
                dict.fromkeys(fused.quality_flags + ("speech_ui_confirmed",))
            ),
        )
        self.clear()
        return fused

    def clear(self) -> None:
        self._context = None
        self._candidates.clear()

    def cancel_pending_speech(self) -> bool:
        """Remove speech evidence without discarding a target-owned gesture."""

        removed = self._candidates.pop("speech", None) is not None
        if not self._candidates:
            self._context = None
        return removed

    def _prune(self, latest_ns: int) -> None:
        cutoff = latest_ns - self.max_skew_ms * 1_000_000
        self._candidates = {
            source: item
            for source, item in self._candidates.items()
            if item.observed_at_ns >= cutoff
        }
        if not self._candidates:
            self._context = None
