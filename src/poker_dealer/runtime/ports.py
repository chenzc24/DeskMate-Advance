"""Model-neutral input and event ports for one authoritative hand loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol

from poker_dealer.domain import (
    CardObservation,
    ControlObservation,
    FramePacket,
    HandPhase,
    PlayerActionObservation,
    PlayerActionType,
    Seat,
    VisionSlot,
)
from poker_dealer.perception.attribution import ActorBinding
from poker_dealer.perception.identity import FaceIdentityObservation

from .registration import FrozenSessionRoster


class FrameReadState(StrEnum):
    OK = "ok"
    MISSING = "missing"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class FrameRead:
    state: FrameReadState
    observed_at_ns: int
    frame: FramePacket | None
    camera_epoch: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.observed_at_ns < 0 or self.camera_epoch < 0:
            raise ValueError("frame timestamps and epochs must be non-negative")
        if self.state is FrameReadState.OK and self.frame is None:
            raise ValueError("an OK frame read requires a frame")
        if self.state is not FrameReadState.OK and self.frame is not None:
            raise ValueError("non-OK frame reads cannot carry a frame")


@dataclass(frozen=True, slots=True)
class RuntimeObservationContext:
    session_id: str
    hand_id: str
    state_version: int
    hand_phase: HandPhase
    focus_seat: Seat | None
    legal_actions: tuple[PlayerActionType, ...]
    required_card_slots: tuple[VisionSlot, ...]
    camera_epoch: int = 0


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    observation: PlayerActionObservation
    actor_binding: ActorBinding | None = None
    attribution_source: str | None = None
    attribution_confidence: float | None = None
    quality_flags: tuple[str, ...] = ()
    identity_revocation_reason: str | None = None

    def __post_init__(self) -> None:
        if self.actor_binding is None:
            if self.attribution_source is not None or self.attribution_confidence is not None:
                raise ValueError("attribution metadata requires an actor binding")
            if self.identity_revocation_reason is not None and not self.identity_revocation_reason.strip():
                raise ValueError("identity revocation reason must not be blank")
            return
        if self.identity_revocation_reason is not None:
            raise ValueError("bound action evidence cannot also revoke identity")
        if not (self.attribution_source or "").strip():
            raise ValueError("bound action evidence requires an attribution source")
        if self.attribution_confidence is None or not 0.0 <= self.attribution_confidence <= 1.0:
            raise ValueError("bound action attribution confidence must be in [0, 1]")
        if not self.actor_binding.matches_observation(self.observation):
            raise ValueError("actor binding does not match the action observation")


class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> FrameRead: ...

    def close(self) -> None: ...


class RegistrationSource(Protocol):
    def acquire_roster(
        self,
        *,
        frame_source: FrameSource,
        control_source: ControlSource,
        event_sink: RuntimeEventSink,
        session_id: str,
        button: Seat,
        deadline_ns: int,
    ) -> FrozenSessionRoster: ...


class IdentitySource(Protocol):
    def observe_identity(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> FaceIdentityObservation | None: ...


class ActionSource(Protocol):
    def observe_action(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> ActionEvidence | None: ...


class CardSource(Protocol):
    def observe_card(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        slot: VisionSlot,
        observed_at_ns: int,
    ) -> CardObservation | None: ...


class VisualSettleSource(Protocol):
    def reset_visual_settle(self, context: RuntimeObservationContext) -> None: ...

    def visual_is_settled(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> bool | None: ...


class ControlSource(Protocol):
    def poll_controls(self, observed_at_ns: int) -> tuple[ControlObservation, ...]: ...


class RuntimeEventSink(Protocol):
    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None: ...


class NullControlSource:
    def poll_controls(self, observed_at_ns: int) -> tuple[ControlObservation, ...]:
        del observed_at_ns
        return ()


class NullEventSink:
    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        del kind, observed_at_ns, payload


__all__ = [
    "ActionEvidence",
    "ActionSource",
    "CardSource",
    "ControlSource",
    "FrameRead",
    "FrameReadState",
    "FrameSource",
    "IdentitySource",
    "NullControlSource",
    "NullEventSink",
    "RegistrationSource",
    "RuntimeEventSink",
    "RuntimeObservationContext",
    "VisualSettleSource",
]
