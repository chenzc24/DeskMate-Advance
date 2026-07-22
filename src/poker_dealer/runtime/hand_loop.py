"""Bounded single-frame event loop for one authoritative ``HandRuntime``."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from poker_dealer.domain import HandPhase
from poker_dealer.game import SlotLifecycle
from poker_dealer.perception.actions import observation_to_dict
from poker_dealer.perception.attribution import (
    AttributedActionCandidate,
    actor_binding_to_dict,
)
from poker_dealer.perception.cards import card_observation_to_dict
from poker_dealer.perception.identity import identity_observation_to_dict
from poker_dealer.robotics.dealer import DealerPort

from .event_log import RuntimeEventWriter
from .hand_runtime import HandRuntime
from .ports import (
    ActionSource,
    CardSource,
    ControlSource,
    FrameRead,
    FrameReadState,
    FrameSource,
    IdentitySource,
    RuntimeObservationContext,
    VisualSettleSource,
)
from .sequential_part_a import PartAPhase
from .sequential_part_b import PartBMode, PartBPhase


Clock = Callable[[], int]


@dataclass(frozen=True, slots=True)
class HandLoopResult:
    completed: bool
    reason: str
    hand_phase: HandPhase
    steps: int
    state_version: int


class HandRuntimeLoop:
    """Coordinate ports without granting any model game-state authority."""

    def __init__(
        self,
        runtime: HandRuntime,
        dealer: DealerPort,
        *,
        identity_source: IdentitySource,
        action_source: ActionSource,
        card_source: CardSource,
        event_writer: RuntimeEventWriter,
        frame_source: FrameSource | None = None,
        visual_settle_source: VisualSettleSource | None = None,
        control_source: ControlSource | None = None,
        clock_ns: Clock = time.monotonic_ns,
    ) -> None:
        self.runtime = runtime
        self.dealer = dealer
        self.identity_source = identity_source
        self.action_source = action_source
        self.card_source = card_source
        self.event_writer = event_writer
        self.frame_source = frame_source
        self.visual_settle_source = visual_settle_source
        self.control_source = control_source
        self.clock_ns = clock_ns
        self.steps = 0
        self._camera_epoch = 0
        self._last_frame_read: FrameRead | None = None
        self.event_writer.sync_engine(self.runtime.engine.log)

    def context(self) -> RuntimeObservationContext:
        required_slots = ()
        if self.runtime.part_b is not None:
            step = self.runtime.part_b.current_step
            required_slots = step.vision_slots if step is not None else ()
        state = self.runtime.engine.state
        return RuntimeObservationContext(
            session_id=self.runtime.session_id,
            hand_id=state.hand_id,
            state_version=state.state_version,
            hand_phase=state.phase,
            focus_seat=state.acting_seat,
            legal_actions=state.legal_actions,
            required_card_slots=required_slots,
            camera_epoch=self._camera_epoch,
        )

    def run(self, *, max_steps: int = 10_000) -> HandLoopResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        while self.steps < max_steps:
            phase = self.runtime.phase
            if phase is HandPhase.SETTLED:
                return self._result(True, "hand_settled")
            if phase in {HandPhase.PAUSED_RECOVERY, HandPhase.VOIDED}:
                return self._result(False, phase.value)
            try:
                self.step()
            except Exception as exc:
                if self.runtime.phase not in {
                    HandPhase.PAUSED_RECOVERY,
                    HandPhase.SETTLED,
                    HandPhase.VOIDED,
                }:
                    self.runtime.engine.pause(
                        f"runtime:{self.runtime.engine.state.hand_id}:exception:{self.steps}",
                        f"runtime_loop_exception:{type(exc).__name__}",
                    )
                    self.runtime.sync()
                    self.event_writer.sync_engine(self.runtime.engine.log)
                raise
        if self.runtime.phase not in {
            HandPhase.PAUSED_RECOVERY,
            HandPhase.SETTLED,
            HandPhase.VOIDED,
        }:
            self.runtime.engine.pause(
                f"runtime:{self.runtime.engine.state.hand_id}:step-budget:{self.steps}",
                "runtime_loop_step_budget_exhausted",
            )
            self.runtime.sync()
        return self._result(False, "max_steps_reached")

    def step(self) -> None:
        self.steps += 1
        now_ns = self.clock_ns()
        if self.runtime.check_timeout(now_ns):
            self.event_writer.sync_engine(self.runtime.engine.log)
            return
        if self.runtime.part_b is not None:
            self._step_part_b(now_ns)
        elif self.runtime.part_a is not None:
            self._step_part_a(now_ns)
        else:
            raise RuntimeError(
                f"no active runtime lane in {self.runtime.phase.value}"
            )
        self.event_writer.sync_engine(self.runtime.engine.log)

    def _step_part_b(self, now_ns: int) -> None:
        coordinator = self.runtime.part_b
        assert coordinator is not None
        if coordinator.phase is PartBPhase.WAITING_ROTATION_ACK:
            command = self.runtime.request_rotation(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self.dealer.execute(command, observed_at_ns=self.clock_ns())
            self.runtime.accept_rotation_ack(ack)
            return
        if coordinator.phase is PartBPhase.WAITING_DISPENSE_ACK:
            command = self.runtime.request_dispense(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self.dealer.execute(command, observed_at_ns=self.clock_ns())
            self.runtime.accept_dispense_ack(ack)
            return
        if coordinator.phase is not PartBPhase.WAITING_VISUAL_CONFIRMATION:
            raise RuntimeError(f"unsupported Part B phase: {coordinator.phase.value}")
        frame = self._read_frame(now_ns)
        if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
            return
        step = coordinator.current_step
        assert step is not None
        expected = (
            SlotLifecycle.PRESENT_FACE_DOWN
            if coordinator.mode is PartBMode.HOLE_DEAL
            else SlotLifecycle.CONFIRMED
        )
        for slot in step.vision_slots:
            if self.runtime.engine.state.slot_states[slot] is expected:
                continue
            context = self.context()
            self._dispatch_controls(now_ns, context)
            observation = self.card_source.observe_card(
                frame, context, slot, now_ns
            )
            if observation is None:
                return
            self.event_writer.emit(
                "card_observation",
                observed_at_ns=observation.observed_at_ns,
                payload=card_observation_to_dict(observation),
            )
            self.runtime.accept_card_observation(observation)

    def _step_part_a(self, now_ns: int) -> None:
        coordinator = self.runtime.part_a
        assert coordinator is not None
        if coordinator.phase is PartAPhase.WAITING_ROTATION_ACK:
            command = self.runtime.request_rotation(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self.dealer.execute(command, observed_at_ns=self.clock_ns())
            accepted = self.runtime.accept_rotation_ack(ack)
            if (
                accepted
                and self.runtime.part_a is not None
                and self.runtime.part_a.phase is PartAPhase.WAITING_VISUAL_SETTLE
            ):
                if self.visual_settle_source is None:
                    raise RuntimeError("live Part A requires a visual-settle source")
                self.visual_settle_source.reset_visual_settle(self.context())
            return
        if coordinator.phase is PartAPhase.WAITING_VISUAL_SETTLE:
            if self.visual_settle_source is None:
                raise RuntimeError("live Part A requires a visual-settle source")
            frame = self._read_frame(now_ns)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            settled = self.visual_settle_source.visual_is_settled(
                frame, context, now_ns
            )
            if settled is True:
                self.runtime.accept_visual_settle()
            return
        if coordinator.phase is PartAPhase.VERIFYING_IDENTITY:
            frame = self._read_frame(now_ns)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            observation = self.identity_source.observe_identity(
                frame, context, now_ns
            )
            if observation is None:
                return
            self.event_writer.emit(
                "face_identity_observation",
                observed_at_ns=observation.observed_at_ns,
                payload=identity_observation_to_dict(observation),
            )
            self.runtime.accept_identity(observation)
            return
        if coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION:
            frame = self._read_frame(now_ns)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            evidence = self.action_source.observe_action(
                frame, context, now_ns
            )
            if evidence is None:
                return
            observation = evidence.observation
            payload = observation_to_dict(observation)
            if evidence.actor_binding is not None:
                payload = {
                    **payload,
                    "actor_binding": actor_binding_to_dict(evidence.actor_binding),
                    "attribution_source": evidence.attribution_source,
                    "attribution_confidence": evidence.attribution_confidence,
                    "attribution_quality_flags": list(evidence.quality_flags),
                }
            self.event_writer.emit(
                "player_action_observation",
                observed_at_ns=observation.observed_at_ns,
                payload=payload,
            )
            if evidence.identity_revocation_reason is not None:
                coordinator.revoke_identity(evidence.identity_revocation_reason)
                return
            if evidence.actor_binding is None:
                self.runtime.accept_action(observation)
                return
            active = coordinator.active_actor_binding
            if active is None or active.binding_id != evidence.actor_binding.binding_id:
                self.runtime.bind_actor(evidence.actor_binding)
            self.runtime.accept_attributed_action(
                AttributedActionCandidate(
                    observation=observation,
                    binding=evidence.actor_binding,
                    attribution_source=evidence.attribution_source or "unknown",
                    attribution_confidence=evidence.attribution_confidence or 0.0,
                    quality_flags=evidence.quality_flags,
                )
            )
            return
        raise RuntimeError(f"unsupported Part A phase: {coordinator.phase.value}")

    def _dispatch_controls(
        self, observed_at_ns: int, context: RuntimeObservationContext
    ) -> None:
        controls = (
            self.control_source.poll_controls(observed_at_ns)
            if self.control_source is not None
            else ()
        )
        for control in controls:
            self.event_writer.emit(
                "runtime_control",
                observed_at_ns=control.observed_at_ns,
                payload={
                    "observation_id": control.observation_id,
                    "intent": control.intent.value,
                    "source": control.source.value,
                    "control_id": control.control_id,
                    "device_state_version": control.device_state_version,
                    "hand_phase": context.hand_phase.value,
                },
            )
        consumers = {
            id(source): source
            for source in (self.identity_source, self.action_source, self.card_source)
        }
        for source in consumers.values():
            handler = getattr(source, "accept_runtime_controls", None)
            if handler is not None:
                handler(controls, context)

    def _read_frame(self, now_ns: int):
        if self.frame_source is None:
            return None
        read = self.frame_source.read()
        self._last_frame_read = read
        self._camera_epoch = read.camera_epoch
        if read.state is FrameReadState.DISCONNECTED:
            self.runtime.engine.pause(
                f"runtime:{self.runtime.engine.state.hand_id}:camera-disconnected:{self.steps}",
                "camera_disconnected",
            )
            self.runtime.sync()
            return None
        if read.state is FrameReadState.MISSING:
            self.event_writer.emit(
                "camera_read",
                observed_at_ns=read.observed_at_ns,
                payload={"state": read.state.value, "reason": read.reason},
            )
            return None
        return read.frame

    def _result(self, completed: bool, reason: str) -> HandLoopResult:
        self.event_writer.sync_engine(self.runtime.engine.log)
        state = self.runtime.engine.state
        return HandLoopResult(
            completed=completed,
            reason=reason,
            hand_phase=state.phase,
            steps=self.steps,
            state_version=state.state_version,
        )


__all__ = ["HandLoopResult", "HandRuntimeLoop"]
