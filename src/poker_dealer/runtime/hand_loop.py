"""Bounded single-frame event loop for one authoritative ``HandRuntime``."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TYPE_CHECKING, TypeVar

from poker_dealer.domain import (
    ChipObservation,
    HandPhase,
    NavigationAck,
    NavigationCommand,
    RobotPoseNode,
)
from poker_dealer.game import SlotLifecycle
from poker_dealer.io.camera import CameraRoute
from poker_dealer.perception.actions import observation_to_dict
from poker_dealer.perception.attribution import (
    AttributedActionCandidate,
    actor_binding_to_dict,
)
from poker_dealer.perception.cards import card_observation_to_dict
from poker_dealer.perception.identity import identity_observation_to_dict
from poker_dealer.robotics.dealer import DealerPort
from poker_dealer.robotics.navigation import (
    DEFAULT_INTER_MOTION_DELAY_MS,
    NavigationPort,
    NavigationTimingGate,
)

from .event_log import RuntimeEventWriter
from .hand_runtime import HandRuntime
from .ports import (
    ActionSource,
    CardSource,
    ChipSource,
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

if TYPE_CHECKING:
    from .diagnostics import DiagnosticSink


Clock = Callable[[], int]
T = TypeVar("T")


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
        chip_source: ChipSource | None = None,
        navigation_port: NavigationPort | None = None,
        frame_source: FrameSource | None = None,
        visual_settle_source: VisualSettleSource | None = None,
        control_source: ControlSource | None = None,
        clock_ns: Clock = time.monotonic_ns,
        diagnostic_sink: DiagnosticSink | None = None,
        state_observer: object | None = None,
        inter_motion_delay_ms: int = DEFAULT_INTER_MOTION_DELAY_MS,
        cooldown_poll_interval_ms: int = 50,
    ) -> None:
        if cooldown_poll_interval_ms <= 0:
            raise ValueError("cooldown poll interval must be positive")
        self.runtime = runtime
        self.dealer = dealer
        self.identity_source = identity_source
        self.action_source = action_source
        self.card_source = card_source
        self.chip_source = chip_source
        self.navigation_port = navigation_port
        self.event_writer = event_writer
        self.frame_source = frame_source
        self.visual_settle_source = visual_settle_source
        self.control_source = control_source
        self.clock_ns = clock_ns
        self.diagnostic_sink = diagnostic_sink
        self.state_observer = state_observer
        self._navigation_timing = NavigationTimingGate(inter_motion_delay_ms)
        self._cooldown_poll_interval_ms = cooldown_poll_interval_ms
        self.steps = 0
        self._camera_epoch = 0
        self._last_frame_read: FrameRead | None = None
        self._last_frame_observed_at_ns: dict[CameraRoute, int] = {}
        self._active_camera_route = CameraRoute.PLAYER
        self.event_writer.sync_engine(self.runtime.engine.log)
        self._publish_state()

    def context(self) -> RuntimeObservationContext:
        required_slots = ()
        if self.runtime.part_b is not None:
            step = self.runtime.part_b.current_step
            required_slots = step.vision_slots if step is not None else ()
        state = self.runtime.engine.state
        requirement = self.runtime.robot_requirement()
        robot_pose = RobotPoseNode.UNKNOWN
        robot_pose_version = 0
        if self.navigation_port is not None:
            health = self.navigation_port.health()
            robot_pose = health.pose
            robot_pose_version = health.pose_version
        return RuntimeObservationContext(
            session_id=self.runtime.session_id,
            hand_id=state.hand_id,
            state_version=state.state_version,
            hand_phase=state.phase,
            focus_seat=state.acting_seat,
            legal_actions=state.legal_actions,
            required_card_slots=required_slots,
            camera_epoch=self._camera_epoch,
            robot_node=requirement.node,
            expected_robot_inputs=requirement.accepted_inputs,
            robot_pose=robot_pose,
            robot_pose_version=robot_pose_version,
        )

    def run(self, *, max_steps: int = 10_000) -> HandLoopResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.diagnostic_sink is not None:
            self.diagnostic_sink.emit(
                "hand_loop_started",
                {**self._diagnostic_context(), "max_steps": max_steps},
            )
        while self.steps < max_steps:
            phase = self.runtime.phase
            if phase is HandPhase.SETTLED:
                return self._diagnostic_result(True, "hand_settled")
            if phase in {HandPhase.PAUSED_RECOVERY, HandPhase.VOIDED}:
                return self._diagnostic_result(False, phase.value)
            try:
                self.step()
                remaining_ms = self.navigation_cooldown_remaining_ms()
                if remaining_ms > 0 and self._navigation_is_pending():
                    time.sleep(
                        min(remaining_ms, self._cooldown_poll_interval_ms) / 1000
                    )
                elif (
                    self.runtime.part_b is not None
                    and self.runtime.part_b.phase
                    is PartBPhase.WAITING_POST_BOARD_DELAY
                ):
                    delay_ms = self.runtime.part_b.post_board_delay_remaining_ms(
                        self.clock_ns()
                    )
                    if delay_ms > 0:
                        time.sleep(
                            min(delay_ms, self._cooldown_poll_interval_ms) / 1000
                        )
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
                if self.diagnostic_sink is not None:
                    self.diagnostic_sink.emit(
                        "hand_loop_exception",
                        {
                            **self._diagnostic_context(),
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                        },
                        level="error",
                    )
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
        return self._diagnostic_result(False, "max_steps_reached")

    def navigation_cooldown_remaining_ms(self, now_ns: int | None = None) -> int:
        """Return the physical-navigation delay without blocking the caller."""

        port = self.navigation_port
        if port is None or not port.physical_motion:
            return 0
        observed_at_ns = self.clock_ns() if now_ns is None else now_ns
        return self._navigation_timing.remaining_ms(observed_at_ns)

    def _navigation_is_pending(self) -> bool:
        if self.runtime.part_a is not None:
            return self.runtime.part_a.phase in {
                PartAPhase.WAITING_ROTATION_ACK,
                PartAPhase.WAITING_NAVIGATION_ACK,
            }
        if self.runtime.part_b is not None:
            return self.runtime.part_b.phase in {
                PartBPhase.WAITING_ROTATION_ACK,
                PartBPhase.WAITING_NAVIGATION_ACK,
            }
        return False

    def step(self) -> None:
        started_ns = time.monotonic_ns()
        self.steps += 1
        before = self._diagnostic_context()
        try:
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
        finally:
            self._publish_state()
            if self.diagnostic_sink is not None:
                elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000
                self.diagnostic_sink.metric(
                    "runtime_step_duration",
                    elapsed_ms,
                    {
                        **before,
                        "after_state_version": self.runtime.engine.state.state_version,
                        "after_hand_phase": self.runtime.phase.value,
                    },
                )

    def _publish_state(self) -> None:
        if self.state_observer is None:
            return
        publish = getattr(self.state_observer, "publish_hand_state", None)
        if publish is not None:
            publish(self.runtime)

    def _step_part_b(self, now_ns: int) -> None:
        coordinator = self.runtime.part_b
        assert coordinator is not None
        if coordinator.phase in {
            PartBPhase.WAITING_ROTATION_ACK,
            PartBPhase.WAITING_NAVIGATION_ACK,
        } and self.navigation_port is not None:
            self._execute_navigation(now_ns)
            return
        if coordinator.phase is PartBPhase.WAITING_ROTATION_ACK:
            command = self.runtime.request_rotation(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self._measure(
                "dealer_command_duration",
                {
                    **self._diagnostic_context(),
                    "command_id": command.command_id,
                    "command": command.command.value,
                    "target_slot": (
                        command.target_slot.value if command.target_slot else None
                    ),
                },
                lambda: self.dealer.execute(
                    command, observed_at_ns=self.clock_ns()
                ),
            )
            self.runtime.accept_rotation_ack(ack)
            return
        if coordinator.phase is PartBPhase.WAITING_DISPENSE_ACK:
            command = self.runtime.request_dispense(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self._measure(
                "dealer_command_duration",
                {
                    **self._diagnostic_context(),
                    "command_id": command.command_id,
                    "command": command.command.value,
                    "target_slot": None,
                },
                lambda: self.dealer.execute(
                    command, observed_at_ns=self.clock_ns()
                ),
            )
            self.runtime.accept_dispense_ack(ack)
            return
        if coordinator.phase is PartBPhase.WAITING_POST_BOARD_DELAY:
            self.runtime.advance_post_board_delay(now_ns)
            return
        if coordinator.phase is not PartBPhase.WAITING_VISUAL_CONFIRMATION:
            raise RuntimeError(f"unsupported Part B phase: {coordinator.phase.value}")
        frame = self._read_frame(now_ns, CameraRoute.TABLE)
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
            observation = self._measure(
                "card_observation_duration",
                {**self._diagnostic_context(), "slot": slot.value},
                lambda: self.card_source.observe_card(
                    frame, context, slot, now_ns
                ),
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
        if coordinator.phase in {
            PartAPhase.WAITING_ROTATION_ACK,
            PartAPhase.WAITING_NAVIGATION_ACK,
        } and self.navigation_port is not None:
            accepted = self._execute_navigation(now_ns)
            if (
                accepted
                and self.runtime.part_a is not None
                and self.runtime.part_a.phase is PartAPhase.WAITING_VISUAL_SETTLE
            ):
                if self.visual_settle_source is None:
                    raise RuntimeError("live Part A requires a visual-settle source")
                self.visual_settle_source.reset_visual_settle(self.context())
            return
        if coordinator.phase is PartAPhase.WAITING_ROTATION_ACK:
            command = self.runtime.request_rotation(now_ns)
            self.event_writer.sync_engine(self.runtime.engine.log)
            ack = self._measure(
                "dealer_command_duration",
                {
                    **self._diagnostic_context(),
                    "command_id": command.command_id,
                    "command": command.command.value,
                    "target_slot": (
                        command.target_slot.value if command.target_slot else None
                    ),
                },
                lambda: self.dealer.execute(
                    command, observed_at_ns=self.clock_ns()
                ),
            )
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
            frame = self._read_frame(now_ns, CameraRoute.PLAYER)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            settled = self._measure(
                "visual_settle_duration",
                self._diagnostic_context(),
                lambda: self.visual_settle_source.visual_is_settled(
                    frame, context, now_ns
                ),
            )
            if settled is True:
                self.runtime.accept_visual_settle()
            return
        if coordinator.phase is PartAPhase.VERIFYING_IDENTITY:
            frame = self._read_frame(now_ns, CameraRoute.PLAYER)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            observation = self._measure(
                "identity_observation_duration",
                self._diagnostic_context(),
                lambda: self.identity_source.observe_identity(
                    frame, context, now_ns
                ),
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
            frame = self._read_frame(now_ns, CameraRoute.PLAYER)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            evidence = self._measure(
                "action_observation_duration",
                self._diagnostic_context(),
                lambda: self.action_source.observe_action(
                    frame, context, now_ns
                ),
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
            if not evidence.actor_binding.is_valid_at(observation.observed_at_ns):
                self.event_writer.emit(
                    "player_action_observation_rejected",
                    observed_at_ns=now_ns,
                    payload={
                        "observation_id": observation.observation_id,
                        "binding_id": evidence.actor_binding.binding_id,
                        "reason": "actor_binding_time_mismatch",
                        "observation_observed_at_ns": observation.observed_at_ns,
                        "binding_verified_at_ns": (
                            evidence.actor_binding.verified_at_ns
                        ),
                        "binding_valid_until_ns": (
                            evidence.actor_binding.valid_until_ns
                        ),
                    },
                )
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
        if coordinator.phase is PartAPhase.WAITING_CHIP_OBSERVATION:
            if self.chip_source is None:
                raise RuntimeError(
                    "chip-gated bet/raise requires a ChipSource implementation"
                )
            frame = self._read_frame(now_ns, CameraRoute.PLAYER)
            if self.runtime.phase is HandPhase.PAUSED_RECOVERY:
                return
            context = self.context()
            self._dispatch_controls(now_ns, context)
            observation = self._measure(
                "chip_observation_duration",
                self._diagnostic_context(),
                lambda: self.chip_source.observe_chips(frame, context, now_ns),
            )
            if observation is None:
                return
            self.event_writer.emit(
                "chip_observation",
                observed_at_ns=observation.observed_at_ns,
                payload=_chip_observation_to_dict(observation),
            )
            self.runtime.accept_chip_observation(observation)
            return
        raise RuntimeError(f"unsupported Part A phase: {coordinator.phase.value}")

    def _execute_navigation(self, now_ns: int) -> bool:
        port = self.navigation_port
        if port is None:
            raise RuntimeError("navigation interface is not configured")
        if port.physical_motion and not self._navigation_timing.can_start(now_ns):
            return False
        health = port.health()
        if not health.available or not health.opened:
            raise RuntimeError(health.reason or "navigation interface is not open")
        if health.pose is RobotPoseNode.UNKNOWN:
            raise RuntimeError("navigation interface reported an unknown robot pose")
        command = self.runtime.request_navigation(
            now_ns,
            start_pose=health.pose,
            expected_pose_version=health.pose_version,
            inter_motion_delay_ms=self._navigation_timing.inter_motion_delay_ms,
        )
        self.event_writer.emit(
            "navigation_command",
            observed_at_ns=command.issued_at_ns,
            payload=_navigation_command_to_dict(command),
        )
        ack = self._measure(
            "navigation_command_duration",
            {
                **self._diagnostic_context(),
                "command_id": command.command_id,
                "target_slot": (
                    command.target_slot.value if command.target_slot else None
                ),
            },
            lambda: port.execute(command, observed_at_ns=self.clock_ns()),
        )
        self.event_writer.emit(
            "navigation_ack",
            observed_at_ns=ack.observed_at_ns,
            payload=_navigation_ack_to_dict(ack),
        )
        accepted = self.runtime.accept_navigation_ack(ack)
        if accepted:
            if port.physical_motion:
                self._navigation_timing.record_success(ack.observed_at_ns)
                self.event_writer.emit(
                    "navigation_inter_motion_delay_started",
                    observed_at_ns=ack.observed_at_ns,
                    payload={
                        "command_id": command.command_id,
                        "inter_motion_delay_ms": (
                            command.inter_motion_delay_ms
                        ),
                        "next_motion_not_before_ns": (
                            self._navigation_timing.next_motion_not_before_ns
                        ),
                    },
                )
            self.dealer.confirm_navigation_target(ack)
        return accepted

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

    def _read_frame(self, now_ns: int, camera_route: CameraRoute):
        if self.frame_source is None:
            return None
        self._active_camera_route = camera_route
        selector = getattr(self.frame_source, "select_camera_route", None)
        if selector is None:
            selector = getattr(self.frame_source, "select_route", None)
        if selector is not None:
            selector(camera_route)
        read = self._measure(
            "camera_read_duration",
            {
                **self._diagnostic_context(),
                "camera_route": camera_route.value,
            },
            self.frame_source.read,
        )
        self._last_frame_read = read
        self._camera_epoch = read.camera_epoch
        previous_observed_at_ns = self._last_frame_observed_at_ns.get(camera_route)
        if previous_observed_at_ns is not None and self.diagnostic_sink is not None:
            self.diagnostic_sink.metric(
                "camera_frame_interval",
                (read.observed_at_ns - previous_observed_at_ns) / 1_000_000,
                {
                    **self._diagnostic_context(),
                    "camera_route": camera_route.value,
                    "read_state": read.state.value,
                },
            )
        self._last_frame_observed_at_ns[camera_route] = read.observed_at_ns
        if read.state is FrameReadState.DISCONNECTED:
            if self.diagnostic_sink is not None:
                self.diagnostic_sink.emit(
                    "camera_disconnected",
                    {
                        **self._diagnostic_context(),
                        "camera_route": camera_route.value,
                        "reason": read.reason,
                    },
                    level="error",
                )
            self.runtime.engine.pause(
                f"runtime:{self.runtime.engine.state.hand_id}:camera-disconnected:{self.steps}",
                "camera_disconnected",
            )
            self.runtime.sync()
            return None
        if read.state is FrameReadState.MISSING:
            if self.diagnostic_sink is not None:
                self.diagnostic_sink.emit(
                    "camera_frame_missing",
                    {
                        **self._diagnostic_context(),
                        "camera_route": camera_route.value,
                        "reason": read.reason,
                    },
                    level="warning",
                )
            self.event_writer.emit(
                "camera_read",
                observed_at_ns=read.observed_at_ns,
                payload={
                    "state": read.state.value,
                    "camera_route": camera_route.value,
                    "reason": read.reason,
                },
            )
            return None
        return read.frame

    def _measure(
        self,
        name: str,
        context: dict[str, object],
        operation: Callable[[], T],
    ) -> T:
        started_ns = time.monotonic_ns()
        try:
            return operation()
        finally:
            if self.diagnostic_sink is not None:
                self.diagnostic_sink.metric(
                    name,
                    (time.monotonic_ns() - started_ns) / 1_000_000,
                    context,
                )

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

    def _diagnostic_result(self, completed: bool, reason: str) -> HandLoopResult:
        result = self._result(completed, reason)
        if self.diagnostic_sink is not None:
            self.diagnostic_sink.emit(
                "hand_loop_finished",
                {
                    **self._diagnostic_context(),
                    "completed": completed,
                    "reason": reason,
                },
                level="info" if completed else "warning",
            )
        return result

    def _diagnostic_context(self) -> dict[str, object]:
        state = self.runtime.engine.state
        part_a_phase = (
            self.runtime.part_a.phase.value if self.runtime.part_a is not None else None
        )
        part_b_phase = (
            self.runtime.part_b.phase.value if self.runtime.part_b is not None else None
        )
        return {
            "session_id": self.runtime.session_id,
            "hand_id": state.hand_id,
            "step": self.steps,
            "state_version": state.state_version,
            "hand_phase": state.phase.value,
            "acting_seat": state.acting_seat.value if state.acting_seat else None,
            "part_a_phase": part_a_phase,
            "part_b_phase": part_b_phase,
            "camera_epoch": self._camera_epoch,
            "camera_route": self._active_camera_route.value,
        }


def _navigation_command_to_dict(command: NavigationCommand) -> dict[str, object]:
    return {
        "command_id": command.command_id,
        "session_id": command.session_id,
        "hand_id": command.hand_id,
        "expected_state_version": command.expected_state_version,
        "expected_pose_version": command.expected_pose_version,
        "issued_at_ns": command.issued_at_ns,
        "action": command.action.value,
        "start_pose": command.start_pose.value,
        "target_pose": command.target_pose.value,
        "target_slot": command.target_slot.value if command.target_slot else None,
        "timeout_ms": command.timeout_ms,
        "inter_motion_delay_ms": command.inter_motion_delay_ms,
    }


def _navigation_ack_to_dict(ack: NavigationAck) -> dict[str, object]:
    return {
        "command_id": ack.command_id,
        "session_id": ack.session_id,
        "hand_id": ack.hand_id,
        "expected_state_version": ack.expected_state_version,
        "action": ack.action.value,
        "target_slot": ack.target_slot.value if ack.target_slot else None,
        "status": ack.status.value,
        "observed_at_ns": ack.observed_at_ns,
        "actual_pose": ack.actual_pose.value,
        "pose_version": ack.pose_version,
        "pose_confidence": ack.pose_confidence,
        "line_locked": ack.line_locked,
        "endpoint_confirmed": ack.endpoint_confirmed,
        "target_aligned": ack.target_aligned,
        "stable_frames": ack.stable_frames,
        "face_center_error_px": ack.face_center_error_px,
        "error_code": ack.error_code.value if ack.error_code else None,
        "reason": ack.reason,
    }


def _chip_observation_to_dict(observation: ChipObservation) -> dict[str, object]:
    return {
        "observation_id": observation.observation_id,
        "hand_id": observation.hand_id,
        "expected_state_version": observation.expected_state_version,
        "focus_seat": observation.focus_seat.value,
        "observed_at_ns": observation.observed_at_ns,
        "status": observation.status.value,
        "amount_scope": observation.amount_scope.value,
        "chip_counts": [
            {
                "denomination_units": item.denomination_units,
                "count": item.count,
            }
            for item in observation.chip_counts
        ],
        "total_units": observation.total_units,
        "confidence": observation.confidence,
        "stable_frames": observation.stable_frames,
        "model_version": observation.model_version,
        "calibration_version": observation.calibration_version,
        "quality_flags": list(observation.quality_flags),
    }


__all__ = ["HandLoopResult", "HandRuntimeLoop"]
