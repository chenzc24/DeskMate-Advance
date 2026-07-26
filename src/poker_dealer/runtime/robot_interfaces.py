"""Expose the active robot-facing input gate for an authoritative hand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from poker_dealer.domain import (
    CardObservation,
    ChipObservation,
    ControlObservation,
    DealerAck,
    DealerTargetSlot,
    FaceEnrollmentObservation,
    HandPhase,
    NavigationAck,
    PlayerActionObservation,
    RobotInputKind,
    RobotWorkflowNode,
    Seat,
    VisionSlot,
)
from poker_dealer.perception.identity import FaceIdentityObservation

from .sequential_part_a import PartAPhase
from .sequential_part_b import PartBMode, PartBPhase
from .registration import RegistrationPhase

if TYPE_CHECKING:
    from .hand_runtime import HandRuntime
    from .registration import RegistrationRuntime
    from .session_runtime import SessionRuntime


_INPUT_TYPES: dict[RobotInputKind, tuple[type[object], ...]] = {
    RobotInputKind.NONE: (),
    RobotInputKind.NAVIGATION_ACK: (NavigationAck,),
    RobotInputKind.LEGACY_DEALER_ACK: (DealerAck,),
    RobotInputKind.DISPENSE_ACK: (DealerAck,),
    RobotInputKind.VISUAL_SETTLE: (bool,),
    RobotInputKind.FACE_ENROLLMENT: (FaceEnrollmentObservation,),
    RobotInputKind.FACE_IDENTITY: (FaceIdentityObservation,),
    RobotInputKind.PLAYER_ACTION: (PlayerActionObservation,),
    RobotInputKind.CHIP_OBSERVATION: (ChipObservation,),
    RobotInputKind.CARD_OBSERVATION: (CardObservation,),
    RobotInputKind.OPERATOR_CONTROL: (ControlObservation,),
}


@dataclass(frozen=True, slots=True)
class RobotInterfaceRequirement:
    """One inspectable state-machine node and its accepted external input."""

    node: RobotWorkflowNode
    accepted_inputs: tuple[RobotInputKind, ...]
    hand_phase: HandPhase | None
    state_version: int
    target_seat: Seat | None = None
    target_slot: DealerTargetSlot | None = None
    vision_slots: tuple[VisionSlot, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state_version < 0:
            raise ValueError("robot interface state_version must be non-negative")
        if not self.accepted_inputs:
            raise ValueError("robot interface must declare at least one input kind")
        if (
            RobotInputKind.NONE in self.accepted_inputs
            and self.accepted_inputs != (RobotInputKind.NONE,)
        ):
            raise ValueError("NONE cannot be combined with an active input kind")
        if len(self.vision_slots) != len(set(self.vision_slots)):
            raise ValueError("robot interface vision slots must be unique")

    @property
    def accepted_python_types(self) -> tuple[type[object], ...]:
        result: list[type[object]] = []
        for kind in self.accepted_inputs:
            for value_type in _INPUT_TYPES[kind]:
                if value_type not in result:
                    result.append(value_type)
        return tuple(result)

    def accepts(self, value: object) -> bool:
        return any(
            isinstance(value, value_type)
            for value_type in self.accepted_python_types
        )


def hand_robot_requirement(runtime: HandRuntime) -> RobotInterfaceRequirement:
    """Return the only external-input gate that may advance the hand."""

    state = runtime.engine.state
    common = {
        "hand_phase": state.phase,
        "state_version": state.state_version,
    }
    if state.phase is HandPhase.PAUSED_RECOVERY:
        return RobotInterfaceRequirement(
            RobotWorkflowNode.WAITING_OPERATOR_CONTROL,
            (RobotInputKind.OPERATOR_CONTROL,),
            reason=state.paused_reason or "operator_recovery_required",
            **common,
        )
    if state.phase is HandPhase.SETTLED:
        return RobotInterfaceRequirement(
            RobotWorkflowNode.COMPLETE,
            (RobotInputKind.NONE,),
            reason="hand_settled",
            **common,
        )
    if state.phase is HandPhase.VOIDED:
        return RobotInterfaceRequirement(
            RobotWorkflowNode.VOIDED,
            (RobotInputKind.NONE,),
            reason="hand_voided",
            **common,
        )

    if runtime.part_a is not None:
        coordinator = runtime.part_a
        target = (
            DealerTargetSlot(coordinator.focus_seat.value)
            if coordinator.focus_seat is not None
            else None
        )
        if coordinator.phase is PartAPhase.WAITING_NAVIGATION_ACK:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_TARGET_ACK,
                (RobotInputKind.NAVIGATION_ACK,),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartAPhase.WAITING_ROTATION_ACK:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_TARGET_ACK,
                (
                    RobotInputKind.NAVIGATION_ACK,
                    RobotInputKind.LEGACY_DEALER_ACK,
                ),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartAPhase.WAITING_VISUAL_SETTLE:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_VISUAL_SETTLE,
                (RobotInputKind.VISUAL_SETTLE,),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartAPhase.VERIFYING_IDENTITY:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_FACE_IDENTITY,
                (RobotInputKind.FACE_IDENTITY,),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartAPhase.WAITING_PLAYER_ACTION:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_PLAYER_ACTION,
                (RobotInputKind.PLAYER_ACTION,),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartAPhase.WAITING_CHIP_OBSERVATION:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_CHIP_OBSERVATION,
                (RobotInputKind.CHIP_OBSERVATION,),
                target_seat=coordinator.focus_seat,
                target_slot=target,
                reason=coordinator.last_reason,
                **common,
            )

    if runtime.part_b is not None:
        coordinator = runtime.part_b
        step = coordinator.current_step
        target = step.target if step is not None else None
        slots = step.vision_slots if step is not None else ()
        if coordinator.phase is PartBPhase.WAITING_NAVIGATION_ACK:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_TARGET_ACK,
                (RobotInputKind.NAVIGATION_ACK,),
                target_slot=target,
                vision_slots=slots,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartBPhase.WAITING_ROTATION_ACK:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_TARGET_ACK,
                (
                    RobotInputKind.NAVIGATION_ACK,
                    RobotInputKind.LEGACY_DEALER_ACK,
                ),
                target_slot=target,
                vision_slots=slots,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartBPhase.WAITING_DISPENSE_ACK:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_DISPENSE_ACK,
                (RobotInputKind.DISPENSE_ACK,),
                target_slot=target,
                vision_slots=slots,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartBPhase.WAITING_VISUAL_CONFIRMATION:
            node = RobotWorkflowNode.WAITING_CARD_OBSERVATION
            if coordinator.mode is PartBMode.BOARD_DEAL:
                node = RobotWorkflowNode.WAITING_BOARD_REVEAL
            elif coordinator.mode is PartBMode.SHOWDOWN_REVEAL:
                node = RobotWorkflowNode.WAITING_SHOWDOWN_CARDS
            return RobotInterfaceRequirement(
                node,
                (RobotInputKind.CARD_OBSERVATION,),
                target_slot=target,
                vision_slots=slots,
                reason=coordinator.last_reason,
                **common,
            )
        if coordinator.phase is PartBPhase.WAITING_POST_BOARD_DELAY:
            return RobotInterfaceRequirement(
                RobotWorkflowNode.WAITING_POST_BOARD_DELAY,
                (RobotInputKind.NONE,),
                target_slot=target,
                vision_slots=slots,
                reason=coordinator.last_reason,
                **common,
            )

    return RobotInterfaceRequirement(
        RobotWorkflowNode.GAME_INTERNAL,
        (RobotInputKind.NONE,),
        reason="no_external_input_required",
        **common,
    )


def registration_robot_requirement(
    runtime: RegistrationRuntime,
) -> RobotInterfaceRequirement:
    """Expose the input gate for consented, pre-hand player registration."""

    common = {
        "hand_phase": None,
        "state_version": runtime.roster_version,
        "target_seat": runtime.focus_seat,
    }
    if runtime.phase is RegistrationPhase.CAPTURING_FACE:
        return RobotInterfaceRequirement(
            RobotWorkflowNode.WAITING_FACE_ENROLLMENT,
            (
                RobotInputKind.FACE_ENROLLMENT,
                RobotInputKind.OPERATOR_CONTROL,
            ),
            reason="face_capture_result_or_cancel_required",
            **common,
        )
    if runtime.phase in {
        RegistrationPhase.READY_FOR_FACE,
        RegistrationPhase.READY_TO_START,
    }:
        return RobotInterfaceRequirement(
            RobotWorkflowNode.WAITING_REGISTRATION_CONTROL,
            (RobotInputKind.OPERATOR_CONTROL,),
            reason=runtime.phase.value,
            **common,
        )
    return RobotInterfaceRequirement(
        RobotWorkflowNode.GAME_INTERNAL,
        (RobotInputKind.NONE,),
        reason="registration_roster_frozen",
        **common,
    )


def session_robot_requirement(runtime: SessionRuntime) -> RobotInterfaceRequirement:
    """Expose the active hand gate or the operator-only between-hand gate."""

    if runtime.active_hand is not None:
        return hand_robot_requirement(runtime.active_hand)
    return RobotInterfaceRequirement(
        (
            RobotWorkflowNode.COMPLETE
            if runtime.ended
            else RobotWorkflowNode.WAITING_SESSION_CONTROL
        ),
        (
            (RobotInputKind.NONE,)
            if runtime.ended
            else (RobotInputKind.OPERATOR_CONTROL,)
        ),
        hand_phase=None,
        state_version=runtime.next_hand_number - 1,
        reason=(
            "session_ended"
            if runtime.ended
            else "start_hand_table_clear_rebuy_or_end_control_required"
        ),
    )


__all__ = [
    "RobotInterfaceRequirement",
    "hand_robot_requirement",
    "registration_robot_requirement",
    "session_robot_requirement",
]
