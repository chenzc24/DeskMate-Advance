"""ACK- and vision-gated no-burn card delivery coordination."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from poker_dealer.domain import (
    CardObservation,
    DealerAck,
    DealerAckStatus,
    DealerCommand,
    DealerCommandType,
    DealerTargetSlot,
    HandPhase,
    ObservationStatus,
    SEAT_ORDER,
    Seat,
    VisionSlot,
    board_deal_targets,
    clockwise_order_after,
    hole_deal_targets,
)
from poker_dealer.game import (
    CardObservationResult,
    HandEngine,
    HandRank,
    HOLE_SLOTS,
    SlotLifecycle,
)


class PartBMode(StrEnum):
    HOLE_DEAL = "hole_deal"
    BOARD_DEAL = "board_deal"
    SHOWDOWN_REVEAL = "showdown_reveal"


class PartBPhase(StrEnum):
    WAITING_ROTATION_ACK = "waiting_rotation_ack"
    WAITING_DISPENSE_ACK = "waiting_dispense_ack"
    WAITING_VISUAL_CONFIRMATION = "waiting_visual_confirmation"
    COMPLETE = "complete"
    RECOVERY_REQUIRED = "recovery_required"


@dataclass(frozen=True, slots=True)
class PartBStep:
    target: DealerTargetSlot
    vision_slots: tuple[VisionSlot, ...]
    dispense: bool


class SequentialPartBCoordinator:
    """Advance one physical/visual card step at a time.

    The coordinator owns command correlation and a bounded local cursor. The
    engine remains the sole owner of hand phase, slot lifecycle and settlement.
    """

    def __init__(
        self,
        engine: HandEngine,
        *,
        command_timeout_ms: int = 5000,
        visual_timeout_ms: int = 5000,
    ) -> None:
        if command_timeout_ms <= 0 or visual_timeout_ms <= 0:
            raise ValueError("Part B timeouts must be positive")
        self.engine = engine
        self.command_timeout_ms = command_timeout_ms
        self.visual_timeout_ms = visual_timeout_ms
        self.mode, self.steps = self._build_steps()
        if not self.steps:
            raise ValueError("Part B requires at least one delivery or reveal step")
        self.phase = PartBPhase.WAITING_ROTATION_ACK
        self.step_index = 0
        self.pending_command: DealerCommand | None = None
        self.visual_window_opened_at_ns: int | None = None
        self.last_reason = "rotation_not_requested"
        self.showdown_ranks: Mapping[Seat, HandRank] | None = None
        self._command_sequence = 0
        self._accepted_ack_ids: set[str] = set()
        self._restore_cursor()

    @property
    def current_step(self) -> PartBStep | None:
        if self.phase in {PartBPhase.COMPLETE, PartBPhase.RECOVERY_REQUIRED}:
            return None
        return self.steps[self.step_index]

    def _build_steps(self) -> tuple[PartBMode, tuple[PartBStep, ...]]:
        state = self.engine.state
        if state.phase is HandPhase.DEALING_HOLE:
            targets = hole_deal_targets(state.button)
            seats_per_round = len(SEAT_ORDER)
            steps = tuple(
                PartBStep(
                    target,
                    (HOLE_SLOTS[Seat(target.value)][index // seats_per_round],),
                    True,
                )
                for index, target in enumerate(targets)
            )
            return PartBMode.HOLE_DEAL, steps
        if state.phase is HandPhase.DEALING_BOARD:
            assert state.street is not None
            steps = tuple(
                PartBStep(target, (VisionSlot(target.value),), True)
                for target in board_deal_targets(state.street)
            )
            return PartBMode.BOARD_DEAL, steps
        if state.phase is HandPhase.SHOWDOWN:
            ordered_live = clockwise_order_after(state.button, state.live_seats())
            steps = tuple(
                PartBStep(
                    DealerTargetSlot(seat.value),
                    HOLE_SLOTS[seat],
                    False,
                )
                for seat in ordered_live
            )
            return PartBMode.SHOWDOWN_REVEAL, steps
        raise ValueError("Part B requires dealing-hole, dealing-board or showdown")

    def request_rotation(self, issued_at_ns: int) -> DealerCommand:
        if self.phase is not PartBPhase.WAITING_ROTATION_ACK:
            raise ValueError("rotation is not expected in the current Part B phase")
        if self.pending_command is not None:
            return self.pending_command
        step = self.current_step
        assert step is not None
        self._command_sequence += 1
        self.pending_command = DealerCommand(
            command_id=(
                f"part-b:{self.engine.state.hand_id}:{self.engine.state.state_version}:"
                f"{self.mode.value}:rotate:{self._command_sequence}"
            ),
            issued_at_ns=issued_at_ns,
            command=DealerCommandType.ROTATE_TO,
            target_slot=step.target,
            timeout_ms=self.command_timeout_ms,
        )
        self.engine.record_dealer_command(self.pending_command)
        self.last_reason = "waiting_for_matching_rotation_ack"
        return self.pending_command

    def request_dispense(self, issued_at_ns: int) -> DealerCommand:
        if self.phase is not PartBPhase.WAITING_DISPENSE_ACK:
            raise ValueError("dispense is not expected in the current Part B phase")
        if self.pending_command is not None:
            return self.pending_command
        self._command_sequence += 1
        self.pending_command = DealerCommand(
            command_id=(
                f"part-b:{self.engine.state.hand_id}:{self.engine.state.state_version}:"
                f"{self.mode.value}:dispense:{self._command_sequence}"
            ),
            issued_at_ns=issued_at_ns,
            command=DealerCommandType.DISPENSE_ONE,
            timeout_ms=self.command_timeout_ms,
        )
        self.engine.record_dealer_command(self.pending_command)
        self.last_reason = "waiting_for_matching_dispense_ack"
        return self.pending_command

    def accept_rotation_ack(self, ack: DealerAck) -> bool:
        if ack.command_id in self._accepted_ack_ids:
            return True
        self.engine.record_dealer_ack(ack)
        if self.phase is not PartBPhase.WAITING_ROTATION_ACK:
            raise ValueError("no rotation acknowledgement is expected")
        if not self._accept_matching_ack(ack):
            return False
        step = self.current_step
        assert step is not None
        if step.dispense:
            self.phase = PartBPhase.WAITING_DISPENSE_ACK
            self.last_reason = "rotation_confirmed_request_dispense"
        else:
            self.phase = PartBPhase.WAITING_VISUAL_CONFIRMATION
            self.visual_window_opened_at_ns = ack.observed_at_ns
            self.last_reason = "rotation_confirmed_wait_reveal"
        return True

    def accept_dispense_ack(self, ack: DealerAck) -> bool:
        if ack.command_id in self._accepted_ack_ids:
            return True
        self.engine.record_dealer_ack(ack)
        if self.phase is not PartBPhase.WAITING_DISPENSE_ACK:
            raise ValueError("no dispense acknowledgement is expected")
        if not self._accept_matching_ack(ack):
            return False
        step = self.current_step
        assert step is not None and len(step.vision_slots) == 1
        self.engine.mark_delivery_pending(
            f"delivery:{ack.command_id}", step.vision_slots[0], ack.observed_at_ns
        )
        self.phase = PartBPhase.WAITING_VISUAL_CONFIRMATION
        self.visual_window_opened_at_ns = ack.observed_at_ns
        self.last_reason = "dispense_confirmed_wait_visual"
        return True

    def _accept_matching_ack(self, ack: DealerAck) -> bool:
        command = self.pending_command
        if command is None:
            raise ValueError("no dealer command is pending")
        if (
            ack.command_id != command.command_id
            or ack.command is not command.command
            or ack.target_slot is not command.target_slot
        ):
            self._enter_recovery("dealer_ack_command_or_target_mismatch")
            return False
        if ack.status is not DealerAckStatus.SUCCEEDED:
            self._enter_recovery(f"dealer_ack_{ack.status.value}")
            return False
        self._accepted_ack_ids.add(ack.command_id)
        self.pending_command = None
        return True

    def accept_card_observation(
        self, observation: CardObservation
    ) -> CardObservationResult:
        if self.phase is not PartBPhase.WAITING_VISUAL_CONFIRMATION:
            return CardObservationResult(
                False, "visual_window_not_open", self.engine.snapshot()
            )
        step = self.current_step
        assert step is not None
        if observation.slot_id not in step.vision_slots:
            return CardObservationResult(
                False, "observation_outside_current_step", self.engine.snapshot()
            )
        if (
            self.visual_window_opened_at_ns is None
            or observation.observed_at_ns < self.visual_window_opened_at_ns
        ):
            return CardObservationResult(
                False, "observation_precedes_visual_window", self.engine.snapshot()
            )
        if self.mode is PartBMode.HOLE_DEAL and observation.status not in {
            ObservationStatus.FACE_DOWN,
            ObservationStatus.UNKNOWN,
            ObservationStatus.OCCLUDED,
        }:
            return CardObservationResult(
                False, "hole_card_not_face_down", self.engine.snapshot()
            )
        if self.mode is not PartBMode.HOLE_DEAL and observation.status in {
            ObservationStatus.EMPTY,
            ObservationStatus.FACE_DOWN,
        }:
            return CardObservationResult(
                False, "visible_card_not_confirmable", self.engine.snapshot()
            )

        result = self.engine.apply_card_observation(observation)
        if not result.accepted:
            if self.engine.state.phase is HandPhase.PAUSED_RECOVERY:
                self.phase = PartBPhase.RECOVERY_REQUIRED
                self.last_reason = result.reason
            return result
        if self._current_step_complete():
            self._advance_step(observation.observed_at_ns)
        return result

    def _current_step_complete(self) -> bool:
        step = self.current_step
        assert step is not None
        expected = (
            SlotLifecycle.PRESENT_FACE_DOWN
            if self.mode is PartBMode.HOLE_DEAL
            else SlotLifecycle.CONFIRMED
        )
        return all(
            self.engine.state.slot_states[slot] is expected
            for slot in step.vision_slots
        )

    def _restore_cursor(self) -> None:
        """Resume at the first incomplete slot without repeating a dispense."""

        expected = (
            SlotLifecycle.PRESENT_FACE_DOWN
            if self.mode is PartBMode.HOLE_DEAL
            else SlotLifecycle.CONFIRMED
        )
        for index, step in enumerate(self.steps):
            if all(
                self.engine.state.slot_states[slot] is expected
                for slot in step.vision_slots
            ):
                continue
            self.step_index = index
            visual_pending = {SlotLifecycle.DELIVERY_PENDING}
            if self.mode is PartBMode.BOARD_DEAL:
                visual_pending.add(SlotLifecycle.FACE_UP_UNCONFIRMED)
            if any(
                self.engine.state.slot_states[slot] in visual_pending
                for slot in step.vision_slots
            ):
                self.phase = PartBPhase.WAITING_VISUAL_CONFIRMATION
                delivery_event = next(
                    (
                        event
                        for event in reversed(self.engine.log.events)
                        if event.kind == "card_delivery_acknowledged"
                        and event.payload.get("slot_id")
                        in {slot.value for slot in step.vision_slots}
                    ),
                    None,
                )
                self.visual_window_opened_at_ns = (
                    delivery_event.observed_at_ns if delivery_event else 0
                )
                self.last_reason = "recovered_waiting_visual_confirmation"
            return
        self._complete_mode(self.engine.log.events[-1].observed_at_ns)

    def _advance_step(self, observed_at_ns: int) -> None:
        self.visual_window_opened_at_ns = None
        if self.step_index + 1 < len(self.steps):
            self.step_index += 1
            self.phase = PartBPhase.WAITING_ROTATION_ACK
            self.last_reason = "step_complete_rotate_to_next_target"
            return

        self._complete_mode(observed_at_ns)

    def _complete_mode(self, observed_at_ns: int) -> None:
        if self.mode is PartBMode.HOLE_DEAL:
            self.engine.confirm_hole_dealt(
                f"part-b:{self.engine.state.hand_id}:hole-complete:{observed_at_ns}"
            )
        elif self.mode is PartBMode.BOARD_DEAL:
            self.engine.confirm_board_dealt(
                f"part-b:{self.engine.state.hand_id}:board-complete:{observed_at_ns}"
            )
        else:
            self.showdown_ranks = self.engine.settle_confirmed_showdown(
                f"part-b:{self.engine.state.hand_id}:showdown:{observed_at_ns}"
            )
        self.phase = PartBPhase.COMPLETE
        self.last_reason = f"part_b_complete:{self.engine.state.phase.value}"

    def check_timeout(self, now_ns: int) -> bool:
        if now_ns < 0:
            raise ValueError("timeout clock must be non-negative")
        if self.pending_command is not None:
            deadline = (
                self.pending_command.issued_at_ns
                + self.pending_command.timeout_ms * 1_000_000
            )
            if now_ns >= deadline:
                self._enter_recovery("dealer_command_timeout")
                return True
        if (
            self.phase is PartBPhase.WAITING_VISUAL_CONFIRMATION
            and self.visual_window_opened_at_ns is not None
            and now_ns
            >= self.visual_window_opened_at_ns + self.visual_timeout_ms * 1_000_000
        ):
            self._enter_recovery("card_visual_timeout")
            return True
        return False

    def _enter_recovery(self, reason: str) -> None:
        self.phase = PartBPhase.RECOVERY_REQUIRED
        self.pending_command = None
        self.visual_window_opened_at_ns = None
        self.last_reason = reason
        if self.engine.state.phase is not HandPhase.PAUSED_RECOVERY:
            self.engine.pause(
                f"part-b:{self.engine.state.hand_id}:pause:{self.engine.state.state_version}",
                reason,
            )


__all__ = [
    "PartBMode",
    "PartBPhase",
    "PartBStep",
    "SequentialPartBCoordinator",
]
