"""Single facade coordinating the authoritative engine with Part A and B."""

from __future__ import annotations

from typing import Mapping

from poker_dealer.domain import (
    CardObservation,
    DealerAck,
    DealerCommand,
    HandPhase,
    PlayerActionObservation,
    Seat,
)
from poker_dealer.game import (
    CardObservationResult,
    FixedLimitRules,
    HandEngine,
    HandRank,
)
from poker_dealer.perception.attribution import ActorBinding, AttributedActionCandidate
from poker_dealer.perception.identity import FaceIdentityObservation

from .sequential_part_a import CoordinatorActionOutcome, PartAPhase, SequentialPartACoordinator
from .sequential_part_b import PartBPhase, SequentialPartBCoordinator
from .registration import FrozenSessionRoster


class HandRuntime:
    """Expose one legal path through a complete hand.

    This facade deliberately has no independent high-level phase. It follows
    `HandEngine.state.phase` and creates exactly one active lane at a time.
    """

    def __init__(
        self,
        engine: HandEngine,
        session_id: str,
        *,
        require_actor_binding: bool = True,
        require_visual_settle: bool = True,
        visual_settle_timeout_ms: int = 5000,
        command_timeout_ms: int = 5000,
        visual_timeout_ms: int = 5000,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        self.engine = engine
        self.session_id = session_id
        self.require_actor_binding = require_actor_binding
        self.require_visual_settle = require_visual_settle
        self.visual_settle_timeout_ms = visual_settle_timeout_ms
        self.command_timeout_ms = command_timeout_ms
        self.visual_timeout_ms = visual_timeout_ms
        self.part_a: SequentialPartACoordinator | None = None
        self.part_b: SequentialPartBCoordinator | None = None
        self.last_showdown_ranks: Mapping[Seat, HandRank] | None = None
        if (
            self.engine.state.pending_command_id is not None
            and self.engine.state.phase
            not in {
                HandPhase.PAUSED_RECOVERY,
                HandPhase.SETTLED,
                HandPhase.VOIDED,
            }
        ):
            self.engine.pause(
                f"runtime:{self.engine.state.hand_id}:restart-pause",
                "recovered_with_pending_dealer_command",
            )
        self.sync()

    @classmethod
    def new_hand(
        cls,
        *,
        hand_id: str,
        session_id: str,
        button: Seat,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
        require_actor_binding: bool = True,
        require_visual_settle: bool = True,
        visual_settle_timeout_ms: int = 5000,
        command_timeout_ms: int = 5000,
        visual_timeout_ms: int = 5000,
    ) -> HandRuntime:
        engine = HandEngine.setup_session(hand_id, button, stacks, rules)
        engine.begin_hand(f"{hand_id}:begin")
        return cls(
            engine,
            session_id,
            require_actor_binding=require_actor_binding,
            require_visual_settle=require_visual_settle,
            visual_settle_timeout_ms=visual_settle_timeout_ms,
            command_timeout_ms=command_timeout_ms,
            visual_timeout_ms=visual_timeout_ms,
        )

    @classmethod
    def from_roster(
        cls,
        *,
        hand_id: str,
        roster: FrozenSessionRoster,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
        require_actor_binding: bool = True,
        require_visual_settle: bool = True,
        visual_settle_timeout_ms: int = 5000,
        command_timeout_ms: int = 5000,
        visual_timeout_ms: int = 5000,
    ) -> HandRuntime:
        """Start the product path only from a frozen four-player roster."""

        if not isinstance(roster, FrozenSessionRoster):
            raise ValueError("a FrozenSessionRoster is required")
        seats = {participant.seat for participant in roster.participants}
        if len(roster.participants) != 4 or seats != set(Seat):
            raise ValueError("the frozen roster must contain all four unique seats")
        return cls.new_hand(
            hand_id=hand_id,
            session_id=roster.session_id,
            button=roster.button,
            stacks=stacks,
            rules=rules,
            require_actor_binding=require_actor_binding,
            require_visual_settle=require_visual_settle,
            visual_settle_timeout_ms=visual_settle_timeout_ms,
            command_timeout_ms=command_timeout_ms,
            visual_timeout_ms=visual_timeout_ms,
        )

    @property
    def phase(self) -> HandPhase:
        return self.engine.state.phase

    def sync(self) -> None:
        """Select the only lane allowed by the current authoritative phase."""

        phase = self.engine.state.phase
        if phase is HandPhase.AWAITING_ACTION:
            self.part_b = None
            if self.part_a is None or self.part_a.phase in {
                PartAPhase.ROUND_COMPLETE,
                PartAPhase.RECOVERY_REQUIRED,
            }:
                self.part_a = SequentialPartACoordinator(
                    self.engine,
                    self.session_id,
                    require_actor_binding=self.require_actor_binding,
                    require_visual_settle=self.require_visual_settle,
                    visual_settle_timeout_ms=self.visual_settle_timeout_ms,
                )
            return
        if phase in {
            HandPhase.DEALING_HOLE,
            HandPhase.DEALING_BOARD,
            HandPhase.SHOWDOWN,
        }:
            self.part_a = None
            if self.part_b is None or self.part_b.phase in {
                PartBPhase.COMPLETE,
                PartBPhase.RECOVERY_REQUIRED,
            }:
                self.part_b = SequentialPartBCoordinator(
                    self.engine,
                    command_timeout_ms=self.command_timeout_ms,
                    visual_timeout_ms=self.visual_timeout_ms,
                )
                if (
                    self.part_b.phase is PartBPhase.COMPLETE
                    and self.engine.state.phase is not phase
                ):
                    self.sync()
            return
        self.part_a = None
        self.part_b = None

    def request_rotation(self, issued_at_ns: int) -> DealerCommand:
        if self.part_a is not None:
            return self.part_a.request_rotation(issued_at_ns)
        if self.part_b is not None:
            return self.part_b.request_rotation(issued_at_ns)
        raise ValueError("the current hand phase does not request rotation")

    def accept_rotation_ack(self, ack: DealerAck) -> bool:
        if self.part_a is not None:
            accepted = self.part_a.accept_rotation_ack(ack)
        elif self.part_b is not None:
            accepted = self.part_b.accept_rotation_ack(ack)
        else:
            raise ValueError("the current hand phase does not accept rotation ACKs")
        self.sync()
        return accepted

    def request_dispense(self, issued_at_ns: int) -> DealerCommand:
        if self.part_b is None:
            raise ValueError("dispense is only available in Part B")
        return self.part_b.request_dispense(issued_at_ns)

    def accept_dispense_ack(self, ack: DealerAck) -> bool:
        if self.part_b is None:
            raise ValueError("dispense ACK is only available in Part B")
        accepted = self.part_b.accept_dispense_ack(ack)
        self.sync()
        return accepted

    def accept_card_observation(
        self, observation: CardObservation
    ) -> CardObservationResult:
        if self.part_b is None:
            return CardObservationResult(
                False, "part_b_not_active", self.engine.snapshot()
            )
        coordinator = self.part_b
        result = coordinator.accept_card_observation(observation)
        if coordinator.showdown_ranks is not None:
            self.last_showdown_ranks = coordinator.showdown_ranks
        self.sync()
        return result

    def accept_visual_settle(self) -> None:
        if self.part_a is None:
            raise ValueError("visual settle is only available in Part A")
        self.part_a.accept_visual_settle()

    def accept_identity(self, observation: FaceIdentityObservation) -> bool:
        if self.part_a is None:
            raise ValueError("identity is only available in Part A")
        return self.part_a.accept_identity(observation)

    def bind_actor(self, binding: ActorBinding) -> None:
        if self.part_a is None:
            raise ValueError("actor binding is only available in Part A")
        self.part_a.bind_actor(binding)

    def accept_attributed_action(
        self, candidate: AttributedActionCandidate
    ) -> CoordinatorActionOutcome:
        if self.part_a is None:
            raise ValueError("player action is only available in Part A")
        outcome = self.part_a.accept_attributed_action(candidate)
        self.sync()
        return outcome

    def accept_action(
        self, observation: PlayerActionObservation
    ) -> CoordinatorActionOutcome:
        if self.part_a is None:
            raise ValueError("player action is only available in Part A")
        outcome = self.part_a.accept_action(observation)
        self.sync()
        return outcome

    def check_timeout(self, now_ns: int) -> bool:
        if self.part_a is not None:
            timed_out = self.part_a.check_timeout(now_ns)
        elif self.part_b is not None:
            timed_out = self.part_b.check_timeout(now_ns)
        else:
            timed_out = False
        self.sync()
        return timed_out

    def void(self, event_id: str, reason: str) -> None:
        self.engine.void(event_id, reason)
        self.sync()


__all__ = ["HandRuntime"]
