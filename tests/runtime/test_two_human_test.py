from __future__ import annotations

from poker_dealer.domain import (
    HandPhase,
    PlayerActionType,
    Seat,
)
from poker_dealer.game import ActionPromoter, HandEngine, PromotionPolicy
from poker_dealer.perception.identity import FaceIdentityState
from poker_dealer.runtime import RuntimeObservationContext, TwoHumanAutoFoldSource


class _HumanDelegate:
    def __init__(self) -> None:
        self.identity_calls = 0
        self.action_calls = 0

    def observe_identity(self, frame, context, observed_at_ns):
        del frame, context, observed_at_ns
        self.identity_calls += 1
        return None

    def observe_action(self, frame, context, observed_at_ns):
        del frame, context, observed_at_ns
        self.action_calls += 1
        return None


def _context(seat: Seat) -> RuntimeObservationContext:
    return RuntimeObservationContext(
        session_id="two-human-session",
        hand_id="hand-001",
        state_version=7,
        hand_phase=HandPhase.AWAITING_ACTION,
        focus_seat=seat,
        legal_actions=(PlayerActionType.FOLD, PlayerActionType.CALL),
        required_card_slots=(),
        camera_epoch=2,
    )


def test_simulated_seat_emits_bound_auto_fold_evidence() -> None:
    delegate = _HumanDelegate()
    promotion_policy = PromotionPolicy(
        minimum_confidence=0.60,
        minimum_stable_frames=5,
        minimum_stable_duration_ms=350,
    )
    source = TwoHumanAutoFoldSource(
        delegate,
        {
            Seat.B: "development-simulator-seat-b",
            Seat.C: "development-simulator-seat-c",
        },
        promotion_policy=promotion_policy,
    )
    context = _context(Seat.B)
    identity = source.observe_identity(None, context, 1_000_000)
    evidence = source.observe_action(None, context, 500_000_000)

    assert identity is not None
    assert identity.identity_state is FaceIdentityState.MATCHED
    assert identity.player_id == "development-simulator-seat-b"
    assert "simulated_identity" in identity.quality_flags
    assert evidence is not None
    assert evidence.observation.candidate_action is PlayerActionType.FOLD
    assert evidence.observation.expected_state_version == 7
    assert evidence.observation.stable_frames == 5
    assert evidence.observation.stable_duration_ms == 350
    assert evidence.observation.window_started_at_ns == 150_000_000
    assert evidence.actor_binding is not None
    assert evidence.actor_binding.player_id == "development-simulator-seat-b"
    assert evidence.attribution_source == "development_simulator"
    assert "auto_fold" in evidence.quality_flags
    assert delegate.identity_calls == 0
    assert delegate.action_calls == 0


def test_simulated_auto_fold_advances_engine_with_matching_policy() -> None:
    delegate = _HumanDelegate()
    promotion_policy = PromotionPolicy(
        minimum_confidence=0.60,
        minimum_stable_frames=4,
        minimum_stable_duration_ms=250,
    )
    engine = HandEngine.start("auto-fold-integration", Seat.A)
    engine.promoter = ActionPromoter(promotion_policy)
    assert engine.state.acting_seat is Seat.D
    context = RuntimeObservationContext(
        session_id="two-human-session",
        hand_id=engine.state.hand_id,
        state_version=engine.state.state_version,
        hand_phase=engine.state.phase,
        focus_seat=engine.state.acting_seat,
        legal_actions=engine.state.legal_actions,
        required_card_slots=(),
        camera_epoch=2,
    )
    source = TwoHumanAutoFoldSource(
        delegate,
        {Seat.D: "development-simulator-seat-d"},
        promotion_policy=promotion_policy,
    )

    evidence = source.observe_action(None, context, 1_000_000_000)

    assert evidence is not None
    result = engine.apply_observation(evidence.observation)
    assert result.accepted
    assert engine.state.state_version == 1
    assert engine.state.players[Seat.D].folded
    assert engine.state.acting_seat is Seat.A


def test_human_seats_delegate_to_live_sources() -> None:
    delegate = _HumanDelegate()
    source = TwoHumanAutoFoldSource(
        delegate,
        {Seat.B: "development-simulator-seat-b"},
    )
    context = _context(Seat.A)

    assert source.observe_identity(None, context, 1_000_000) is None
    assert source.observe_action(None, context, 2_000_000) is None
    assert delegate.identity_calls == 1
    assert delegate.action_calls == 1
