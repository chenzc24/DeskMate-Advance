import pytest

from poker_dealer.domain import HandPhase, Seat
from poker_dealer.game import HandEngine
from poker_dealer.runtime import HandRuntime


def test_recovery_requires_operator_parity_and_restores_attention_phase() -> None:
    engine = HandEngine.start_predealt_fixture("recovery", Seat.A)
    acting = engine.state.acting_seat
    engine.pause("pause-1", "camera_disconnected")
    runtime = HandRuntime(engine, "session", require_visual_settle=False)
    with pytest.raises(ValueError, match="parity"):
        runtime.resume_from_recovery(
            "resume-1",
            operator_id="operator-1",
            reason="camera_reconnected",
            physical_state_confirmed=False,
        )
    runtime.resume_from_recovery(
        "resume-2",
        operator_id="operator-1",
        reason="camera_reconnected_and_table_unchanged",
        physical_state_confirmed=True,
    )
    assert runtime.phase is HandPhase.AWAITING_ACTION
    assert runtime.engine.state.acting_seat is acting
    assert runtime.part_a is not None
    assert runtime.engine.log.events[-1].kind == "hand_recovery_resumed"
