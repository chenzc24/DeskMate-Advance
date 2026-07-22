"""Pilot only: exercise Fixed-Limit controls and ledger without perception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from poker_dealer.domain import (
    ControlIntent,
    ControlObservation,
    ControlSource,
    HandPhase,
    Seat,
    role_for_seat,
)
from poker_dealer.game import FixedLimitRules, HandEngine
from poker_dealer.pilots import ButtonBettingRuntime


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--button",
        required=True,
        choices=tuple(seat.value for seat in Seat),
    )
    return parser.parse_args()


def _summary(engine: HandEngine, runtime: ButtonBettingRuntime) -> dict[str, object]:
    state = engine.state
    return {
        "pilot_only": True,
        "phase": state.phase.value,
        "street": state.street.value if state.street else None,
        "acting_role": (
            role_for_seat(state.button, state.acting_seat).value
            if state.acting_seat is not None
            else None
        ),
        "legal_actions": [item.value for item in state.legal_actions],
        "selected_action": (
            runtime.selected_action.value if runtime.selected_action else None
        ),
        "stacks": {seat.value: player.stack_units for seat, player in state.players.items()},
        "pot_units": state.pot_units,
        "state_version": state.state_version,
    }


def main() -> int:
    args = parse_args()
    rules = FixedLimitRules.from_project_config(ROOT / "configs/game/core_v1.json")
    engine = HandEngine.start("laptop-button-pilot", Seat(args.button), rules=rules)
    runtime = ButtonBettingRuntime(engine, allow_direct_engine_pilot=True)
    sequence = 0
    print("PILOT ONLY | N next | P previous | Enter confirm | Q quit", flush=True)
    while engine.state.phase is HandPhase.AWAITING_ACTION:
        print(json.dumps(_summary(engine, runtime)), flush=True)
        entered = input("pilot-control> ").strip().lower()
        if entered == "q":
            break
        intent = {
            "n": ControlIntent.NEXT_OPTION,
            "p": ControlIntent.PREVIOUS_OPTION,
            "": ControlIntent.CONFIRM,
        }.get(entered)
        if intent is None:
            print("unknown control", flush=True)
            continue
        sequence += 1
        observed_at_ns = time.monotonic_ns()
        outcome = runtime.accept_control(
            ControlObservation(
                observation_id=f"laptop-button:{sequence}:{observed_at_ns}",
                intent=intent,
                source=ControlSource.LAPTOP_KEYBOARD,
                observed_at_ns=observed_at_ns,
                control_id="keyboard",
                device_state_version=sequence,
            )
        )
        print(
            json.dumps(
                {
                    "accepted": outcome.accepted,
                    "reason": outcome.reason,
                    "selected_action": (
                        outcome.selected_action.value if outcome.selected_action else None
                    ),
                }
            ),
            flush=True,
        )
    print(json.dumps(_summary(engine, runtime)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
