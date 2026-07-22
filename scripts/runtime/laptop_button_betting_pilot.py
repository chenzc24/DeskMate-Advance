"""Exercise Fixed-Limit button betting and the digital ledger without cameras."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from poker_dealer.runtime import (
    ButtonBettingRuntime,
    ConsoleAnnouncer,
    EventAnnouncer,
    WindowsSpeechAnnouncer,
)


ROOT = Path(__file__).resolve().parents[2]
ROLE_LABELS = {
    "button": "Button",
    "small_blind": "Small Blind",
    "big_blind": "Big Blind",
    "under_the_gun": "Under the Gun",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--button",
        required=True,
        choices=tuple(seat.value for seat in Seat),
        help="internal physical seat currently holding Button",
    )
    parser.add_argument(
        "--announcer", choices=("off", "console", "windows"), default="console"
    )
    return parser.parse_args()


def _summary(engine: HandEngine, runtime: ButtonBettingRuntime) -> dict[str, object]:
    state = engine.state
    acting_role = (
        role_for_seat(state.button, state.acting_seat).value
        if state.acting_seat is not None
        else None
    )
    return {
        "phase": state.phase.value,
        "street": state.street.value if state.street else None,
        "acting_role": acting_role,
        "legal_actions": [item.value for item in state.legal_actions],
        "selected_action": (
            runtime.selected_action.value if runtime.selected_action else None
        ),
        "stacks": {
            ROLE_LABELS[role_for_seat(state.button, seat).value]: player.stack_units
            for seat, player in state.players.items()
        },
        "pot_units": state.pot_units,
        "state_version": state.state_version,
    }


def main() -> int:
    args = parse_args()
    rules = FixedLimitRules.from_project_config(ROOT / "configs/game/core_v1.json")
    engine = HandEngine.start("laptop-button-pilot", Seat(args.button), rules=rules)
    runtime = ButtonBettingRuntime(engine)
    announcer_context = nullcontext(None)
    if args.announcer == "windows":
        announcer_context = WindowsSpeechAnnouncer()
    elif args.announcer == "console":
        announcer_context = nullcontext(ConsoleAnnouncer())

    with announcer_context as announcer_port:
        announcer = EventAnnouncer(announcer_port) if announcer_port else None
        if announcer is not None:
            announcer.publish(
                "blind_posted", role="small_blind", amount_units=rules.small_blind_units
            )
            announcer.publish(
                "blind_posted", role="big_blind", amount_units=rules.big_blind_units
            )
            assert engine.state.acting_seat is not None
            announcer.publish(
                "turn_started",
                role=role_for_seat(engine.state.button, engine.state.acting_seat).value,
            )

        sequence = 0
        print("N next | P previous | Enter confirm | Q quit", flush=True)
        while engine.state.phase is HandPhase.AWAITING_ACTION:
            print(json.dumps(_summary(engine, runtime), ensure_ascii=False), flush=True)
            entered = input("control> ").strip().lower()
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
            observation = ControlObservation(
                f"laptop-button:{sequence}:{observed_at_ns}",
                intent,
                ControlSource.LAPTOP_KEYBOARD,
                observed_at_ns,
                "keyboard",
                sequence,
            )
            acting_seat = engine.state.acting_seat
            before_stack = (
                engine.state.players[acting_seat].stack_units
                if acting_seat is not None
                else 0
            )
            outcome = runtime.accept_control(observation)
            print(
                json.dumps(
                    {
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "selected_action": (
                            outcome.selected_action.value
                            if outcome.selected_action
                            else None
                        ),
                    }
                ),
                flush=True,
            )
            if outcome.action_result is not None and outcome.action_result.accepted:
                assert acting_seat is not None and outcome.selected_action is not None
                contribution = max(
                    0, before_stack - engine.state.players[acting_seat].stack_units
                )
                if announcer is not None:
                    announcer.publish(
                        "action_committed",
                        role=role_for_seat(engine.state.button, acting_seat).value,
                        action=outcome.selected_action.value,
                        amount_units=(contribution or None),
                    )
                    if engine.state.acting_seat is not None:
                        announcer.publish(
                            "turn_started",
                            role=role_for_seat(
                                engine.state.button, engine.state.acting_seat
                            ).value,
                        )

        print(json.dumps(_summary(engine, runtime), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
