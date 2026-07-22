"""Validated Core game configuration loaded by runtime composition roots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from poker_dealer.domain import SEAT_ORDER, Seat

from .rules import FixedLimitRules


@dataclass(frozen=True, slots=True)
class CoreGameConfig:
    schema_version: str
    rules: FixedLimitRules
    starting_stack_units: int
    minimum_stack_to_start_hand_units: int

    @classmethod
    def from_json(cls, path: str | Path) -> "CoreGameConfig":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Core game config root must be an object")
        table = value.get("table")
        session = value.get("session_defaults")
        if not isinstance(table, Mapping) or not isinstance(session, Mapping):
            raise ValueError("Core game table and session_defaults are required")
        if int(table.get("players", 0)) != len(SEAT_ORDER):
            raise ValueError("Core runtime requires exactly four players")
        if tuple(table.get("seats_clockwise", ())) != tuple(
            seat.value for seat in SEAT_ORDER
        ):
            raise ValueError("Core seat order differs from the domain contract")
        starting = int(session.get("starting_stack_units", 0))
        minimum = int(session.get("minimum_stack_to_start_hand_units", 0))
        if starting <= 0 or minimum <= 0 or starting < minimum:
            raise ValueError("Core session stack defaults are invalid")
        rules = FixedLimitRules.from_project_config(source)
        if minimum < rules.big_blind_units:
            raise ValueError("minimum starting stack must cover the big blind")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            rules=rules,
            starting_stack_units=starting,
            minimum_stack_to_start_hand_units=minimum,
        )

    def default_stacks(self) -> dict[Seat, int]:
        return {seat: self.starting_stack_units for seat in SEAT_ORDER}


__all__ = ["CoreGameConfig"]
