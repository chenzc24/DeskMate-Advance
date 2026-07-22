"""Validated deterministic betting-rule parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from poker_dealer.domain import Street


@dataclass(frozen=True, slots=True)
class FixedLimitRules:
    small_blind_units: int = 1
    big_blind_units: int = 2
    small_bet_units: int = 2
    big_bet_units: int = 4
    max_full_bets_per_street: int = 4
    action_timeout_seconds: int = 30
    rules_version: str = "1.3"
    product_status: str = "confirmed_core_v1"

    def __post_init__(self) -> None:
        numeric = (
            self.small_blind_units,
            self.big_blind_units,
            self.small_bet_units,
            self.big_bet_units,
            self.max_full_bets_per_street,
            self.action_timeout_seconds,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("Fixed-Limit values must be positive")
        if self.small_blind_units > self.big_blind_units:
            raise ValueError("small blind cannot exceed big blind")

    @classmethod
    def from_project_config(cls, path: str | Path) -> FixedLimitRules:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        betting = config["betting"]
        blinds = config["blinds_defaults"]
        if betting.get("structure") != "fixed_limit":
            raise ValueError("Core v1 requires betting.structure=fixed_limit")
        if betting.get("product_decision_status") != "confirmed_core_v1":
            raise ValueError("Core v1 Fixed-Limit product decision is not confirmed")
        return cls(
            small_blind_units=blinds["small_blind_units"],
            big_blind_units=blinds["big_blind_units"],
            small_bet_units=betting["small_bet_units_default"],
            big_bet_units=betting["big_bet_units_default"],
            max_full_bets_per_street=betting[
                "max_full_bets_per_street_default"
            ],
            action_timeout_seconds=betting["action_timeout_seconds_default"],
            rules_version=config["schema_version"],
            product_status=betting["product_decision_status"],
        )

    def bet_size(self, street: Street) -> int:
        if street in {Street.PREFLOP, Street.FLOP}:
            return self.small_bet_units
        if street in {Street.TURN, Street.RIVER}:
            return self.big_bet_units
        raise ValueError("showdown has no bet size")


__all__ = ["FixedLimitRules"]
