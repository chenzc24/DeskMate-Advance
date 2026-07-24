from __future__ import annotations

import pytest

from scripts.perception.evaluate_card_detector import grouped_macro_metrics


def test_grouped_macro_metrics_aggregates_rank_and_suit() -> None:
    records = [
        {
            "rank": "5",
            "suit": "D",
            "precision": 0.8,
            "recall": 0.6,
            "f1": 0.7,
            "map50": 0.9,
            "map50_95": 0.5,
        },
        {
            "rank": "5",
            "suit": "H",
            "precision": 1.0,
            "recall": 0.8,
            "f1": 0.9,
            "map50": 0.7,
            "map50_95": 0.7,
        },
    ]

    by_rank = grouped_macro_metrics(records, "rank")

    assert by_rank["5"] == pytest.approx(
        {
            "precision": 0.9,
            "recall": 0.7,
            "f1": 0.8,
            "map50": 0.8,
            "map50_95": 0.6,
        }
    )
