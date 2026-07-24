from __future__ import annotations

from collections import Counter

from scripts.data.build_card_big_data_v3 import select_validation_class_ids


def test_validation_class_selection_is_deterministic_and_balanced_by_suit() -> None:
    classes = [
        f"{rank}{suit}"
        for rank in ("A", "K", "Q", "J")
        for suit in ("C", "D", "H", "S")
    ]

    selected = select_validation_class_ids(classes, seed=20260724)

    assert selected == select_validation_class_ids(classes, seed=20260724)
    assert len(selected) == 8
    assert Counter(classes[class_id][-1] for class_id in selected) == Counter(
        {"C": 2, "D": 2, "H": 2, "S": 2}
    )
