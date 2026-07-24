from __future__ import annotations

from scripts.data.build_card_v4_training_view import split_rotation_records


def test_rotation_siblings_stay_together_by_source_class() -> None:
    records = [
        {
            "class_id": class_id,
            "source_image_sha256": f"source-{class_id}",
            "rotation_degrees": angle,
        }
        for class_id in range(4)
        for angle in (0, 90, 180, 270)
    ]

    train, validation = split_rotation_records(records, {1})

    assert len(train) == 12
    assert len(validation) == 4
    assert {record["class_id"] for record in validation} == {1}
    assert {record["source_image_sha256"] for record in train}.isdisjoint(
        {record["source_image_sha256"] for record in validation}
    )
