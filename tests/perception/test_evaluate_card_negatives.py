from scripts.perception.evaluate_card_negatives import aggregate_confidences


def test_aggregate_confidences_counts_images_and_boxes() -> None:
    metrics = aggregate_confidences(
        [[0.7, 0.2], [], [0.14], [0.9]],
        0.15,
    )

    assert metrics == {
        "images": 4,
        "images_with_false_detection": 2,
        "false_boxes": 3,
        "false_boxes_per_image": 0.75,
    }
