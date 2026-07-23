from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.data.build_card_retrain_v2 import (
    _select_with_coverage,
    split_local_sources,
)


def _write_source(root: Path, class_id: int, name: str, nested_images: bool) -> None:
    image_root = root / "images" if nested_images else root
    labels_root = root / "labels"
    image_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    pixel_value = 160 + class_id + (20 if nested_images else 0)
    image = np.full((120, 160, 3), pixel_value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    (image_root / f"{name}.png").write_bytes(encoded.tobytes())
    (labels_root / f"{name}.txt").write_text(
        f"{class_id} 0.30 0.30 0.10 0.10\n"
        f"{class_id} 0.70 0.70 0.10 0.10\n",
        encoding="utf-8",
    )


def test_local_split_balances_origins_and_keeps_source_hashes_disjoint(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    for class_id in range(2):
        _write_source(old_root, class_id, f"old_{class_id}", nested_images=True)
        _write_source(new_root, class_id, f"new_{class_id}", nested_images=False)

    result = split_local_sources(
        old_root,
        new_root,
        tmp_path / "split",
        ["A", "B"],
        seed=7,
    )

    assert result["train_sources"] == 2
    assert result["validation_sources"] == 2
    assert result["train_old"] == result["train_new"] == 1
    assert result["validation_old"] == result["validation_new"] == 1
    assert result["source_sha256_overlap"] == []


def test_external_selection_is_exact_and_covers_every_class() -> None:
    records = [
        {
            "image": Path(f"{index}.jpg"),
            "class_ids": [index % 4, (index + 1) % 4],
        }
        for index in range(40)
    ]

    selected = _select_with_coverage(records, 20, class_count=4, minimum_per_class=3)

    assert len(selected) == 20
    covered = {
        class_id for record in selected for class_id in record["class_ids"]
    }
    assert covered == {0, 1, 2, 3}
