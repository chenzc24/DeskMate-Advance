from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/data"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_chip_card_hard_negative_view import build_view  # noqa: E402


def _write_image(
    path: Path,
    color: tuple[int, int, int],
    marker: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((96, 128, 3), color, dtype=np.uint8)
    center = (36 + marker * 18, 30 + marker * 9)
    cv2.circle(image, center, 12 + marker * 4, tuple(reversed(color)), cv2.FILLED)
    cv2.putText(
        image,
        str(marker),
        (10 + marker * 25, 85 - marker * 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )
    assert cv2.imwrite(str(path), image)


def _base_dataset(root: Path) -> None:
    for index, split in enumerate(("train", "valid", "test")):
        image = root / split / "images" / f"{split}.jpg"
        label = root / split / "labels" / f"{split}.txt"
        _write_image(image, (20 + index * 40, 90, 180), marker=index + 4)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "target_validation.yaml").write_text(
        "path: .\ntrain: valid/images\nval: valid/images\nnames:\n  0: poker_chip\n",
        encoding="utf-8",
    )


def test_build_view_preserves_base_and_keeps_negative_groups_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_image(source / "chip_neg/face_1.png", (10, 30, 200), marker=1)
    _write_image(source / "chip_neg/face_2.png", (30, 200, 10), marker=2)
    _write_image(source / "cards_neg/back_1.png", (200, 10, 30), marker=3)
    base = tmp_path / "base"
    _base_dataset(base)

    report = build_view(
        source_root=source,
        raw_root=tmp_path / "raw",
        work_root=tmp_path / "work",
        base_root=base,
        variants_per_train_source=3,
    )

    manifest_path = Path(report["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_dataset_replay"] == {
        "train_images": 1,
        "valid_images": 1,
        "test_images": 1,
        "target_validation_yaml": str((base / "target_validation.yaml").resolve()),
    }
    assert manifest["new_negative_sources"] == {"train": 2, "heldout": 1}
    assert manifest["new_negative_images"] == {"train": 6, "heldout": 1}
    assert manifest["all_new_labels_empty"] is True
    labels = list((tmp_path / "work/dataset").rglob("*.txt"))
    new_labels = [
        path for path in labels if path.name != "negative_holdout.txt"
    ]
    assert len(new_labels) == 7
    assert all(path.read_bytes() == b"" for path in new_labels)
    data_yaml = (tmp_path / "work/dataset/data.yaml").read_text(encoding="utf-8")
    assert str((base / "train/images").resolve().as_posix()) in data_yaml
    assert str((base / "valid/images").resolve().as_posix()) in data_yaml
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == report[
        "manifest_sha256"
    ]
