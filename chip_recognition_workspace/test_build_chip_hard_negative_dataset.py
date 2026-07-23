from __future__ import annotations

import json

import cv2
import numpy as np

from build_chip_hard_negative_dataset import build_dataset


def _write_base(root):
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        assert cv2.imwrite(
            str(image_dir / f"{split}.png"),
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        (label_dir / f"{split}.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\n"
        "test: test/images\nnames:\n  0: poker_chip\n",
        encoding="utf-8",
    )
    (root / "dataset_manifest.json").write_text(
        '{"schema_version":"1.0"}\n',
        encoding="utf-8",
    )


def test_negative_session_is_train_only_with_empty_labels(tmp_path):
    base = tmp_path / "base"
    negatives = tmp_path / "negatives"
    output = tmp_path / "output"
    _write_base(base)
    negatives.mkdir()
    assert cv2.imwrite(
        str(negatives / "hard.png"),
        np.full((24, 40, 3), 127, dtype=np.uint8),
    )

    result = build_dataset(base, negatives, output, repeats=3)

    assert result["split_image_counts"] == {"train": 4, "valid": 1, "test": 1}
    negative_labels = sorted((output / "train" / "labels").glob("target_*.txt"))
    assert len(negative_labels) == 3
    assert all(path.read_text(encoding="utf-8") == "" for path in negative_labels)
    assert len(list((output / "valid" / "images").iterdir())) == 1
    manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["negative_unique_images"] == 1
    assert manifest["negative_repeat_factor"] == 3
