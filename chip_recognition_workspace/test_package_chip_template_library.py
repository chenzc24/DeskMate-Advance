from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from package_chip_template_library import package_library, sha256


def test_package_library_strips_private_paths_and_hashes_masks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    masks = source / "masks"
    masks.mkdir(parents=True)
    templates = []
    for denomination in (1, 5, 10, 20):
        template_id = f"chip_{denomination}_sample"
        mask_path = masks / f"{template_id}.png"
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.putText(
            mask,
            str(denomination),
            (24, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            255,
            3,
        )
        assert cv2.imwrite(str(mask_path), mask)
        templates.append(
            {
                "template_id": template_id,
                "denomination": denomination,
                "source": f"C:/private/{template_id}.png",
                "source_sha256": f"{denomination:064x}",
                "mask_file": f"masks/{template_id}.png",
                "normalized_file": f"C:/private/work/{template_id}.png",
                "center_file": f"C:/private/work/{template_id}.png",
                "color_signature": [float(denomination)] * 12,
            }
        )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "denominations": [1, 5, 10, 20],
                "center_fraction": 0.4,
                "templates": templates,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    packaged = package_library(source, output)

    assert packaged["template_count"] == 4
    assert packaged["counts"] == {"1": 1, "5": 1, "10": 1, "20": 1}
    serialized = (output / "manifest.json").read_text(encoding="utf-8")
    assert "C:/private" not in serialized
    for item in packaged["templates"]:
        mask_path = output / item["mask_file"]
        assert mask_path.is_file()
        assert item["mask_sha256"] == sha256(mask_path)
