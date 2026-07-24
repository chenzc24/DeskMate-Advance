"""Extend the fixed-design chip library with reviewed front views of 10 and 20."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "chip_recognition_workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chip_template_matcher import (  # noqa: E402
    CENTER_FRACTION,
    color_signature,
    center_number_view,
    digit_mask,
)
from rectify_chip_images import (  # noqa: E402
    _derive_top_ellipse_from_inlay,
    _ellipse_to_circle,
    _fit_top_ellipse,
    _grabcut_chip_mask,
)


DEFAULT_RAW = ROOT / "data/raw/chips/2026-07-24-chip-v2-source"
DEFAULT_ANNOTATIONS = (
    ROOT
    / "data/work/chips/2026-07-24-chip-v2-optimization/reviewed_annotations_candidate.json"
)
DEFAULT_BASE = (
    ROOT / "models/assets/chip_recognition/las-vegas-denomination-templates-v1"
)
DEFAULT_OUTPUT = (
    ROOT / "data/work/chips/2026-07-24-chip-v2-optimization/denomination_library"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--base-library", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalized-size", type=int, default=384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    library = output / "library"
    if output.exists():
        if output.parent != (ROOT / "data/work/chips/2026-07-24-chip-v2-optimization").resolve():
            raise SystemExit(f"refusing unexpected output cleanup: {output}")
        shutil.rmtree(output)
    masks_dir = library / "masks"
    normalized_dir = output / "normalized"
    center_dir = output / "center_views"
    masks_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)
    center_dir.mkdir(parents=True)

    base_manifest_path = args.base_library / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    templates = [dict(item) for item in base_manifest["templates"]]
    for item in templates:
        source = args.base_library / item["mask_file"]
        destination = library / item["mask_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    rejected: list[dict[str, object]] = []
    added: list[dict[str, object]] = []
    for record in annotations["records"]:
        if not str(record["capture_group"]).startswith("straight:"):
            continue
        if len(record["instances"]) != 1:
            raise ValueError(f"front-view record must have one chip: {record['relative_path']}")
        source = raw_root / record["relative_path"]
        if sha256(source) != record["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"undecodable source: {source}")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(float, record["instances"][0]["box_xyxy"])
        pad_x = (x2 - x1) * 0.20
        pad_y = (y2 - y1) * 0.20
        left = max(0, round(x1 - pad_x))
        top = max(0, round(y1 - pad_y))
        right = min(width, round(x2 + pad_x))
        bottom = min(height, round(y2 + pad_y))
        crop = np.ascontiguousarray(image[top:bottom, left:right])
        cv2.setRNGSeed(int(record["source_sha256"][:8], 16) & 0x7FFFFFFF)
        outer_mask = _grabcut_chip_mask(crop)
        preliminary, _ = _fit_top_ellipse(outer_mask)
        if preliminary is None:
            rejected.append({"relative_path": record["relative_path"], "reason": "no_outer_ellipse"})
            continue
        ellipse, _ = _derive_top_ellipse_from_inlay(crop, preliminary)
        if ellipse is None:
            rejected.append({"relative_path": record["relative_path"], "reason": "no_inlay_ellipse"})
            continue
        _, normalized = _ellipse_to_circle(crop, ellipse, args.normalized_size)
        center = center_number_view(normalized)
        mask = digit_mask(center)
        foreground_ratio = float(np.count_nonzero(mask)) / mask.size
        if not 0.008 <= foreground_ratio <= 0.22:
            rejected.append(
                {
                    "relative_path": record["relative_path"],
                    "reason": "invalid_digit_foreground_ratio",
                    "foreground_ratio": foreground_ratio,
                }
            )
            continue
        denomination = int(record["instances"][0]["denomination"])
        template_id = f"chip_{denomination}_chip_v2_{record['source_sha256'][:16]}"
        mask_path = masks_dir / f"{template_id}.png"
        normalized_path = normalized_dir / f"{template_id}.png"
        center_path = center_dir / f"{template_id}.png"
        for destination, payload in (
            (mask_path, mask),
            (normalized_path, normalized),
            (center_path, center),
        ):
            if not cv2.imwrite(str(destination), payload):
                raise RuntimeError(f"failed to write: {destination}")
        entry = {
            "template_id": template_id,
            "denomination": denomination,
            "source_sha256": record["source_sha256"],
            "source_group": record["capture_group"],
            "mask_file": mask_path.relative_to(library).as_posix(),
            "mask_sha256": sha256(mask_path),
            "foreground_ratio": round(foreground_ratio, 6),
            "ellipse_quality": ellipse.quality,
            "color_signature": [
                round(float(value), 6) for value in color_signature(normalized)
            ],
        }
        templates.append(entry)
        added.append(entry)

    counts = {
        str(value): sum(int(item["denomination"]) == value for item in templates)
        for value in (1, 5, 10, 20)
    }
    manifest = {
        "schema_version": "1.0",
        "asset_id": "chip-denomination-las-vegas-templates",
        "version": "chip-v2-development-20260724",
        "state": "development",
        "task": "fixed-design 10/20 matching; legacy 1/5 masks retained for compatibility",
        "design_scope": "user-provided LAS VEGAS POKER CLUB set",
        "active_denominations": [10, 20],
        "denominations": [1, 5, 10, 20],
        "template_size": [128, 128],
        "center_fraction": CENTER_FRACTION,
        "rotation_step_degrees": 10,
        "template_count": len(templates),
        "counts": counts,
        "base_manifest_sha256": sha256(base_manifest_path),
        "added_source_policy": "nine immutable reviewed straight captures; derived masks only",
        "templates": templates,
        "added_templates": [entry["template_id"] for entry in added],
        "rejected": rejected,
    }
    manifest_path = library / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "added": len(added),
                "rejected": len(rejected),
                "counts": counts,
                "manifest_sha256": sha256(manifest_path),
            }
        )
    )
    return 0 if len(added) == 9 and not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
