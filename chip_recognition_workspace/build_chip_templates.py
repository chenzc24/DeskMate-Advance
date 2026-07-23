"""Build fixed-design chip denomination masks from front-view source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from chip_template_matcher import (
    CENTER_FRACTION,
    DENOMINATIONS,
    center_number_view,
    color_signature,
    digit_mask,
)
from rectify_chip_images import (
    DEFAULT_MODEL,
    _derive_top_ellipse_from_inlay,
    _ellipse_to_circle,
    _expand_bbox,
    _fit_top_ellipse,
    _grabcut_chip_mask,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "raw" / "chip_templates"
DEFAULT_OUTPUT = (
    ROOT / "data" / "work" / "chips" / "2026-07-23-template-matching"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--padding", type=float, default=0.14)
    parser.add_argument("--normalized-size", type=int, default=384)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sources(root: Path) -> list[tuple[int, Path]]:
    records: list[tuple[int, Path]] = []
    for denomination in DENOMINATIONS:
        directory = root / str(denomination)
        if not directory.is_dir():
            raise SystemExit(f"denomination directory is missing: {directory}")
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            raise SystemExit(f"no source images found in: {directory}")
        records.extend((denomination, path) for path in paths)
    return records


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input directory is missing: {args.input}")
    if not args.model.is_file():
        raise SystemExit(f"model is missing: {args.model}")
    sources = _sources(args.input)
    library = args.output / "library"
    normalized_dir = args.output / "normalized"
    center_dir = args.output / "center_views"
    mask_dir = library / "masks"
    for directory in (library, normalized_dir, center_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model.resolve()))
    predictions = model.predict(
        [str(path) for _, path in sources],
        conf=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    templates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for (denomination, source), prediction in zip(sources, predictions):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"source": _relative(source), "reason": "read_failed"})
            continue
        if len(prediction.boxes) == 0:
            rejected.append({"source": _relative(source), "reason": "no_detection"})
            continue
        confidences = prediction.boxes.conf.detach().cpu().numpy()
        selected = int(np.argmax(confidences))
        raw_box = prediction.boxes.xyxy[selected].detach().cpu().numpy()
        expanded = _expand_bbox(raw_box, image.shape, args.padding)
        x1, y1, x2, y2 = expanded
        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
        outer_mask = _grabcut_chip_mask(crop)
        preliminary, _ = _fit_top_ellipse(outer_mask)
        if preliminary is None:
            rejected.append(
                {"source": _relative(source), "reason": "no_outer_ellipse"}
            )
            continue
        ellipse, _ = _derive_top_ellipse_from_inlay(crop, preliminary)
        if ellipse is None:
            rejected.append(
                {"source": _relative(source), "reason": "no_center_inlay_ellipse"}
            )
            continue
        _, normalized = _ellipse_to_circle(crop, ellipse, args.normalized_size)
        center = center_number_view(normalized)
        mask = digit_mask(center)
        foreground_ratio = float(np.count_nonzero(mask)) / mask.size
        if not 0.008 <= foreground_ratio <= 0.22:
            rejected.append(
                {
                    "source": _relative(source),
                    "reason": "invalid_digit_foreground_ratio",
                    "foreground_ratio": round(foreground_ratio, 6),
                }
            )
            continue

        template_id = f"chip_{denomination}_{source.stem}"
        normalized_path = normalized_dir / f"{template_id}.png"
        center_path = center_dir / f"{template_id}.png"
        mask_path = mask_dir / f"{template_id}.png"
        for path, output in (
            (normalized_path, normalized),
            (center_path, center),
            (mask_path, mask),
        ):
            if not cv2.imwrite(str(path), output):
                raise RuntimeError(f"failed to write: {path}")
        templates.append(
            {
                "template_id": template_id,
                "denomination": denomination,
                "source": _relative(source),
                "source_sha256": _sha256(source),
                "detector_confidence": round(float(confidences[selected]), 6),
                "detections_in_source": len(prediction.boxes),
                "selected_bbox_xyxy": [round(float(value), 3) for value in raw_box],
                "ellipse_quality": ellipse.quality,
                "foreground_ratio": round(foreground_ratio, 6),
                "color_signature": [
                    round(float(value), 6) for value in color_signature(normalized)
                ],
                "mask_file": str(mask_path.relative_to(library)).replace("\\", "/"),
                "normalized_file": _relative(normalized_path),
                "center_file": _relative(center_path),
            }
        )

    manifest = {
        "schema_version": "1.1",
        "task": "fixed_design_chip_denomination_templates",
        "denominations": list(DENOMINATIONS),
        "source_root": _relative(args.input),
        "model": _relative(args.model),
        "center_fraction": CENTER_FRACTION,
        "template_count": len(templates),
        "counts": {
            str(value): sum(item["denomination"] == value for item in templates)
            for value in DENOMINATIONS
        },
        "templates": templates,
        "rejected": rejected,
    }
    manifest_path = library / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if templates and not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
