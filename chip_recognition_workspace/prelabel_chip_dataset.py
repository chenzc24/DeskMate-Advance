"""Create reviewable foreground-chip candidate boxes without changing source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "chips"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "chips" / "2026-07-22-denominations"
MODEL_DIR = (
    ROOT
    / "chip_recognition_workspace"
    / "pretrained"
    / "Shiranai17-poker-chips-dice-openvino-int8"
)
MODEL_XML = MODEL_DIR / "best.xml"
CLASS_MAP = {"1": 0, "5": 1, "10": 2, "20": 3}
CLASS_NAMES = {0: "chip_1", 1: "chip_5", 2: "chip_10", 3: "chip_20"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INPUT_SIZE = 640


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(image: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height))
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), (scale, pad_x, pad_y)


def decode_chip_candidates(
    output: np.ndarray,
    image_shape: tuple[int, int],
    letterbox: tuple[float, int, int],
) -> list[dict[str, object]]:
    values = np.asarray(output, dtype=np.float32)
    if values.shape == (1, 6, 8400):
        values = values[0].T
    elif values.shape == (1, 8400, 6):
        values = values[0]
    else:
        raise RuntimeError(f"unexpected old model output shape: {values.shape}")
    height, width = image_shape
    scale, pad_x, pad_y = letterbox
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for row in values:
        confidence = float(row[5])
        if confidence < 0.05:
            continue
        center_x, center_y, box_width, box_height = map(float, row[:4])
        x1 = max(0, int(round((center_x - box_width / 2 - pad_x) / scale)))
        y1 = max(0, int(round((center_y - box_height / 2 - pad_y) / scale)))
        x2 = min(width, int(round((center_x + box_width / 2 - pad_x) / scale)))
        y2 = min(height, int(round((center_y + box_height / 2 - pad_y) / scale)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.05, 0.45)
    return [
        {"confidence": confidences[int(index)], "bbox_xywh": boxes[int(index)]}
        for index in np.asarray(indices).reshape(-1)
    ]


def candidate_score(candidate: dict[str, object], width: int, height: int) -> float:
    x, y, box_width, box_height = candidate["bbox_xywh"]
    area_fraction = (box_width * box_height) / (width * height)
    aspect = box_width / box_height
    if not 0.002 <= area_fraction <= 0.18 or not 0.5 <= aspect <= 3.5:
        return -math.inf
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    center_distance = math.hypot(center_x - 0.5, center_y - 0.55)
    center_score = max(0.0, 1.0 - center_distance / 0.75)
    size_score = math.exp(-abs(math.log(area_fraction / 0.045)))
    return 2.0 * float(candidate["confidence"]) + center_score + size_score


def build_contact_sheet(records: list[dict[str, object]], output_path: Path) -> None:
    columns, rows = 4, 4
    cell_width, cell_height = 320, 270
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 28, np.uint8)
    for index, record in enumerate(records):
        image = cv2.imread(str(ROOT / record["source_path"]), cv2.IMREAD_COLOR)
        candidate_box = record["candidate_bbox_xywh"]
        if candidate_box is not None:
            x, y, width, height = candidate_box
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)
            label = (
                f"{record['record_id']} {record['class_name']} "
                f"c={record['candidate_confidence']:.2f}"
            )
        else:
            label = f"{record['record_id']} {record['class_name']} MISSING"
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        scale = min(cell_width / image.shape[1], (cell_height - 28) / image.shape[0])
        resized = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        )
        row, column = divmod(index, columns)
        x_offset = column * cell_width + (cell_width - resized.shape[1]) // 2
        y_offset = row * cell_height + 28
        sheet[
            y_offset : y_offset + resized.shape[0],
            x_offset : x_offset + resized.shape[1],
        ] = resized
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write QA contact sheet: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    prelabel_dir = output / "prelabel"
    qa_dir = prelabel_dir / "qa"
    manifest_path = prelabel_dir / "prelabel_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"prelabel snapshot already exists; refusing overwrite: {manifest_path}")
    qa_dir.mkdir(parents=True, exist_ok=True)
    if any(qa_dir.iterdir()):
        raise SystemExit(f"QA output directory is not empty: {qa_dir}")

    core = ov.Core()
    compiled = core.compile_model(core.read_model(MODEL_XML), "CPU")
    output_layer = compiled.output(0)
    records: list[dict[str, object]] = []
    record_number = 0
    for folder_name, class_id in CLASS_MAP.items():
        image_paths = sorted(
            path
            for path in (source / folder_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"unreadable image: {image_path}")
            height, width = image.shape[:2]
            tensor, letterbox = preprocess(image)
            candidates = decode_chip_candidates(
                compiled([tensor])[output_layer], (height, width), letterbox
            )
            scored = [
                (candidate_score(candidate, width, height), candidate)
                for candidate in candidates
            ]
            score, selected = max(scored, key=lambda item: item[0], default=(-math.inf, None))
            record_number += 1
            records.append(
                {
                    "record_id": f"C{record_number:03d}",
                    "source_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_sha256": sha256_file(image_path),
                    "width": width,
                    "height": height,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "candidate_bbox_xywh": (
                        selected["bbox_xywh"]
                        if selected is not None and math.isfinite(score)
                        else None
                    ),
                    "candidate_confidence": (
                        selected["confidence"]
                        if selected is not None and math.isfinite(score)
                        else None
                    ),
                    "selection_score": score if math.isfinite(score) else None,
                    "candidate_status": "requires_human_visual_review",
                }
            )

    source_snapshot = json.dumps(
        [
            {
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
            }
            for record in records
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "prelabel_requires_human_visual_review",
        "source_root": str(source),
        "source_images_modified": False,
        "source_image_count": len(records),
        "source_snapshot_sha256": sha256_bytes(source_snapshot),
        "class_names": CLASS_NAMES,
        "annotation_scope": "one deliberately presented foreground chip per image",
        "warning": (
            "Background chip stacks are not labeled. Final training views must crop them "
            "out or acquire complete multi-chip labels."
        ),
        "prelabel_model": {
            "role": "candidate_box_proposal_only",
            "admission_status": "rejected_runtime_baseline",
            "xml_sha256": sha256_file(MODEL_DIR / "best.xml"),
            "bin_sha256": sha256_file(MODEL_DIR / "best.bin"),
        },
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for sheet_index, start in enumerate(range(0, len(records), 16), 1):
        build_contact_sheet(
            records[start : start + 16], qa_dir / f"prelabel_contact_{sheet_index:02d}.jpg"
        )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records": len(records),
                "source_snapshot_sha256": manifest["source_snapshot_sha256"],
                "qa_sheets": math.ceil(len(records) / 16),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
