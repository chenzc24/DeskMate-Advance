"""Propose YOLO corner-glyph labels for filename-identified playing cards."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "poker_label" / "new big poker"
DEFAULT_MODEL = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "poker-dealer-v2"
    / "best.pt"
)
DEFAULT_CLASSES = DEFAULT_MODEL.parent / "model.classes.json"
DEFAULT_REVIEW = ROOT / "data" / "work" / "poker_big_data_v3" / "label_review"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUIT_CODES = {"梅花": "C", "方片": "D", "红桃": "H", "黑桃": "S"}
RANKS = {"A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def card_code_from_stem(stem: str) -> str:
    for suit_name, suit_code in SUIT_CODES.items():
        if stem.startswith(suit_name):
            rank = stem[len(suit_name) :].upper()
            if rank in RANKS:
                return f"{rank}{suit_code}"
    raise ValueError(f"filename does not identify a supported card: {stem}")


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def select_detections(
    detections: Sequence[dict[str, object]],
    expected_class_id: int,
    *,
    maximum: int = 2,
    supplement_min_confidence: float = 0.05,
) -> list[dict[str, object]]:
    matching = sorted(
        (
            detection
            for detection in detections
            if int(detection["class_id"]) == expected_class_id
        ),
        key=lambda detection: float(detection["confidence"]),
        reverse=True,
    )
    selected = [dict(detection) for detection in matching[:maximum]]
    if len(selected) >= maximum:
        return selected
    alternatives = sorted(
        (
            detection
            for detection in detections
            if int(detection["class_id"]) != expected_class_id
            and float(detection["confidence"]) >= supplement_min_confidence
        ),
        key=lambda detection: float(detection["confidence"]),
        reverse=True,
    )
    for candidate in alternatives:
        if all(
            box_iou(candidate["xyxy"], chosen["xyxy"]) < 0.20
            for chosen in selected
        ):
            selected.append({**candidate, "class_remapped": True})
        if len(selected) >= maximum:
            break
    return selected


def yolo_line(
    class_id: int, xyxy: Sequence[float], width: int, height: int
) -> str:
    x0 = max(0.0, min(float(width), float(xyxy[0])))
    y0 = max(0.0, min(float(height), float(xyxy[1])))
    x1 = max(0.0, min(float(width), float(xyxy[2])))
    y1 = max(0.0, min(float(height), float(xyxy[3])))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid predicted box: {xyxy}")
    return (
        f"{class_id} {((x0 + x1) * 0.5 / width):.8f} "
        f"{((y0 + y1) * 0.5 / height):.8f} "
        f"{((x1 - x0) / width):.8f} {((y1 - y0) / height):.8f}"
    )


def read_image(path: Path) -> np.ndarray:
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def write_review(
    image_path: Path,
    output_path: Path,
    selected: Sequence[dict[str, object]],
    expected_code: str,
) -> None:
    image = read_image(image_path)
    for index, detection in enumerate(selected, start=1):
        x0, y0, x1, y1 = (int(round(float(value))) for value in detection["xyxy"])
        remapped = bool(detection.get("class_remapped", False))
        color = (0, 165, 255) if remapped else (0, 255, 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
        text = (
            f"{expected_code} {float(detection['confidence']):.3f}"
            f"{' REMAP' if remapped else ''}"
        )
        cv2.putText(
            image,
            text,
            (max(4, x0), max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError(f"could not encode review image: {output_path}")
    encoded.tofile(output_path)


def write_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    columns: int = 4,
    tile_width: int = 320,
    tile_height: int = 240,
) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        28,
        dtype=np.uint8,
    )
    for index, image_path in enumerate(image_paths):
        image = read_image(image_path)
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        y0 = row * tile_height + (tile_height - resized.shape[0]) // 2
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not ok:
        raise ValueError(f"could not encode contact sheet: {output_path}")
    encoded.tofile(output_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.02)
    parser.add_argument("--supplement-min-confidence", type=float, default=0.05)
    parser.add_argument("--low-confidence", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.batch <= 0:
        parser.error("--imgsz and --batch must be positive")
    for value in (
        args.confidence,
        args.supplement_min_confidence,
        args.low_confidence,
    ):
        if not 0.0 <= value <= 1.0:
            parser.error("confidence thresholds must be in [0, 1]")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.resolve()
    labels = (args.labels or source / "labels").resolve()
    review = args.review.resolve()
    image_paths = sorted(
        (
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: card_code_from_stem(path.stem),
    )
    if not image_paths:
        raise ValueError(f"no card images found: {source}")
    existing_labels = list(labels.glob("*.txt")) if labels.exists() else []
    if existing_labels and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {len(existing_labels)} labels in {labels}"
        )
    labels.mkdir(parents=True, exist_ok=True)
    review_images = review / "images"
    review_images.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) != 52:
        raise ValueError("class sidecar must contain 52 class codes")
    class_to_id = {str(name): index for index, name in enumerate(class_names)}
    expected_codes = [card_code_from_stem(path.stem) for path in image_paths]
    if set(expected_codes) != set(class_names):
        raise ValueError("source filenames do not cover the pinned 52 classes exactly")

    model = YOLO(str(args.model.resolve()))
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if model_names != class_names:
        raise ValueError("model class order does not match the class sidecar")
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.confidence,
        iou=0.45,
        max_det=20,
        save=False,
        verbose=False,
    )

    records: list[dict[str, object]] = []
    review_paths: list[Path] = []
    flag_counts: Counter[str] = Counter()
    for image_path, expected_code, result in zip(
        image_paths, expected_codes, results, strict=True
    ):
        height, width = result.orig_shape
        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                detections.append(
                    {
                        "xyxy": [round(float(value), 4) for value in xyxy],
                        "confidence": round(float(confidence), 6),
                        "class_id": int(class_id),
                        "class_name": class_names[int(class_id)],
                    }
                )
        expected_id = class_to_id[expected_code]
        expected_detection_count = sum(
            int(detection["class_id"]) == expected_id for detection in detections
        )
        selected = select_detections(
            detections,
            expected_id,
            supplement_min_confidence=args.supplement_min_confidence,
        )
        selected = sorted(
            selected,
            key=lambda detection: (
                float(detection["xyxy"][1]),
                float(detection["xyxy"][0]),
            ),
        )
        flags: list[str] = []
        if not selected:
            flags.append("no_box")
        elif len(selected) == 1:
            flags.append("single_box")
        if expected_detection_count > 2:
            flags.append("more_than_two_expected")
        if any(bool(item.get("class_remapped", False)) for item in selected):
            flags.append("class_remapped")
        if selected and min(float(item["confidence"]) for item in selected) < args.low_confidence:
            flags.append("low_confidence")
        strong_conflicts = [
            item
            for item in detections
            if int(item["class_id"]) != expected_id
            and float(item["confidence"]) >= args.low_confidence
        ]
        if strong_conflicts:
            flags.append("strong_conflict")
        for flag in set(flags):
            flag_counts[flag] += 1

        label_path = labels / f"{image_path.stem}.txt"
        label_lines = [
            yolo_line(expected_id, item["xyxy"], width, height)
            for item in selected
        ]
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        review_path = review_images / f"{image_path.stem}.jpg"
        write_review(image_path, review_path, selected, expected_code)
        review_paths.append(review_path)
        records.append(
            {
                "image": image_path.name,
                "image_sha256": sha256_file(image_path),
                "label": label_path.name,
                "label_sha256": sha256_file(label_path),
                "expected_class_id": expected_id,
                "expected_class_name": expected_code,
                "detections": detections,
                "selected": selected,
                "flags": flags,
            }
        )

    write_contact_sheet(review_paths, review / "contact_sheet.jpg")
    report = {
        "schema_version": "poker_dealer.card_auto_label_review.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "model": {
            "path": str(args.model.resolve()),
            "sha256": sha256_file(args.model.resolve()),
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "supplement_min_confidence": args.supplement_min_confidence,
            "low_confidence": args.low_confidence,
        },
        "summary": {
            "images": len(records),
            "labels": len(records),
            "selected_boxes": sum(len(record["selected"]) for record in records),
            "two_box_images": sum(len(record["selected"]) == 2 for record in records),
            "one_box_images": sum(len(record["selected"]) == 1 for record in records),
            "zero_box_images": sum(len(record["selected"]) == 0 for record in records),
            "flag_counts": dict(sorted(flag_counts.items())),
            "class_coverage": len({record["expected_class_id"] for record in records}),
        },
        "records": records,
    }
    review.mkdir(parents=True, exist_ok=True)
    (review / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(report["summary"]["zero_box_images"] > 0)


if __name__ == "__main__":
    sys.exit(main())
