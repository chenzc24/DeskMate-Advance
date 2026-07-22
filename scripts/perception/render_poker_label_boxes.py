"""Render YOLO card labels as reviewable bounding-box overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


SUIT_CODES = {"梅花": "C", "方片": "D", "红桃": "H", "黑桃": "S"}
SUIT_ORDER = tuple(SUIT_CODES)
RANK_ORDER = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
BOX_COLORS = ("#00E5FF", "#FFEA00", "#FF4D9D", "#76FF03")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageResult:
    image: str
    label: str
    expected_class: str | None
    class_ids: list[int]
    box_count: int
    width: int
    height: int
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"class sidecar must be a non-empty JSON string list: {path}")
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_class_from_stem(stem: str) -> str | None:
    for suit_name, suit_code in SUIT_CODES.items():
        if stem.startswith(suit_name):
            rank = stem[len(suit_name) :]
            if rank in RANK_ORDER:
                return f"{rank}{suit_code}"
    return None


def load_boxes(path: Path) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected 5 fields, found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            errors.append(f"line {line_number}: non-numeric field ({exc})")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: non-finite coordinate")
            continue
        x_center, y_center, width, height = values
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width and height must be positive")
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"line {line_number}: normalized values must be in [0, 1]")
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        if left < 0 or top < 0 or right > 1 or bottom > 1:
            errors.append(f"line {line_number}: box extends outside image")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return boxes, errors


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def pixel_box(box: YoloBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left = round((box.x_center - box.width / 2) * image_width)
    top = round((box.y_center - box.height / 2) * image_height)
    right = round((box.x_center + box.width / 2) * image_width)
    bottom = round((box.y_center + box.height / 2) * image_height)
    return left, top, right, bottom


def draw_tag(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, label_font: ImageFont.ImageFont) -> None:
    x, y = xy
    text_box = draw.textbbox((x, y), text, font=label_font, stroke_width=1)
    padding = 4
    background = (
        text_box[0] - padding,
        text_box[1] - padding,
        text_box[2] + padding,
        text_box[3] + padding,
    )
    draw.rectangle(background, fill="#101010", outline=color, width=2)
    draw.text((x, y), text, fill=color, font=label_font, stroke_width=1, stroke_fill="#000000")


def render_one(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    classes: list[str],
) -> tuple[ImageResult, Image.Image]:
    boxes, errors = load_boxes(label_path)
    expected_class = expected_class_from_stem(image_path.stem)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_width, image_height = image.size
    draw = ImageDraw.Draw(image)
    title_font = font(max(22, image_width // 34))
    label_font = font(max(16, image_width // 52))

    class_ids = sorted({box.class_id for box in boxes})
    for class_id in class_ids:
        if class_id < 0 or class_id >= len(classes):
            errors.append(f"class id {class_id} outside [0, {len(classes) - 1}]")
    if len(class_ids) > 1:
        errors.append(f"multiple class ids in one card image: {class_ids}")
    actual_classes = [classes[class_id] for class_id in class_ids if 0 <= class_id < len(classes)]
    if expected_class is None:
        errors.append("filename does not map to a supported card class")
    elif actual_classes != [expected_class]:
        errors.append(f"filename expects {expected_class}, label contains {actual_classes}")

    title = f"{image_path.stem}  expected={expected_class or '?'}  boxes={len(boxes)}"
    title_bounds = draw.textbbox((12, 10), title, font=title_font, stroke_width=1)
    draw.rectangle((6, 4, title_bounds[2] + 8, title_bounds[3] + 8), fill="#101010")
    draw.text((12, 10), title, fill="#FFFFFF", font=title_font, stroke_width=1, stroke_fill="#000000")

    for index, box in enumerate(boxes, start=1):
        color = BOX_COLORS[(index - 1) % len(BOX_COLORS)]
        left, top, right, bottom = pixel_box(box, image_width, image_height)
        draw.rectangle((left, top, right, bottom), outline=color, width=max(3, image_width // 250))
        class_name = classes[box.class_id] if 0 <= box.class_id < len(classes) else "INVALID"
        tag_y = max(title_bounds[3] + 14 if top < title_bounds[3] + 14 else 2, top - label_font.size - 12)
        draw_tag(draw, (max(4, left), tag_y), f"#{index} id={box.class_id} {class_name}", color, label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    result = ImageResult(
        image=image_path.name,
        label=label_path.name,
        expected_class=expected_class,
        class_ids=class_ids,
        box_count=len(boxes),
        width=image_width,
        height=image_height,
        errors=errors,
    )
    return result, image


def card_sort_key(stem: str) -> tuple[int, int, str]:
    expected = expected_class_from_stem(stem)
    if expected is None:
        return len(SUIT_ORDER), len(RANK_ORDER), stem
    suit_code = expected[-1]
    rank = expected[:-1]
    suit_index = tuple(SUIT_CODES.values()).index(suit_code)
    return suit_index, RANK_ORDER.index(rank), stem


def make_contact_sheet(items: Iterable[tuple[str, Image.Image]], output_path: Path) -> None:
    item_list = list(items)
    if not item_list:
        return
    columns = 3
    thumb_width, thumb_height = 488, 360
    caption_height = 42
    rows = math.ceil(len(item_list) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + caption_height)), "#202020")
    sheet_draw = ImageDraw.Draw(sheet)
    caption_font = font(24)
    for index, (caption, image) in enumerate(item_list):
        column = index % columns
        row = index // columns
        x = column * thumb_width
        y = row * (thumb_height + caption_height)
        thumbnail = image.copy()
        thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x, y))
        sheet_draw.text((x + 8, y + thumb_height + 6), caption, fill="#FFFFFF", font=caption_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def main() -> int:
    args = parse_args()
    image_dir = args.dataset / "images"
    label_dir = args.dataset / "labels"
    output_images = args.output / "images"
    classes = load_classes(args.classes)
    image_paths = sorted(image_dir.glob("*.png"), key=lambda path: card_sort_key(path.stem))
    label_paths = {path.stem: path for path in label_dir.glob("*.txt")}

    results: list[ImageResult] = []
    rendered: dict[str, Image.Image] = {}
    missing_labels: list[str] = []
    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path.name)
            continue
        result, image = render_one(
            image_path,
            label_path,
            output_images / image_path.name,
            classes,
        )
        results.append(result)
        rendered[image_path.stem] = image

    image_stems = {path.stem for path in image_paths}
    orphan_labels = sorted(path.name for stem, path in label_paths.items() if stem not in image_stems)
    for suit_name in SUIT_ORDER:
        suit_items = [
            (stem, rendered[stem])
            for rank in RANK_ORDER
            if (stem := f"{suit_name}{rank}") in rendered
        ]
        make_contact_sheet(suit_items, args.output / f"contact_sheet_{suit_name}.png")

    source_files = sorted((*image_paths, *label_paths.values()), key=lambda path: path.as_posix())
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "files": [
            {
                "path": path.relative_to(args.dataset).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    (args.output / "input_manifest.json").write_bytes(manifest_bytes)

    report = {
        "dataset": str(args.dataset.resolve()),
        "classes": str(args.classes.resolve()),
        "class_count": len(classes),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "rendered_count": len(results),
        "total_boxes": sum(result.box_count for result in results),
        "box_count_distribution": {
            str(count): sum(result.box_count == count for result in results)
            for count in sorted({result.box_count for result in results})
        },
        "non_two_box_images": [result.image for result in results if result.box_count != 2],
        "input_manifest": "input_manifest.json",
        "input_manifest_sha256": manifest_sha256,
        "files_with_errors": sum(bool(result.errors) for result in results),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "images": [asdict(result) for result in results],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in report if key != "images"}, ensure_ascii=False, indent=2))
    return 1 if missing_labels or orphan_labels or report["files_with_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
