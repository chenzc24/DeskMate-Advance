"""Package a derived chip-template library as a compact runtime asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2


EXPECTED_DENOMINATIONS = (1, 5, 10, 20)
EXPECTED_MASK_SHAPE = (128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_library(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if tuple(source_manifest["denominations"]) != EXPECTED_DENOMINATIONS:
        raise ValueError(
            f"unexpected denominations: {source_manifest['denominations']}"
        )

    masks_output = output / "masks"
    masks_output.mkdir(parents=True)
    packaged_templates: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    mask_records: list[dict[str, object]] = []
    counts = {str(value): 0 for value in EXPECTED_DENOMINATIONS}

    for item in source_manifest["templates"]:
        template_id = str(item["template_id"])
        denomination = int(item["denomination"])
        if denomination not in EXPECTED_DENOMINATIONS:
            raise ValueError(f"unexpected denomination for {template_id}")
        source_mask = source / str(item["mask_file"])
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask is None or mask.shape != EXPECTED_MASK_SHAPE:
            shape = None if mask is None else mask.shape
            raise ValueError(f"invalid mask {source_mask}: {shape}")
        color_signature = [float(value) for value in item["color_signature"]]
        if len(color_signature) != 12:
            raise ValueError(f"invalid colour signature for {template_id}")

        target_mask = masks_output / f"{template_id}.png"
        shutil.copy2(source_mask, target_mask)
        mask_hash = sha256(target_mask)
        source_hash = str(item["source_sha256"])
        mask_file = target_mask.relative_to(output).as_posix()
        packaged_templates.append(
            {
                "template_id": template_id,
                "denomination": denomination,
                "source_sha256": source_hash,
                "mask_file": mask_file,
                "mask_sha256": mask_hash,
                "color_signature": color_signature,
            }
        )
        source_records.append(
            {
                "template_id": template_id,
                "denomination": denomination,
                "source_sha256": source_hash,
            }
        )
        mask_records.append(
            {
                "mask_file": mask_file,
                "mask_sha256": mask_hash,
            }
        )
        counts[str(denomination)] += 1

    packaged_templates.sort(key=lambda item: str(item["template_id"]))
    source_records.sort(key=lambda item: str(item["template_id"]))
    mask_records.sort(key=lambda item: str(item["mask_file"]))
    manifest = {
        "schema_version": "1.0",
        "asset_id": "chip-denomination-las-vegas-templates",
        "version": "v1-20260723",
        "state": "development",
        "task": "fixed-design 1/5/10/20 chip denomination matching",
        "design_scope": (
            "user-provided LAS VEGAS POKER CLUB chip set only; "
            "not a generic casino-chip denomination map"
        ),
        "source_policy": (
            "project-captured raw front views remain private and are not "
            "distributed; only derived 128x128 binary masks and colour "
            "signatures are packaged"
        ),
        "denominations": list(EXPECTED_DENOMINATIONS),
        "template_size": list(EXPECTED_MASK_SHAPE),
        "center_fraction": float(source_manifest["center_fraction"]),
        "rotation_step_degrees": 10,
        "template_count": len(packaged_templates),
        "counts": counts,
        "source_template_set_sha256": canonical_sha256(source_records),
        "mask_set_sha256": canonical_sha256(mask_records),
        "templates": packaged_templates,
    }
    (output / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = package_library(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "template_count": manifest["template_count"],
                "counts": manifest["counts"],
                "source_template_set_sha256": (
                    manifest["source_template_set_sha256"]
                ),
                "mask_set_sha256": manifest["mask_set_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
