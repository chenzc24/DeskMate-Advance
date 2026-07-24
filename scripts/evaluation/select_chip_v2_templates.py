"""Select only new value-10 templates that improve the oblique evaluation."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil

from evaluate_chip_v2_denomination import ROOT, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/chips/2026-07-24-chip-v2-source",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/reviewed_annotations_candidate.json",
    )
    parser.add_argument(
        "--base-library",
        type=Path,
        default=ROOT
        / "models/assets/chip_recognition/las-vegas-denomination-templates-v1",
    )
    parser.add_argument(
        "--candidate-library",
        type=Path,
        default=ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/denomination_library/library",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/work/chips/2026-07-24-chip-v2-optimization/selected_denomination_library",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    base_manifest = json.loads((args.base_library / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads(
        (args.candidate_library / "manifest.json").read_text(encoding="utf-8")
    )
    additions = [
        item
        for item in candidate_manifest["templates"]
        if int(item["denomination"]) == 10
        and str(item["template_id"]).startswith("chip_10_chip_v2_")
    ]
    if len(additions) != 4:
        raise SystemExit(f"expected four new value-10 templates, got {len(additions)}")

    search_root = args.output.parent / "template_subset_search"
    if search_root.exists():
        if search_root.parent != args.output.parent.resolve():
            raise SystemExit(f"refusing unexpected search cleanup: {search_root}")
        shutil.rmtree(search_root)
    search_root.mkdir(parents=True)
    results = []
    for size in range(len(additions) + 1):
        for subset in combinations(additions, size):
            subset_ids = [item["template_id"] for item in subset]
            subset_dir = search_root / ("none" if not subset_ids else "__".join(subset_ids))
            masks_dir = subset_dir / "masks"
            masks_dir.mkdir(parents=True)
            templates = [dict(item) for item in base_manifest["templates"]] + [
                dict(item) for item in subset
            ]
            for item in templates:
                source_root = (
                    args.candidate_library
                    if str(item["template_id"]).startswith("chip_10_chip_v2_")
                    else args.base_library
                )
                source = source_root / item["mask_file"]
                destination = subset_dir / item["mask_file"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            manifest = {
                **base_manifest,
                "version": "chip-v2-subset-search",
                "state": "development",
                "active_denominations": [10, 20],
                "template_count": len(templates),
                "templates": templates,
            }
            (subset_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            metrics = evaluate(subset_dir, args.raw_root, records)
            results.append(
                {
                    "selected_template_ids": subset_ids,
                    "correct": metrics["correct"],
                    "wrong": metrics["wrong"],
                    "accepted": metrics["accepted"],
                    "rejected": metrics["rejected"],
                    "overall_correct_rate": metrics["overall_correct_rate"],
                    "accepted_accuracy": metrics["accepted_accuracy"],
                    "library": str(subset_dir),
                }
            )
    results.sort(
        key=lambda item: (
            int(item["correct"]),
            -int(item["wrong"]),
            int(item["accepted"]),
            -len(item["selected_template_ids"]),
        ),
        reverse=True,
    )
    winner = results[0]
    source_library = Path(winner["library"])
    if args.output.exists():
        if args.output.parent != (
            ROOT / "data/work/chips/2026-07-24-chip-v2-optimization"
        ).resolve():
            raise SystemExit(f"refusing unexpected output cleanup: {args.output}")
        shutil.rmtree(args.output)
    shutil.copytree(source_library, args.output)
    report = {"schema_version": "1.0", "winner": winner, "all_results": results}
    report_path = args.output.parent / "template_selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected_manifest_path = args.output / "manifest.json"
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    selected_manifest.update(
        {
            "version": "chip-v2-selected-development-20260724",
            "state": "development",
            "created_utc": date.today().isoformat(),
            "active_denominations": [10, 20],
            "selection_scope": "10_and_20_only",
            "selection_evidence": {
                "source_instances": 66,
                "selected_template_ids": winner["selected_template_ids"],
                "correct": winner["correct"],
                "wrong": winner["wrong"],
                "accepted": winner["accepted"],
                "rejected": winner["rejected"],
                "overall_correct_rate": winner["overall_correct_rate"],
                "accepted_accuracy": winner["accepted_accuracy"],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
        }
    )
    selected_manifest_path.write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"winner": winner, "report": str(report_path.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
