"""Validate immutable action media metadata and assign participant-safe splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_dealer.evaluation import (
    assign_participant_splits,
    canonical_sha256,
    validate_action_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", default="stage2a-action-split-v1")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        errors = validate_action_manifest(
            manifest, root=ROOT, verify_files=args.verify_files
        )
        if errors:
            raise ValueError("; ".join(errors))
        if args.validate_only:
            report = {
                "result": "PASS",
                "records": len(manifest["records"]),
                "manifest_sha256": canonical_sha256(manifest),
            }
        else:
            if args.output is None:
                raise ValueError("--output is required unless --validate-only is used")
            resolved = assign_participant_splits(manifest, seed=args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(resolved, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            report = {
                "result": "PASS",
                "records": len(resolved["records"]),
                "source_records_sha256": resolved["source_records_sha256"],
                "resolved_manifest_sha256": canonical_sha256(resolved),
                "output": str(args.output),
            }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
