"""Fine-tune the local card detector without runtime model downloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

import torch
import yaml
from ultralytics import YOLO
import ultralytics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (
    ROOT
    / "models"
    / "assets"
    / "card_recognition"
    / "lgd-cards-gen3"
    / "best.pt"
)
DEFAULT_PROJECT = ROOT / "runs" / "card_finetune"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_names(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names)]
    if isinstance(raw_names, list):
        return [str(value) for value in raw_names]
    raise ValueError(f"invalid dataset names: {path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.005)
    parser.add_argument("--hsv-s", type=float, default=0.20)
    parser.add_argument("--hsv-v", type=float, default=0.20)
    args = parser.parse_args(argv)
    if (
        args.epochs <= 0
        or args.batch <= 0
        or args.workers < 0
        or args.patience < 0
    ):
        parser.error("epochs/batch must be positive and workers non-negative")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    weights = args.weights.resolve()
    data_path = args.data.resolve()
    project = args.project.resolve()
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        raise ValueError(f"weights must be an existing local .pt file: {weights}")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not torch.cuda.is_available() and str(args.device) != "cpu":
        raise RuntimeError("CUDA is unavailable; refusing unintended CPU training")

    model = YOLO(str(weights))
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    data_names = dataset_names(data_path)
    if model_names != data_names:
        raise ValueError("dataset class order does not match the pinned checkpoint")

    run_dir = project / args.name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    project.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema_version": "poker_dealer.card_finetune_launch.v1",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "project": str(project),
        "name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "fraction": args.fraction,
        "seed": args.seed,
        "patience": args.patience,
        "mosaic": args.mosaic,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "validation": (
            "disabled_by_cli"
            if args.disable_validation
            else "enabled_source_isolated_dataset_split"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    launch_path = project / f"{args.name}.launch.json"
    launch_path.write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model.train(
        data=str(data_path),
        project=str(project),
        name=args.name,
        exist_ok=False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        val=not args.disable_validation,
        plots=not args.disable_validation,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        cache=False,
        amp=False,
        mosaic=args.mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        close_mosaic=0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        erasing=0.0,
        multi_scale=False,
        pretrained=True,
        verbose=True,
    )
    result = {
        **launch,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
    }
    (run_dir / "poker_dealer_run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
