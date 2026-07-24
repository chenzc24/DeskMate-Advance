"""Fine-tune the pinned YOLO11n COCO base on target-camera poker chips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import torch
import ultralytics
import yaml
from ultralytics import YOLO


WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE.parent
DEFAULT_CONFIG = WORKSPACE / "chip_yolo11n_finetune_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_dataset_yaml(path: Path, dataset_config: dict[str, object]) -> None:
    if not path.is_file():
        raise SystemExit(
            "labeled chip dataset is not ready: "
            f"{path}\nCollect and annotate independent target-camera sessions first."
        )
    contents = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contents, dict):
        raise SystemExit("dataset YAML must contain a mapping")
    missing = [
        name
        for name in dataset_config["required_splits"]
        if name not in contents
    ]
    if missing:
        raise SystemExit(
            "dataset YAML is missing required splits: "
            + ", ".join(missing)
        )
    names = contents.get("names")
    normalized_names = (
        {str(key): value for key, value in names.items()}
        if isinstance(names, dict)
        else None
    )
    if normalized_names != dataset_config["class_names"]:
        raise SystemExit(
            "dataset class map does not match the pinned training configuration: "
            f"expected {dataset_config['class_names']}, got {normalized_names}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on a small fraction after the dataset gate passes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0":
        raise SystemExit("unsupported training config schema")
    allowed_statuses = {
        "development_waiting_for_labeled_sessions",
        "development_fit_only_no_independent_validation",
        "development_no_target_camera_holdout_no_negative_samples",
        "development_hard_negative_fit_only_no_target_holdout",
        "development_target_capture_holdout",
    }
    if config.get("status") not in allowed_statuses:
        raise SystemExit("unexpected chip training status")

    base_config = config["base_model"]
    dataset_config = config["dataset"]
    framework = config["framework"]
    if framework.get("polars_skip_cpu_check"):
        # Polars documents this for false-positive/unknown feature detection.
        # It only affects the CSV metrics reader used when saving checkpoints.
        os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
    train = dict(config["train"])
    base_path = root_path(base_config["path"])
    dataset_yaml = root_path(dataset_config["yaml_path"])
    manifest_path = (
        root_path(dataset_config["manifest_path"])
        if dataset_config.get("manifest_path")
        else None
    )

    if not base_path.is_file():
        raise SystemExit(f"YOLO11n base is missing: {base_path}")
    actual_hash = sha256(base_path)
    expected_hash = base_config["sha256"].lower()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"YOLO11n SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise SystemExit(f"dataset manifest is missing: {manifest_path}")
        actual_manifest_hash = sha256(manifest_path)
        expected_manifest_hash = str(dataset_config["manifest_sha256"]).lower()
        if actual_manifest_hash != expected_manifest_hash:
            raise SystemExit(
                "dataset manifest SHA-256 mismatch: expected "
                f"{expected_manifest_hash}, got {actual_manifest_hash}"
            )
    if ultralytics.__version__ != framework["ultralytics"]:
        raise SystemExit(
            "ultralytics version mismatch: expected "
            f"{framework['ultralytics']}, got {ultralytics.__version__}"
        )
    if torch.__version__ != framework["torch"]:
        raise SystemExit(
            f"torch version mismatch: expected {framework['torch']}, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing an accidental CPU training run")

    validate_dataset_yaml(dataset_yaml, dataset_config)
    device = int(framework["device"])
    if device >= torch.cuda.device_count():
        raise SystemExit(f"configured CUDA device does not exist: {device}")

    project = root_path(train.pop("project"))
    name = str(train.pop("name"))
    if args.smoke:
        project = (ROOT / "runs/chip_finetune_smoke").resolve()
        name += "_smoke"
        train.update(
            {
                "epochs": 1,
                "batch": 4,
                "workers": 0,
                "fraction": 0.05,
                "save": False,
                "save_period": -1,
            }
        )
    save_dir = project / name
    if save_dir.exists():
        raise SystemExit(f"run directory already exists; refusing to mix runs: {save_dir}")

    model = YOLO(str(base_path))
    results = model.train(
        data=str(dataset_yaml),
        device=device,
        project=str(project),
        name=name,
        exist_ok=False,
        **train,
    )
    save_dir = Path(results.save_dir).resolve()
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "training_id": config["training_id"],
        "status": "smoke_only" if args.smoke else config["status"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "base_model_sha256": actual_hash,
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": sha256(dataset_yaml),
        "dataset_manifest": str(manifest_path) if manifest_path is not None else None,
        "dataset_manifest_sha256": (
            sha256(manifest_path) if manifest_path is not None else None
        ),
        "split_policy": dataset_config["split_policy"],
        "independent_validation": bool(dataset_config.get("independent_validation", False)),
        "metrics_policy": dataset_config.get(
            "metrics_policy",
            "development only; do not report training fit as held-out accuracy",
        ),
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "run_dir": str(save_dir),
    }
    for filename in ("weights/best.pt", "weights/last.pt"):
        path = save_dir / filename
        if path.is_file():
            metadata[filename.replace("/", "_") + "_sha256"] = sha256(path)
    (save_dir / "poker_dealer_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
