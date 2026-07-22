"""Train and export the optional compact landmark TCN from ignored derived views."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from poker_dealer.training import (
    ActionTcnConfig,
    TrainingDependencyError,
    build_compact_tcn,
    make_sequence_windows,
    summarize_view_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/training/action_tcn_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_manifest", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_arrays(
    manifest: dict[str, object], config: ActionTcnConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    features_by_split: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    masks_by_split: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    labels_by_split: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    label_index = {label: index for index, label in enumerate(config.labels)}
    for record in manifest["records"]:  # type: ignore[index]
        path = ROOT / str(record["view_path"])
        if not path.is_file() or _sha256(path) != record["view_sha256"]:
            raise ValueError(f"derived view missing or hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            windows, masks, _starts = make_sequence_windows(
                archive["features"],
                archive["valid_mask"],
                sequence_length=config.sequence_length,
                stride=config.window_stride,
            )
        if len(windows) == 0:
            raise ValueError(f"derived view produced no windows: {path}")
        split = str(record["split"])
        features_by_split[split].append(windows)
        masks_by_split[split].append(masks)
        labels_by_split[split].append(
            np.full((len(windows),), label_index[str(record["label"])], dtype=np.int64)
        )
    empty_features = np.empty((0, config.sequence_length, config.feature_dim), np.float32)
    empty_masks = np.empty((0, config.sequence_length), np.bool_)
    empty_labels = np.empty((0,), np.int64)
    return (
        {
            split: np.concatenate(values) if values else empty_features.copy()
            for split, values in features_by_split.items()
        },
        {
            split: np.concatenate(values) if values else empty_masks.copy()
            for split, values in masks_by_split.items()
        },
        {
            split: np.concatenate(values) if values else empty_labels.copy()
            for split, values in labels_by_split.items()
        },
    )


def _classification_metrics(
    truth: np.ndarray, predicted: np.ndarray, labels: tuple[str, ...]
) -> dict[str, object]:
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        confusion[int(actual), int(guess)] += 1
    per_label: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_label[label] = {
            "support": int(confusion[index, :].sum()),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {"confusion_matrix": confusion.tolist(), "per_label": per_label}


def main() -> int:
    args = parse_args()
    try:
        config = ActionTcnConfig.from_json(args.config)
        if args.check_config:
            print(
                json.dumps(
                    {
                        "result": "PASS",
                        "status": config.status,
                        "model_id": config.model_id,
                        "labels": config.labels,
                        "torch_required_for_training": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.view_manifest is None:
            raise ValueError("view_manifest is required unless --check-config is used")
        manifest = json.loads(args.view_manifest.read_text(encoding="utf-8"))
        summary = summarize_view_manifest(manifest, config)
        features, masks, labels = _load_arrays(manifest, config)
        window_summary = {
            split: {
                "windows": int(len(values)),
                "labels": dict(
                    sorted(
                        Counter(config.labels[int(item)] for item in labels[split]).items()
                    )
                ),
            }
            for split, values in features.items()
        }
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "result": "PASS",
                        "status": "data_and_windows_valid_not_trained",
                        "view_summary": summary,
                        "window_summary": window_summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if any(len(features[split]) == 0 for split in ("train", "validation", "test")):
            raise ValueError("training requires non-empty train, validation and test splits")

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(config.training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_compact_tcn(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train_counts = np.bincount(labels["train"], minlength=len(config.labels))
        class_weights = train_counts.sum() / np.maximum(1, train_counts)
        class_weights = class_weights / class_weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.training["learning_rate"]),
            weight_decay=float(config.training["weight_decay"]),
        )

        def loader(split: str, shuffle: bool) -> DataLoader:
            dataset = TensorDataset(
                torch.from_numpy(features[split]),
                torch.from_numpy(masks[split]),
                torch.from_numpy(labels[split]),
            )
            return DataLoader(
                dataset,
                batch_size=int(config.training["batch_size"]),
                shuffle=shuffle,
            )

        train_loader = loader("train", True)
        validation_loader = loader("validation", False)
        best_state: dict[str, object] | None = None
        best_loss = float("inf")
        patience = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, int(config.training["epochs"]) + 1):
            model.train()
            train_loss = 0.0
            for batch_features, batch_masks, batch_labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features.to(device), batch_masks.to(device))
                loss = criterion(logits, batch_labels.to(device))
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * len(batch_features)
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_masks, batch_labels in validation_loader:
                    logits = model(batch_features.to(device), batch_masks.to(device))
                    loss = criterion(logits, batch_labels.to(device))
                    validation_loss += float(loss) * len(batch_features)
            train_loss /= len(features["train"])
            validation_loss /= len(features["validation"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 8),
                    "validation_loss": round(validation_loss, 8),
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= int(config.training["early_stopping_patience"]):
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()

        def predict(split: str) -> np.ndarray:
            predictions: list[np.ndarray] = []
            with torch.no_grad():
                for batch_features, batch_masks, _batch_labels in loader(split, False):
                    logits = model(batch_features.to(device), batch_masks.to(device))
                    predictions.append(logits.argmax(dim=1).cpu().numpy())
            return np.concatenate(predictions)

        test_predictions = predict("test")
        metrics = _classification_metrics(labels["test"], test_predictions, config.labels)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = args.output_dir or ROOT / "runs/action_tcn" / stamp
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = output_dir / "reference-state-dict.pt"
        export_path = output_dir / "action-tcn.torchscript.pt"
        torch.save(best_state, checkpoint_path)
        example_features = torch.from_numpy(features["test"][:1]).to(device)
        example_mask = torch.from_numpy(masks["test"][:1]).to(device)
        traced = torch.jit.trace(model, (example_features, example_mask))
        traced.save(str(export_path))
        loaded = torch.jit.load(str(export_path), map_location=device)
        with torch.no_grad():
            reference = model(example_features, example_mask)
            exported = loaded(example_features, example_mask)
        max_difference = float(torch.max(torch.abs(reference - exported)).cpu())
        if max_difference > 1e-5:
            raise ValueError("reference/export consistency check failed")
        report = {
            "result": "DEVELOPMENT_TRAINING_COMPLETE_NOT_ADMITTED",
            "model_id": config.model_id,
            "config_sha256": _sha256(args.config),
            "view_manifest_sha256": _sha256(args.view_manifest),
            "device": str(device),
            "history": history,
            "metrics": metrics,
            "system_gate_metrics_pending": [
                "false_accepted_actions_per_hour_and_hand",
                "cross_seat_leakage",
                "cancellation_false_confirmation",
                "p95_confirmation_latency",
            ],
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "export": {
                "path": str(export_path),
                "sha256": _sha256(export_path),
                "max_abs_reference_difference": max_difference,
            },
        }
        (output_dir / "training-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        TrainingDependencyError,
    ) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
