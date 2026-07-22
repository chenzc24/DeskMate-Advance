"""Landmark-sequence preparation and optional compact PyTorch TCN."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class TrainingDependencyError(RuntimeError):
    """Raised when an optional offline training dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class ActionTcnConfig:
    schema_version: str
    status: str
    model_id: str
    version: str
    grammar_version: str
    feature_dim: int
    sequence_length: int
    window_stride: int
    labels: tuple[str, ...]
    channels: tuple[int, ...]
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    training: Mapping[str, int | float | str]
    export: Mapping[str, bool | str]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.status != "prepared_not_trained":
            raise ValueError("TCN config must remain prepared_not_trained")
        if self.feature_dim != 63:
            raise ValueError("the v1 input contract requires 21x3=63 features")
        if self.sequence_length <= 0 or self.window_stride <= 0:
            raise ValueError("sequence length and stride must be positive")
        if len(self.labels) != len(set(self.labels)) or self.labels[0] != "no_action":
            raise ValueError("labels must be unique with no_action at index zero")
        if len(self.channels) != len(self.dilations) or not self.channels:
            raise ValueError("each TCN block requires one channel and dilation")
        if self.kernel_size <= 1 or any(value <= 0 for value in self.dilations):
            raise ValueError("TCN kernel and dilations are invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.export.get("offline_only") is not True:
            raise ValueError("unadmitted TCN export must remain offline-only")

    @classmethod
    def from_json(cls, path: str | Path) -> ActionTcnConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        input_config = value["input"]
        architecture = value["architecture"]
        return cls(
            schema_version=value["schema_version"],
            status=value["status"],
            model_id=value["model_id"],
            version=value["version"],
            grammar_version=value["grammar_version"],
            feature_dim=int(input_config["feature_dim"]),
            sequence_length=int(input_config["sequence_length"]),
            window_stride=int(input_config["window_stride"]),
            labels=tuple(value["labels"]),
            channels=tuple(int(item) for item in architecture["channels"]),
            kernel_size=int(architecture["kernel_size"]),
            dilations=tuple(int(item) for item in architecture["dilations"]),
            dropout=float(architecture["dropout"]),
            training=dict(value["training"]),
            export=dict(value["export"]),
        )


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    values = np.asarray(landmarks, dtype=np.float32)
    if values.shape != (21, 3) or not np.isfinite(values).all():
        raise ValueError("hand landmarks must be a finite [21,3] array")
    centered = values - values[0]
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-6:
        raise ValueError("hand landmark scale is degenerate")
    normalized = centered / scale
    return np.ascontiguousarray(normalized.reshape(63), dtype=np.float32)


def make_sequence_windows(
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    mask_values = np.asarray(valid_mask, dtype=np.bool_)
    if feature_values.ndim != 2 or feature_values.shape[1] != 63:
        raise ValueError("features must have shape [frames,63]")
    if mask_values.shape != (feature_values.shape[0],):
        raise ValueError("valid_mask length must match frames")
    if sequence_length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")
    total_frames = feature_values.shape[0]
    if total_frames == 0:
        return (
            np.empty((0, sequence_length, 63), dtype=np.float32),
            np.empty((0, sequence_length), dtype=np.bool_),
            np.empty((0,), dtype=np.int64),
        )
    starts = list(range(0, max(1, total_frames - sequence_length + 1), stride))
    last_start = max(0, total_frames - sequence_length)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in starts:
        end = min(total_frames, start + sequence_length)
        window = np.zeros((sequence_length, 63), dtype=np.float32)
        mask = np.zeros((sequence_length,), dtype=np.bool_)
        copied = end - start
        window[:copied] = feature_values[start:end]
        mask[:copied] = mask_values[start:end]
        windows.append(window)
        masks.append(mask)
    return (
        np.stack(windows),
        np.stack(masks),
        np.asarray(starts, dtype=np.int64),
    )


def summarize_view_manifest(
    manifest: Mapping[str, Any], config: ActionTcnConfig
) -> dict[str, Any]:
    if manifest.get("schema_version") != "1.0" or manifest.get("status") != "derived":
        raise ValueError("view manifest must be schema 1.0 derived")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("view manifest requires records")
    participant_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = str(record.get("split"))
        label = str(record.get("label"))
        participant = str(record.get("participant_code"))
        session = str(record.get("session_id"))
        if split not in {"train", "validation", "test"}:
            raise ValueError("derived record has invalid split")
        if label not in config.labels:
            raise ValueError(f"derived record has unknown label {label}")
        if participant_splits.setdefault(participant, split) != split:
            raise ValueError(f"participant {participant} crosses splits")
        session_key = f"{participant}/{session}"
        if session_splits.setdefault(session_key, split) != split:
            raise ValueError(f"session {session_key} crosses splits")
        if int(record.get("frames", 0)) <= 0:
            raise ValueError("derived record frames must be positive")
        split_counts[split] += 1
        label_counts[split][label] += 1
    return {
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(label_counts.items())
        },
        "participants": len(participant_splits),
        "sessions": len(session_splits),
    }


def build_compact_tcn(config: ActionTcnConfig) -> object:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise TrainingDependencyError(
            "PyTorch is not installed; install the optional training dependencies"
        ) from exc

    class CompactActionTcn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[nn.Module] = []
            in_channels = config.feature_dim
            for out_channels, dilation in zip(
                config.channels, config.dilations, strict=True
            ):
                padding = dilation * (config.kernel_size - 1) // 2
                blocks.extend(
                    (
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            config.kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    )
                )
                in_channels = out_channels
            self.temporal = nn.Sequential(*blocks)
            self.classifier = nn.Linear(in_channels, len(config.labels))

        def forward(self, features: object, valid_mask: object) -> object:
            values = features.transpose(1, 2)
            encoded = self.temporal(values).transpose(1, 2)
            weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.classifier(pooled)

    torch.manual_seed(int(config.training["seed"]))
    return CompactActionTcn()
