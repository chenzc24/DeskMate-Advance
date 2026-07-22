"""Configuration for development actor attribution on the rotating camera."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class ActorAttributionConfig:
    pose_model_id: str
    pose_model_version: str
    pose_asset_path: Path
    pose_asset_sha256: str
    num_poses: int
    minimum_pose_confidence: float
    maximum_face_pose_distance: float
    maximum_track_jump: float
    maximum_hand_wrist_distance: float
    minimum_assignment_margin: float
    actor_lease_ms: int
    max_hands: int

    def __post_init__(self) -> None:
        if not self.pose_model_id.strip() or not self.pose_model_version.strip():
            raise ValueError("pose model identity is required")
        if not 1 <= self.num_poses <= 4 or not 1 <= self.max_hands <= 4:
            raise ValueError("actor attribution supports one to four poses/hands")
        for value in (
            self.minimum_pose_confidence,
            self.maximum_face_pose_distance,
            self.maximum_track_jump,
            self.maximum_hand_wrist_distance,
            self.minimum_assignment_margin,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("actor attribution thresholds must be in [0, 1]")
        if self.actor_lease_ms <= 0:
            raise ValueError("actor lease must be positive")

    @classmethod
    def from_json(cls, path: Path) -> "ActorAttributionConfig":
        value = json.loads(path.read_text(encoding="utf-8"))
        pose = value["pose_model"]
        policy = value["policy"]
        asset_path = Path(pose["asset_path"])
        if not asset_path.is_absolute():
            asset_path = ROOT / asset_path
        return cls(
            pose_model_id=str(pose["model_id"]),
            pose_model_version=str(pose["version"]),
            pose_asset_path=asset_path,
            pose_asset_sha256=str(pose["sha256"]),
            num_poses=int(pose["num_poses"]),
            minimum_pose_confidence=float(pose["minimum_confidence"]),
            maximum_face_pose_distance=float(policy["maximum_face_pose_distance"]),
            maximum_track_jump=float(policy["maximum_track_jump"]),
            maximum_hand_wrist_distance=float(policy["maximum_hand_wrist_distance"]),
            minimum_assignment_margin=float(policy["minimum_assignment_margin"]),
            actor_lease_ms=int(policy["actor_lease_ms"]),
            max_hands=int(policy["max_hands"]),
        )

    def verify_pose_asset(self) -> None:
        if not self.pose_asset_path.is_file():
            raise FileNotFoundError(f"pose model asset is missing: {self.pose_asset_path}")
        digest = hashlib.sha256(self.pose_asset_path.read_bytes()).hexdigest()
        if digest != self.pose_asset_sha256:
            raise ValueError(
                f"pose model SHA-256 mismatch: expected {self.pose_asset_sha256}, got {digest}"
            )
