"""Configuration and immutable asset checks for session face identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FaceModelAsset:
    model_id: str
    version: str
    asset_path: Path
    sha256: str
    framework: str
    framework_version: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.version.strip():
            raise ValueError("face model ID and version are required")
        if len(self.sha256) != 64:
            raise ValueError("face model SHA-256 must have 64 digits")
        int(self.sha256, 16)

    def verify(self) -> str:
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"face model asset is missing: {self.asset_path}")
        digest = hashlib.sha256()
        with self.asset_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != self.sha256:
            raise ValueError(
                f"face model SHA-256 mismatch: expected {self.sha256}, got {actual}"
            )
        return actual


@dataclass(frozen=True, slots=True)
class FaceIdentityConfig:
    schema_version: str
    pilot_status: str
    policy_version: str
    detector: FaceModelAsset
    embedder: FaceModelAsset
    detector_options: Mapping[str, int | float | tuple[int, int]]
    minimum_samples: int
    maximum_players: int
    require_exactly_one_face: bool
    explicit_consent_required: bool
    minimum_similarity: float
    minimum_margin: float
    minimum_stable_frames: int
    minimum_stable_duration_ms: int
    save_frames: bool
    persist_embeddings: bool
    clear_gallery_on_exit: bool
    acting_seat_source: str
    identity_role: str
    camera: Mapping[str, int | float | str]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported face identity schema version")
        if self.pilot_status != "development_feasibility_only":
            raise ValueError("face identity must remain development-only")
        if self.policy_version != "s0-21-session-face-v1":
            raise ValueError("unsupported face identity policy")
        if not 1 <= self.minimum_samples <= 20:
            raise ValueError("enrollment sample count is out of bounds")
        if self.maximum_players != 4:
            raise ValueError("Core face gallery must contain four players maximum")
        if not self.require_exactly_one_face or not self.explicit_consent_required:
            raise ValueError("enrollment requires one face and explicit consent")
        if not -1.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("cosine threshold must be in [-1, 1]")
        if not 0.0 <= self.minimum_margin <= 2.0:
            raise ValueError("cosine margin must be in [0, 2]")
        if self.minimum_stable_frames <= 0 or self.minimum_stable_duration_ms < 0:
            raise ValueError("identity temporal thresholds are invalid")
        if self.save_frames or self.persist_embeddings:
            raise ValueError("session face pilot must not persist biometric data")
        if not self.clear_gallery_on_exit:
            raise ValueError("session gallery must clear on exit")
        if self.acting_seat_source != "deterministic_game_state_machine":
            raise ValueError("face identity cannot select the acting seat")
        if self.identity_role != "verification_only":
            raise ValueError("face identity role must be verification-only")

    @classmethod
    def from_json(cls, path: str | Path) -> FaceIdentityConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parents[2]

        def asset(item: Mapping[str, object]) -> FaceModelAsset:
            return FaceModelAsset(
                model_id=str(item["model_id"]),
                version=str(item["version"]),
                asset_path=project_root / str(item["asset_path"]),
                sha256=str(item["sha256"]).lower(),
                framework=str(item["framework"]),
                framework_version=str(item["framework_version"]),
            )

        detector = value["detector"]
        enrollment = value["enrollment"]
        matching = value["matching"]
        privacy = value["privacy"]
        authority = value["authority"]
        return cls(
            schema_version=value["schema_version"],
            pilot_status=value["pilot_status"],
            policy_version=value["policy_version"],
            detector=asset(detector),
            embedder=asset(value["embedder"]),
            detector_options={
                "input_size": tuple(detector["input_size"]),
                "score_threshold": float(detector["score_threshold"]),
                "nms_threshold": float(detector["nms_threshold"]),
                "top_k": int(detector["top_k"]),
                "minimum_face_size_px": int(detector["minimum_face_size_px"]),
            },
            minimum_samples=int(enrollment["minimum_samples"]),
            maximum_players=int(enrollment["maximum_players"]),
            require_exactly_one_face=bool(enrollment["require_exactly_one_face"]),
            explicit_consent_required=bool(enrollment["explicit_consent_required"]),
            minimum_similarity=float(matching["pilot_minimum_similarity"]),
            minimum_margin=float(matching["pilot_minimum_margin"]),
            minimum_stable_frames=int(matching["minimum_stable_frames"]),
            minimum_stable_duration_ms=int(matching["minimum_stable_duration_ms"]),
            save_frames=bool(privacy["save_frames"]),
            persist_embeddings=bool(privacy["persist_embeddings"]),
            clear_gallery_on_exit=bool(privacy["clear_gallery_on_exit"]),
            acting_seat_source=str(authority["acting_seat_source"]),
            identity_role=str(authority["identity_role"]),
            camera=dict(value["camera"]),
        )

    def verify_assets(self) -> tuple[str, str]:
        return self.detector.verify(), self.embedder.verify()

    @property
    def model_version(self) -> str:
        return (
            f"{self.detector.model_id}@{self.detector.version}+"
            f"{self.embedder.model_id}@{self.embedder.version}"
        )
