"""Privacy-safe in-memory gallery for externally produced speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Sequence

import numpy as np


class SpeakerVerificationState(StrEnum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    ENROLLMENT_REQUIRED = "enrollment_required"
    INSUFFICIENT_AUDIO = "insufficient_audio"


@dataclass(frozen=True, slots=True)
class SpeakerVerificationResult:
    state: SpeakerVerificationState
    player_id: str | None = None
    similarity: float | None = None
    second_best_similarity: float | None = None
    quality_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class _SpeakerRecord:
    player_id: str
    template: np.ndarray


class SessionSpeakerGallery:
    """Store only session templates; callers own audio/model inference."""

    def __init__(
        self,
        session_id: str,
        *,
        minimum_samples: int = 3,
        minimum_speaker_frames: int = 1,
        minimum_similarity: float = 0.70,
        minimum_margin: float = 0.08,
    ) -> None:
        if not session_id.strip():
            raise ValueError("speaker session ID is required")
        if minimum_samples <= 0:
            raise ValueError("speaker minimum samples must be positive")
        if minimum_speaker_frames <= 0:
            raise ValueError("minimum speaker frames must be positive")
        self.session_id = session_id
        self.minimum_samples = minimum_samples
        self.minimum_speaker_frames = minimum_speaker_frames
        self.minimum_similarity = minimum_similarity
        self.minimum_margin = minimum_margin
        self._records: dict[str, _SpeakerRecord] = {}

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1).copy()
        norm = float(np.linalg.norm(value))
        if value.size == 0 or not np.isfinite(norm) or norm <= 0:
            raise ValueError("speaker embedding must be finite and non-zero")
        value /= norm
        return value

    def enroll(self, player_id: str, samples: Sequence[np.ndarray]) -> None:
        if not player_id.strip() or player_id in self._records:
            raise ValueError("speaker player ID is missing or already enrolled")
        if len(samples) < self.minimum_samples:
            raise ValueError("insufficient speaker enrollment samples")
        normalized = [self._normalized(sample) for sample in samples]
        template = self._normalized(np.mean(np.stack(normalized), axis=0))
        for sample in normalized:
            sample.fill(0.0)
        self._records[player_id] = _SpeakerRecord(player_id, template)

    def match(
        self, embedding: np.ndarray, *, speaker_frames: int
    ) -> SpeakerVerificationResult:
        if speaker_frames < self.minimum_speaker_frames:
            return SpeakerVerificationResult(
                SpeakerVerificationState.INSUFFICIENT_AUDIO,
                quality_flags=("speaker_audio_too_short",),
            )
        if not self._records:
            return SpeakerVerificationResult(
                SpeakerVerificationState.ENROLLMENT_REQUIRED
            )
        query = self._normalized(embedding)
        ranked = sorted(
            (
                (
                    max(-1.0, min(1.0, float(np.dot(query, record.template)))),
                    record,
                )
                for record in self._records.values()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        query.fill(0.0)
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        if best_score < self.minimum_similarity:
            return SpeakerVerificationResult(
                SpeakerVerificationState.UNKNOWN,
                similarity=best_score,
                second_best_similarity=second_score,
                quality_flags=("speaker_below_similarity_threshold",),
            )
        if second_score is not None and best_score - second_score < self.minimum_margin:
            return SpeakerVerificationResult(
                SpeakerVerificationState.AMBIGUOUS,
                similarity=best_score,
                second_best_similarity=second_score,
                quality_flags=("speaker_similarity_margin_too_small",),
            )
        return SpeakerVerificationResult(
            SpeakerVerificationState.MATCHED,
            player_id=best.player_id,
            similarity=best_score,
            second_best_similarity=second_score,
            quality_flags=("session_speaker_verified",),
        )

    def clear(self) -> None:
        for record in self._records.values():
            record.template.fill(0.0)
        self._records.clear()

    @property
    def size(self) -> int:
        return len(self._records)

    def is_enrolled(self, player_id: str) -> bool:
        return player_id in self._records

    def __enter__(self) -> "SessionSpeakerGallery":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.clear()
