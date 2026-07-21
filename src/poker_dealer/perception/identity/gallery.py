"""Consent-gated, memory-only player face enrollment and matching."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence

import numpy as np

from poker_dealer.domain import Seat

from .config import FaceIdentityConfig
from .domain import FaceIdentityState
from .opencv_adapter import DetectedFaceFeature, FaceFrameEvidence


PLAYER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


@dataclass(slots=True)
class _EnrollmentRecord:
    player_id: str
    seat: Seat
    sample_count: int
    template: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True)
class FaceMatchResult:
    state: FaceIdentityState
    player_id: str | None = None
    registered_seat: Seat | None = None
    similarity: float | None = None
    second_best_similarity: float | None = None
    quality_flags: tuple[str, ...] = ()


class SessionFaceGallery:
    """A non-serializable gallery whose templates are zeroed on clear."""

    def __init__(self, config: FaceIdentityConfig, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        self.config = config
        self.session_id = session_id
        self._records: dict[str, _EnrollmentRecord] = {}

    @property
    def size(self) -> int:
        return len(self._records)

    def enroll(
        self,
        player_id: str,
        seat: Seat,
        samples: Sequence[DetectedFaceFeature],
        *,
        consent_granted: bool,
    ) -> None:
        if self.config.explicit_consent_required and not consent_granted:
            raise PermissionError("explicit participant consent is required")
        if not PLAYER_ID_PATTERN.fullmatch(player_id):
            raise ValueError("player_id must be 1-32 safe ASCII characters")
        if player_id in self._records:
            raise ValueError("player_id is already enrolled")
        if any(record.seat is seat for record in self._records.values()):
            raise ValueError("seat already has an enrolled player")
        if self.size >= self.config.maximum_players:
            raise ValueError("session gallery is full")
        if len(samples) < self.config.minimum_samples:
            raise ValueError("insufficient enrollment samples")
        vectors = np.stack([sample.embedding for sample in samples]).astype(
            np.float32, copy=False
        )
        template = vectors.mean(axis=0)
        norm = float(np.linalg.norm(template))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("invalid enrollment template")
        template = np.asarray(template / norm, dtype=np.float32)
        for record in self._records.values():
            if float(np.dot(template, record.template)) >= self.config.minimum_similarity:
                template.fill(0.0)
                raise ValueError("face appears already enrolled under another player")
        self._records[player_id] = _EnrollmentRecord(
            player_id, seat, len(samples), template
        )

    def match_frame(self, frame: FaceFrameEvidence) -> FaceMatchResult:
        if frame.detected_face_count == 0:
            return FaceMatchResult(FaceIdentityState.NO_FACE)
        if frame.detected_face_count > 1:
            return FaceMatchResult(
                FaceIdentityState.MULTIPLE_FACES,
                quality_flags=("exactly_one_face_required",),
            )
        if not frame.features:
            return FaceMatchResult(
                FaceIdentityState.LOW_QUALITY,
                quality_flags=("face_too_small_or_embedding_failed",),
            )
        if not self._records:
            return FaceMatchResult(FaceIdentityState.ENROLLMENT_REQUIRED)
        feature = frame.features[0].embedding
        ranked = sorted(
            (
                (float(np.dot(feature, record.template)), record)
                for record in self._records.values()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        if best_score < self.config.minimum_similarity:
            return FaceMatchResult(
                FaceIdentityState.UNKNOWN,
                similarity=best_score,
                second_best_similarity=second_score,
                quality_flags=("below_identity_similarity_threshold",),
            )
        if (
            second_score is not None
            and best_score - second_score < self.config.minimum_margin
        ):
            return FaceMatchResult(
                FaceIdentityState.AMBIGUOUS,
                similarity=best_score,
                second_best_similarity=second_score,
                quality_flags=("identity_margin_too_small",),
            )
        return FaceMatchResult(
            FaceIdentityState.MATCHED,
            player_id=best.player_id,
            registered_seat=best.seat,
            similarity=best_score,
            second_best_similarity=second_score,
        )

    def match_expected_seat(
        self, frame: FaceFrameEvidence, expected_seat: Seat
    ) -> FaceMatchResult:
        """Refuse identity matching when the state-owned seat is unregistered."""

        if not any(record.seat is expected_seat for record in self._records.values()):
            return FaceMatchResult(
                FaceIdentityState.EXPECTED_SEAT_UNENROLLED,
                quality_flags=("state_owned_focus_seat_has_no_session_registration",),
            )
        return self.match_frame(frame)

    def metadata(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "player_id": record.player_id,
                "seat": record.seat.value,
                "sample_count": record.sample_count,
            }
            for record in self._records.values()
        )

    def clear(self) -> None:
        for record in self._records.values():
            record.template.fill(0.0)
        self._records.clear()

    def __enter__(self) -> SessionFaceGallery:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.clear()
