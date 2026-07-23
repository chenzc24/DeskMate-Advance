"""Per-chip temporal confirmation for live denomination observations."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import MutableMapping, Sequence


Detection = MutableMapping[str, object]


def bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    last_seen_frame: int
    history: deque[int] = field(default_factory=lambda: deque(maxlen=7))
    confirmed: int | None = None
    last_evidence_frame: int | None = None
    challenger: int | None = None
    challenger_streak: int = 0
    quality_failures: int = 0
    last_quality_reason: str | None = None


class ChipValueTracker:
    """Associate boxes and expose only temporally confirmed denominations."""

    def __init__(
        self,
        *,
        association_iou: float = 0.30,
        max_missed_frames: int = 15,
        vote_window: int = 7,
        required_votes: int = 5,
        switch_consecutive: int = 3,
        switch_minimum_score: float = 0.70,
        quality_failure_limit: int = 2,
    ) -> None:
        if not 0.0 < association_iou <= 1.0:
            raise ValueError("association_iou must be in (0, 1]")
        if max_missed_frames <= 0:
            raise ValueError("max_missed_frames must be positive")
        if not 1 <= required_votes <= vote_window:
            raise ValueError("required_votes must be in [1, vote_window]")
        if switch_consecutive <= 0:
            raise ValueError("switch_consecutive must be positive")
        self.association_iou = association_iou
        self.max_missed_frames = max_missed_frames
        self.vote_window = vote_window
        self.required_votes = required_votes
        self.switch_consecutive = switch_consecutive
        self.switch_minimum_score = switch_minimum_score
        self.quality_failure_limit = quality_failure_limit
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def _new_track(self, bbox: tuple[int, int, int, int], frame: int) -> _Track:
        track = _Track(self._next_track_id, bbox, frame)
        track.history = deque(maxlen=self.vote_window)
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _ingest_evidence(self, track: _Track, detection: Detection) -> None:
        source_frame = detection.get("value_source_frame")
        if not isinstance(source_frame, int) or source_frame == track.last_evidence_frame:
            return
        track.last_evidence_frame = source_frame
        rejection = detection.get("value_rejection_reason")
        if rejection in {"too_far", "too_flat"}:
            track.quality_failures += 1
            track.last_quality_reason = str(rejection)
            return

        denomination = detection.get("denomination")
        if not isinstance(denomination, int):
            return
        track.quality_failures = 0
        track.last_quality_reason = None
        track.history.append(denomination)
        if track.confirmed is None:
            counts = Counter(track.history)
            candidate, votes = counts.most_common(1)[0]
            if votes >= self.required_votes:
                track.confirmed = candidate
            return

        if denomination == track.confirmed:
            track.challenger = None
            track.challenger_streak = 0
            return
        score = detection.get("value_score")
        if not isinstance(score, (int, float)) or score < self.switch_minimum_score:
            track.challenger = None
            track.challenger_streak = 0
            return
        if track.challenger == denomination:
            track.challenger_streak += 1
        else:
            track.challenger = denomination
            track.challenger_streak = 1
        if track.challenger_streak >= self.switch_consecutive:
            track.confirmed = denomination
            track.history.clear()
            track.history.append(denomination)
            track.challenger = None
            track.challenger_streak = 0

    def _annotate(self, track: _Track, detection: Detection) -> None:
        counts = Counter(track.history)
        votes = max(counts.values(), default=0)
        quality_blocked = track.quality_failures >= self.quality_failure_limit
        stable = None if quality_blocked else track.confirmed
        if quality_blocked:
            state = track.last_quality_reason or "quality_rejected"
        elif stable is not None:
            state = "confirmed"
        else:
            state = f"collecting_{votes}_of_{self.required_votes}"
        detection.update(
            {
                "track_id": track.track_id,
                "stable_denomination": stable,
                "value_state": state,
                "value_votes": votes,
                "value_vote_window": len(track.history),
            }
        )

    def associate(self, frame: int, detections: list[Detection]) -> None:
        """Assign persistent IDs before expensive denomination processing."""

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame - track.last_seen_frame > self.max_missed_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = bbox_iou(track.bbox_xyxy, detection["bbox_xyxy"])
                if overlap >= self.association_iou:
                    candidates.append((overlap, track_id, detection_index))
        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, _Track] = {}
        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            bbox = tuple(int(value) for value in detection["bbox_xyxy"])
            track = assignments.get(detection_index)
            if track is None:
                track = self._new_track(bbox, frame)
            track.bbox_xyxy = bbox
            track.last_seen_frame = frame
            self._annotate(track, detection)

    def ingest(self, detections: list[Detection]) -> None:
        """Consume already-attached observations without reassociating boxes."""

        for detection in detections:
            track_id = detection.get("track_id")
            if not isinstance(track_id, int):
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            self._ingest_evidence(track, detection)
            self._annotate(track, detection)

    def update(self, frame: int, detections: list[Detection]) -> None:
        """Backward-compatible one-shot association and evidence ingestion."""

        self.associate(frame, detections)
        self.ingest(detections)
