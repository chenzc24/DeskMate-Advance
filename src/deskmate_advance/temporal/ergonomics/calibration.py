"""Bounded, privacy-preserving neutral calibration for Part A ergonomics.

Only scalar features are retained. Camera frames, landmarks, observations and
other identity-bearing records are never stored by the collector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
import math
from statistics import median

from deskmate_advance.features.ergonomics.live import LiveSnapshot
from deskmate_advance.perception.ergonomics import ObservationState


class CalibrationState(StrEnum):
    """Lifecycle state for one fixed-duration calibration window."""

    COLLECTING = "collecting"
    READY = "ready"
    NOT_READY = "not_ready"


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Requirements for one neutral calibration window."""

    duration_ms: int = 5_000
    minimum_pose_samples: int = 20
    minimum_face_samples: int = 20
    minimum_luminance_samples: int = 5
    maximum_snapshot_gap_ms: int = 500
    maximum_samples_per_metric: int = 10_000

    def __post_init__(self) -> None:
        for name, value in (
            ("duration_ms", self.duration_ms),
            ("minimum_pose_samples", self.minimum_pose_samples),
            ("minimum_face_samples", self.minimum_face_samples),
            ("minimum_luminance_samples", self.minimum_luminance_samples),
            ("maximum_snapshot_gap_ms", self.maximum_snapshot_gap_ms),
            ("maximum_samples_per_metric", self.maximum_samples_per_metric),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        for name, value in (
            ("minimum_pose_samples", self.minimum_pose_samples),
            ("minimum_face_samples", self.minimum_face_samples),
            ("minimum_luminance_samples", self.minimum_luminance_samples),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_snapshot_gap_ms <= 0:
            raise ValueError("maximum_snapshot_gap_ms must be positive")
        if self.maximum_samples_per_metric < max(
            self.minimum_pose_samples,
            self.minimum_face_samples,
            self.minimum_luminance_samples,
        ):
            raise ValueError(
                "maximum_samples_per_metric must cover every minimum sample count"
            )


@dataclass(frozen=True, slots=True)
class CalibrationSampleCounts:
    """Accepted scalar sample counts; optional metrics are shown separately."""

    pose: int
    torso_lean: int
    face: int
    eye_open: int
    luminance: int


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    """Immutable UI/status view of the current calibration window."""

    state: CalibrationState
    elapsed_ms: float
    duration_ms: int
    duration_fraction: float
    counts: CalibrationSampleCounts
    ready: bool
    not_ready_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Robust neutral baselines containing scalar, non-identity evidence only."""

    source_id: str
    device_index: int
    window_started_at_ns: int
    window_ended_at_ns: int
    pose_samples: int
    torso_lean_samples: int
    face_samples: int
    eye_open_samples: int
    luminance_samples: int
    shoulder_tilt_deg: float
    torso_lean_from_vertical_deg: float | None
    face_bbox_area_ratio: float
    head_rotation_x_deg: float
    head_rotation_y_deg: float
    eye_open_score: float | None
    mean_luminance: float
    p90_luminance: float


class CalibrationCollector:
    """Collect a single bounded neutral profile from fresh live observations.

    ``update`` must receive strictly increasing camera timestamps from one
    camera. A completed window is immutable; call ``reset`` before starting a
    new calibration.
    """

    _SHOULDER = "shoulder_tilt_deg"
    _TORSO = "torso_lean_from_vertical_deg"
    _FACE_AREA = "face_bbox_area_ratio"
    _HEAD_X = "head_rotation_x_deg"
    _HEAD_Y = "head_rotation_y_deg"
    _EYE_OPEN = "eye_open_score"
    _LUMINANCE_MEAN = "mean_luminance"
    _LUMINANCE_P90 = "p90_luminance"

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._values: dict[str, deque[float]] = {}
        self.reset()

    @property
    def profile(self) -> CalibrationProfile | None:
        """The frozen profile, or ``None`` until calibration succeeds."""

        return self._profile

    @property
    def progress(self) -> CalibrationProgress:
        """Current immutable progress without retaining the latest snapshot."""

        return self._progress()

    def reset(self) -> None:
        """Discard scalar samples and begin a new window on the next update."""

        limit = self.config.maximum_samples_per_metric
        self._values = {
            name: deque(maxlen=limit)
            for name in (
                self._SHOULDER,
                self._TORSO,
                self._FACE_AREA,
                self._HEAD_X,
                self._HEAD_Y,
                self._EYE_OPEN,
                self._LUMINANCE_MEAN,
                self._LUMINANCE_P90,
            )
        }
        self._source_id: str | None = None
        self._device_index: int | None = None
        self._started_at_ns: int | None = None
        self._last_snapshot_at_ns: int | None = None
        self._last_pose_at_ns: int | None = None
        self._last_face_at_ns: int | None = None
        self._last_luminance_at_ns: int | None = None
        self._closed_at_ns: int | None = None
        self._terminal_reasons: tuple[str, ...] = ()
        self._profile: CalibrationProfile | None = None

    def update(self, snapshot: LiveSnapshot) -> CalibrationProgress:
        """Validate and consume fresh scalar evidence from one live snapshot."""

        previous_at_ns = self._last_snapshot_at_ns
        self._validate_snapshot_identity(snapshot)
        timestamp_ns = snapshot.frame.captured_at_ns
        if self._started_at_ns is None:
            self._started_at_ns = timestamp_ns
        self._last_snapshot_at_ns = timestamp_ns

        if self._closed_at_ns is not None:
            return self._progress()

        gap_ms = (
            (timestamp_ns - previous_at_ns) / 1_000_000
            if previous_at_ns is not None
            else 0.0
        )
        if gap_ms > self.config.maximum_snapshot_gap_ms:
            return self._close_for_gap(
                timestamp_ns,
                f"snapshot_gap_ms:{gap_ms:.3f}",
            )
        if snapshot.frame.dropped_before > 0:
            return self._close_for_gap(
                timestamp_ns,
                f"camera_frames_dropped:{snapshot.frame.dropped_before}",
            )

        window_end_ns = self._started_at_ns + self.config.duration_ms * 1_000_000
        if timestamp_ns <= window_end_ns:
            self._collect_pose(snapshot)
            self._collect_face(snapshot)
            self._collect_luminance(snapshot)
        if timestamp_ns >= window_end_ns:
            self._closed_at_ns = timestamp_ns
            if not self._sample_reasons(include_duration=False):
                self._profile = self._build_profile(window_end_ns)
        return self._progress()

    def mark_evidence_gap(
        self,
        timestamp_ns: int,
        *,
        reason: str,
    ) -> CalibrationProgress:
        """Freeze an in-progress window when camera evidence is unavailable."""

        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must not be empty")
        if self._started_at_ns is None or self._closed_at_ns is not None:
            return self._progress()
        if (
            self._last_snapshot_at_ns is not None
            and timestamp_ns <= self._last_snapshot_at_ns
        ):
            raise ValueError("calibration gap timestamps must increase strictly")
        self._last_snapshot_at_ns = timestamp_ns
        return self._close_for_gap(timestamp_ns, reason)

    def _close_for_gap(
        self,
        timestamp_ns: int,
        reason: str,
    ) -> CalibrationProgress:
        self._closed_at_ns = timestamp_ns
        self._terminal_reasons = (f"evidence_discontinuity:{reason}",)
        return self._progress()

    def _validate_snapshot_identity(self, snapshot: LiveSnapshot) -> None:
        frame = snapshot.frame
        timestamp_ns = frame.captured_at_ns
        if (
            self._last_snapshot_at_ns is not None
            and timestamp_ns <= self._last_snapshot_at_ns
        ):
            raise ValueError("calibration camera timestamps must increase strictly")
        if self._source_id is not None and (
            frame.source_id != self._source_id
            or frame.device_index != self._device_index
        ):
            raise ValueError("calibration snapshots must come from one source and camera")

        records = (
            ("luminance", snapshot.luminance.source_id, snapshot.luminance.captured_at_ns),
            (
                "pose_observation",
                snapshot.pose_observation.context.source_id,
                snapshot.pose_observation.context.captured_at_ns,
            )
            if snapshot.pose_observation is not None
            else None,
            (
                "pose_features",
                snapshot.pose_features.source_id,
                snapshot.pose_features.captured_at_ns,
            )
            if snapshot.pose_features is not None
            else None,
            (
                "face_observation",
                snapshot.face_observation.context.source_id,
                snapshot.face_observation.context.captured_at_ns,
            )
            if snapshot.face_observation is not None
            else None,
            (
                "face_features",
                snapshot.face_features.source_id,
                snapshot.face_features.captured_at_ns,
            )
            if snapshot.face_features is not None
            else None,
        )
        for record in records:
            if record is None:
                continue
            label, source_id, captured_at_ns = record
            if source_id != frame.source_id:
                raise ValueError(f"{label} source does not match camera frame")
            if captured_at_ns > timestamp_ns:
                raise ValueError(f"{label} timestamp is later than camera frame")

        fresh_records = (
            (
                "pose",
                snapshot.pose_ran,
                snapshot.pose_features.captured_at_ns
                if snapshot.pose_features is not None
                else None,
            ),
            (
                "face",
                snapshot.face_ran,
                snapshot.face_features.captured_at_ns
                if snapshot.face_features is not None
                else None,
            ),
            ("luminance", snapshot.luminance_ran, snapshot.luminance.captured_at_ns),
        )
        for label, ran, captured_at_ns in fresh_records:
            if ran and captured_at_ns is not None and captured_at_ns != timestamp_ns:
                raise ValueError(f"fresh {label} timestamp must match camera frame")

        # Establish identity only after the whole snapshot has passed validation,
        # so rejected input cannot partially initialize the collector.
        if self._source_id is None:
            self._source_id = frame.source_id
            self._device_index = frame.device_index

    def _collect_pose(self, snapshot: LiveSnapshot) -> None:
        features = snapshot.pose_features
        if (
            not snapshot.pose_ran
            or snapshot.pose_stale
            or features is None
            or features.state is not ObservationState.VALID
        ):
            return
        self._require_new_modality_timestamp(
            "pose", features.captured_at_ns, snapshot.frame.captured_at_ns
        )
        shoulder = self._finite(features.shoulder_tilt_deg)
        torso = self._finite(features.torso_lean_from_vertical_deg)
        if shoulder is not None:
            self._values[self._SHOULDER].append(shoulder)
        if torso is not None:
            self._values[self._TORSO].append(torso)

    def _collect_face(self, snapshot: LiveSnapshot) -> None:
        features = snapshot.face_features
        if (
            not snapshot.face_ran
            or snapshot.face_stale
            or features is None
            or features.state is not ObservationState.VALID
        ):
            return
        self._require_new_modality_timestamp(
            "face", features.captured_at_ns, snapshot.frame.captured_at_ns
        )
        area = (
            self._finite(features.face_bbox_area_ratio)
            if features.geometry_state is ObservationState.VALID
            else None
        )
        rotation = (
            features.raw_rotation_xyz_deg
            if features.rotation_state is ObservationState.VALID
            else None
        )
        rotation_x = self._finite(rotation[0]) if rotation is not None else None
        rotation_y = self._finite(rotation[1]) if rotation is not None else None
        if area is not None and area > 0 and rotation_x is not None and rotation_y is not None:
            self._values[self._FACE_AREA].append(area)
            self._values[self._HEAD_X].append(rotation_x)
            self._values[self._HEAD_Y].append(rotation_y)

        blink = (
            self._finite(features.eye_blink_mean)
            if features.blink_state is ObservationState.VALID
            else None
        )
        if blink is not None and 0 <= blink <= 1:
            # MediaPipe's score is eye-closure evidence; expose its inverse so
            # the neutral profile has an explicitly named eye-open score.
            self._values[self._EYE_OPEN].append(1.0 - blink)

    def _collect_luminance(self, snapshot: LiveSnapshot) -> None:
        observation = snapshot.luminance
        if not snapshot.luminance_ran or observation.state is not ObservationState.VALID:
            return
        self._require_new_modality_timestamp(
            "luminance", observation.captured_at_ns, snapshot.frame.captured_at_ns
        )
        mean = self._finite(observation.mean)
        p90 = self._finite(observation.p90)
        if mean is not None and p90 is not None:
            self._values[self._LUMINANCE_MEAN].append(mean)
            self._values[self._LUMINANCE_P90].append(p90)

    def _require_new_modality_timestamp(
        self, modality: str, timestamp_ns: int, frame_timestamp_ns: int
    ) -> None:
        if timestamp_ns != frame_timestamp_ns:
            raise ValueError(f"fresh {modality} timestamp must match camera frame")
        attribute = f"_last_{modality}_at_ns"
        previous = getattr(self, attribute)
        if previous is not None and timestamp_ns <= previous:
            raise ValueError(f"fresh {modality} timestamps must increase strictly")
        setattr(self, attribute, timestamp_ns)

    def _progress(self) -> CalibrationProgress:
        counts = self._counts()
        if self._started_at_ns is None:
            elapsed_ms = 0.0
        elif self._closed_at_ns is not None:
            elapsed_ms = min(
                float(self.config.duration_ms),
                (self._closed_at_ns - self._started_at_ns) / 1_000_000,
            )
        elif self._last_snapshot_at_ns is None:
            elapsed_ms = 0.0
        else:
            elapsed_ms = min(
                float(self.config.duration_ms),
                (self._last_snapshot_at_ns - self._started_at_ns) / 1_000_000,
            )
        nominal_end_ns = (
            self._started_at_ns + self.config.duration_ms * 1_000_000
            if self._started_at_ns is not None
            else None
        )
        reasons = self._sample_reasons(
            include_duration=(
                self._closed_at_ns is None
                or (
                    nominal_end_ns is not None
                    and self._closed_at_ns < nominal_end_ns
                )
            )
        )
        if self._profile is not None:
            state = CalibrationState.READY
        elif self._closed_at_ns is not None:
            state = CalibrationState.NOT_READY
        else:
            state = CalibrationState.COLLECTING
        return CalibrationProgress(
            state=state,
            elapsed_ms=elapsed_ms,
            duration_ms=self.config.duration_ms,
            duration_fraction=min(1.0, elapsed_ms / self.config.duration_ms),
            counts=counts,
            ready=self._profile is not None,
            not_ready_reasons=() if self._profile is not None else reasons,
        )

    def _sample_reasons(self, *, include_duration: bool) -> tuple[str, ...]:
        counts = self._counts()
        reasons: list[str] = []
        reasons.extend(self._terminal_reasons)
        if include_duration:
            reasons.append("calibration_duration_incomplete")
        if counts.pose < self.config.minimum_pose_samples:
            reasons.append(
                f"pose_samples:{counts.pose}/{self.config.minimum_pose_samples}"
            )
        if counts.face < self.config.minimum_face_samples:
            reasons.append(
                f"face_samples:{counts.face}/{self.config.minimum_face_samples}"
            )
        if counts.luminance < self.config.minimum_luminance_samples:
            reasons.append(
                "luminance_samples:"
                f"{counts.luminance}/{self.config.minimum_luminance_samples}"
            )
        return tuple(reasons)

    def _counts(self) -> CalibrationSampleCounts:
        return CalibrationSampleCounts(
            pose=len(self._values[self._SHOULDER]),
            torso_lean=len(self._values[self._TORSO]),
            face=min(
                len(self._values[self._FACE_AREA]),
                len(self._values[self._HEAD_X]),
                len(self._values[self._HEAD_Y]),
            ),
            eye_open=len(self._values[self._EYE_OPEN]),
            luminance=min(
                len(self._values[self._LUMINANCE_MEAN]),
                len(self._values[self._LUMINANCE_P90]),
            ),
        )

    def _build_profile(self, window_end_ns: int) -> CalibrationProfile:
        if self._source_id is None or self._device_index is None or self._started_at_ns is None:
            raise RuntimeError("calibration identity was not initialized")
        counts = self._counts()
        torso_values = self._values[self._TORSO]
        eye_open_values = self._values[self._EYE_OPEN]
        return CalibrationProfile(
            source_id=self._source_id,
            device_index=self._device_index,
            window_started_at_ns=self._started_at_ns,
            window_ended_at_ns=window_end_ns,
            pose_samples=counts.pose,
            torso_lean_samples=counts.torso_lean,
            face_samples=counts.face,
            eye_open_samples=counts.eye_open,
            luminance_samples=counts.luminance,
            shoulder_tilt_deg=float(median(self._values[self._SHOULDER])),
            torso_lean_from_vertical_deg=(
                float(median(torso_values)) if torso_values else None
            ),
            face_bbox_area_ratio=float(median(self._values[self._FACE_AREA])),
            head_rotation_x_deg=float(median(self._values[self._HEAD_X])),
            head_rotation_y_deg=float(median(self._values[self._HEAD_Y])),
            eye_open_score=(
                float(median(eye_open_values)) if eye_open_values else None
            ),
            mean_luminance=float(median(self._values[self._LUMINANCE_MEAN])),
            p90_luminance=float(median(self._values[self._LUMINANCE_P90])),
        )

    @staticmethod
    def _finite(value: float | None) -> float | None:
        if value is None:
            return None
        output = float(value)
        return output if math.isfinite(output) else None
