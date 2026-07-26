"""Transport-neutral robot navigation and chip-observation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .dealer import DealerTargetSlot
from .game import Seat


class RobotPoseNode(StrEnum):
    """Finite, auditable poses from the table-route plan."""

    UNKNOWN = "unknown"
    INIT_TO_END = "i_e"
    INIT_TO_INIT = "i_w"
    INIT_BUTTON = "i_button"
    INIT_UTG = "i_utg"
    BOARD_TO_END = "b_e"
    END_TO_END = "e_e"
    END_TO_INIT = "e_w"
    END_SMALL_BLIND = "e_sb"
    END_BIG_BLIND = "e_bb"


class NavigationAction(StrEnum):
    """Semantic motion only; adapters own wheel speeds and motor details."""

    MOVE_AND_ALIGN_TO_TARGET = "move_and_align_to_target"
    FOLLOW_LINE_TO_INIT = "follow_line_to_init"
    FOLLOW_LINE_TO_END = "follow_line_to_end"
    FOLLOW_LINE_TO_BOARD = "follow_line_to_board"
    FOLLOW_LINE_BOARD_TO_END = "follow_line_board_to_end"
    TURN_LEFT_TO_PLAYER = "turn_left_to_player"
    TURN_RIGHT_TO_PLAYER = "turn_right_to_player"
    RETURN_TO_LINE = "return_to_line"
    STOP = "stop"


class NavigationAckStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class NavigationErrorCode(StrEnum):
    LINE_LOST = "line_lost"
    ENDPOINT_NOT_FOUND = "endpoint_not_found"
    BOARD_MARKER_NOT_FOUND = "board_marker_not_found"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_MISMATCH = "target_mismatch"
    ALIGNMENT_TIMEOUT = "alignment_timeout"
    POSE_UNKNOWN = "pose_unknown"
    INTERLOCK_OPEN = "interlock_open"
    EMERGENCY_STOP = "emergency_stop"
    TRANSPORT_LOST = "transport_lost"
    PROTOCOL_ERROR = "protocol_error"


@dataclass(frozen=True, slots=True)
class NavigationCommand:
    """One versioned semantic navigation request."""

    command_id: str
    session_id: str
    hand_id: str
    expected_state_version: int
    expected_pose_version: int
    issued_at_ns: int
    action: NavigationAction
    start_pose: RobotPoseNode
    target_pose: RobotPoseNode
    target_slot: DealerTargetSlot | None = None
    timeout_ms: int = 5000
    inter_motion_delay_ms: int = 2500

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("navigation command_id is required")
        if not self.session_id.strip() or not self.hand_id.strip():
            raise ValueError("navigation session_id and hand_id are required")
        if (
            self.expected_state_version < 0
            or self.expected_pose_version < 0
            or self.issued_at_ns < 0
        ):
            raise ValueError("navigation versions and timestamp must be non-negative")
        if self.timeout_ms <= 0:
            raise ValueError("navigation timeout_ms must be positive")
        if self.inter_motion_delay_ms < 0:
            raise ValueError("navigation inter_motion_delay_ms must be non-negative")
        if self.start_pose is RobotPoseNode.UNKNOWN:
            raise ValueError("navigation cannot start from an unknown pose")
        if (
            self.target_pose is RobotPoseNode.UNKNOWN
            and self.action is not NavigationAction.MOVE_AND_ALIGN_TO_TARGET
        ):
            raise ValueError("only target navigation may resolve its pose in the adapter")
        if (
            self.action is NavigationAction.MOVE_AND_ALIGN_TO_TARGET
            and self.target_slot is None
        ):
            raise ValueError("move_and_align_to_target requires target_slot")


@dataclass(frozen=True, slots=True)
class NavigationAck:
    """Pose-bearing navigation evidence returned by the mobile robot."""

    command_id: str
    session_id: str
    hand_id: str
    expected_state_version: int
    action: NavigationAction
    target_slot: DealerTargetSlot | None
    status: NavigationAckStatus
    observed_at_ns: int
    actual_pose: RobotPoseNode
    pose_version: int
    pose_confidence: float
    line_locked: bool
    endpoint_confirmed: bool
    target_aligned: bool
    stable_frames: int
    face_center_error_px: float | None = None
    error_code: NavigationErrorCode | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("navigation ACK command_id is required")
        if not self.session_id.strip() or not self.hand_id.strip():
            raise ValueError("navigation ACK session_id and hand_id are required")
        if (
            self.expected_state_version < 0
            or self.observed_at_ns < 0
            or self.pose_version < 0
        ):
            raise ValueError("navigation ACK versions and timestamp must be non-negative")
        if not 0.0 <= self.pose_confidence <= 1.0:
            raise ValueError("navigation pose_confidence must be in [0, 1]")
        if self.stable_frames <= 0:
            raise ValueError("navigation stable_frames must be positive")
        if self.face_center_error_px is not None and self.face_center_error_px < 0:
            raise ValueError("face_center_error_px must be non-negative")
        if self.status is NavigationAckStatus.SUCCEEDED:
            if self.error_code is not None or self.reason is not None:
                raise ValueError("successful navigation ACK cannot carry an error")
            if self.actual_pose is RobotPoseNode.UNKNOWN:
                raise ValueError("successful navigation ACK requires a known pose")
            if self.action is NavigationAction.MOVE_AND_ALIGN_TO_TARGET:
                if self.target_slot is None or not self.target_aligned:
                    raise ValueError(
                        "successful target navigation requires target and alignment"
                    )
        elif self.error_code is None or not (self.reason or "").strip():
            raise ValueError("failed navigation ACK requires error_code and reason")


class ChipObservationStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    OCCLUDED = "occluded"
    UNSTABLE = "unstable"


class ChipAmountScope(StrEnum):
    """Whether total_units describes the new chips or the street-area total."""

    NEW_CONTRIBUTION = "new_contribution"
    STREET_TOTAL = "street_total"


@dataclass(frozen=True, slots=True, order=True)
class ChipCount:
    denomination_units: int
    count: int

    def __post_init__(self) -> None:
        if self.denomination_units <= 0 or self.count <= 0:
            raise ValueError("chip denomination and count must be positive")


@dataclass(frozen=True, slots=True)
class ChipObservation:
    """Non-authoritative chip evidence for the already-selected acting seat."""

    observation_id: str
    hand_id: str
    expected_state_version: int
    focus_seat: Seat
    observed_at_ns: int
    status: ChipObservationStatus
    amount_scope: ChipAmountScope
    chip_counts: tuple[ChipCount, ...]
    total_units: int | None
    confidence: float | None
    stable_frames: int
    model_version: str
    calibration_version: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.hand_id.strip():
            raise ValueError("chip observation and hand IDs are required")
        if self.expected_state_version < 0 or self.observed_at_ns < 0:
            raise ValueError("chip state version and timestamp must be non-negative")
        if self.stable_frames <= 0:
            raise ValueError("chip stable_frames must be positive")
        if not self.model_version.strip() or not self.calibration_version.strip():
            raise ValueError("chip model and calibration versions are required")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("chip confidence must be in [0, 1]")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("chip quality_flags cannot contain blanks")
        denominations = tuple(item.denomination_units for item in self.chip_counts)
        if len(denominations) != len(set(denominations)):
            raise ValueError("chip denominations must be unique")
        if self.status is ChipObservationStatus.CONFIRMED:
            if self.total_units is None or self.total_units <= 0:
                raise ValueError("confirmed chip observation requires positive total_units")
            if self.confidence is None:
                raise ValueError("confirmed chip observation requires confidence")
            if not self.chip_counts:
                raise ValueError("confirmed chip observation requires chip counts")
            counted = sum(
                item.denomination_units * item.count for item in self.chip_counts
            )
            if counted != self.total_units:
                raise ValueError("chip counts must sum to total_units")
        elif self.total_units is not None or self.chip_counts:
            raise ValueError(
                "only confirmed chip observations may carry counts or total_units"
            )


class FaceEnrollmentStatus(StrEnum):
    """Result of one bounded, consented registration capture window."""

    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    MULTIPLE_FACES = "multiple_faces"
    LOW_QUALITY = "low_quality"


@dataclass(frozen=True, slots=True)
class FaceEnrollmentObservation:
    """Registration evidence without persisting images or face embeddings."""

    observation_id: str
    session_id: str
    expected_roster_version: int
    focus_seat: Seat
    observed_at_ns: int
    status: FaceEnrollmentStatus
    sample_count: int
    stable_frames: int
    model_version: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.session_id.strip():
            raise ValueError("face enrollment observation and session IDs are required")
        if self.expected_roster_version < 0 or self.observed_at_ns < 0:
            raise ValueError("face enrollment version and timestamp must be non-negative")
        if self.sample_count < 0 or self.stable_frames <= 0:
            raise ValueError(
                "face enrollment sample_count must be non-negative and "
                "stable_frames must be positive"
            )
        if not self.model_version.strip():
            raise ValueError("face enrollment model_version is required")
        if any(not flag.strip() for flag in self.quality_flags):
            raise ValueError("face enrollment quality_flags cannot contain blanks")
        if (
            self.status is FaceEnrollmentStatus.CONFIRMED
            and self.sample_count <= 0
        ):
            raise ValueError(
                "confirmed face enrollment requires a positive sample_count"
            )
        if (
            self.status is not FaceEnrollmentStatus.CONFIRMED
            and self.sample_count != 0
        ):
            raise ValueError(
                "unconfirmed face enrollment cannot carry accepted samples"
            )


class RobotInputKind(StrEnum):
    NONE = "none"
    NAVIGATION_ACK = "navigation_ack"
    LEGACY_DEALER_ACK = "legacy_dealer_ack"
    DISPENSE_ACK = "dispense_ack"
    VISUAL_SETTLE = "visual_settle"
    FACE_ENROLLMENT = "face_enrollment"
    FACE_IDENTITY = "face_identity"
    PLAYER_ACTION = "player_action"
    CHIP_OBSERVATION = "chip_observation"
    CARD_OBSERVATION = "card_observation"
    OPERATOR_CONTROL = "operator_control"


class RobotWorkflowNode(StrEnum):
    GAME_INTERNAL = "game_internal"
    WAITING_REGISTRATION_CONTROL = "waiting_registration_control"
    WAITING_FACE_ENROLLMENT = "waiting_face_enrollment"
    WAITING_SESSION_CONTROL = "waiting_session_control"
    WAITING_TARGET_ACK = "waiting_target_ack"
    WAITING_VISUAL_SETTLE = "waiting_visual_settle"
    WAITING_FACE_IDENTITY = "waiting_face_identity"
    WAITING_PLAYER_ACTION = "waiting_player_action"
    WAITING_CHIP_OBSERVATION = "waiting_chip_observation"
    WAITING_DISPENSE_ACK = "waiting_dispense_ack"
    WAITING_BOARD_REVEAL = "waiting_board_reveal"
    WAITING_POST_BOARD_DELAY = "waiting_post_board_delay"
    WAITING_SHOWDOWN_CARDS = "waiting_showdown_cards"
    WAITING_CARD_OBSERVATION = "waiting_card_observation"
    WAITING_OPERATOR_CONTROL = "waiting_operator_control"
    COMPLETE = "complete"
    VOIDED = "voided"


__all__ = [
    "ChipAmountScope",
    "ChipCount",
    "ChipObservation",
    "ChipObservationStatus",
    "FaceEnrollmentObservation",
    "FaceEnrollmentStatus",
    "NavigationAck",
    "NavigationAckStatus",
    "NavigationAction",
    "NavigationCommand",
    "NavigationErrorCode",
    "RobotInputKind",
    "RobotPoseNode",
    "RobotWorkflowNode",
]
