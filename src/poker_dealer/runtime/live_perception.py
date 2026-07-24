"""Shared-frame live adapters composed from the existing Stage 2 pilots."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import queue
import time
from typing import Mapping

import numpy as np

from poker_dealer.domain import (
    ActionEvidenceState,
    CardObservation,
    ControlIntent,
    ControlObservation,
    FramePacket,
    HandPhase,
    ObservationStatus,
    PlayerActionObservation,
    Seat,
    VisionSlot,
    role_seats,
)
from poker_dealer.io.camera import CameraReadStatus, OpenCVCamera
from poker_dealer.perception.actions import (
    ActionObservationContext,
    GestureFrameEvidence,
    GesturePilotConfig,
    GestureTemporalAdapter,
    MediaPipeGestureAdapter,
    MultimodalActionWindow,
    SpeechObservationAdapter,
    SpeechPilotConfig,
    SpeakerVerificationConfig,
    SpeechIntentKind,
    VoskSpeechRecognizer,
    classify_speech_intent,
)
from poker_dealer.perception.attribution import (
    ActorAttributionConfig,
    ActorBindingLease,
    MediaPipePoseAdapter,
    TargetPersonTracker,
    SessionSpeakerGallery,
    SpeakerVerificationState,
    attribute_hands_to_target,
)
from poker_dealer.perception.cards import (
    CardObservationPromoter,
    CardPilotConfig,
    CardSlotGeometryConfig,
    OpenCvCardRecognitionAdapter,
    crop_fixed_card_roi,
)
from poker_dealer.perception.identity import (
    DuplicateFaceEnrollmentError,
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityObservation,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
)

from .announcer import SpeechPlaybackGate
from .audio_input import AudioInputHealth, StreamingPcm16Resampler
from .ports import (
    ActionEvidence,
    ControlSource,
    FrameRead,
    FrameReadState,
    FrameSource,
    RuntimeEventSink,
    RuntimeObservationContext,
)
from .registration import (
    ROLE_ORDER,
    FrozenSessionRoster,
    RegistrationPhase,
    RegistrationRuntime,
)
from .visual_settle import VisualSettleGate, VisualSettleState

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class LivePerceptionConfig:
    identity_config: Path
    gesture_config: Path
    speech_config: Path
    speaker_config: Path
    attribution_config: Path
    card_config: Path
    card_geometry_config: Path
    consent_confirmed: bool
    speech_enabled: bool
    speech_device: int | str | None
    speech_capture_sample_rate_hz: int | None
    runtime_calibration_id: str
    target_geometry_validated: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_calibration_id.strip():
            raise ValueError("runtime calibration ID is required")
        if (
            self.speech_capture_sample_rate_hz is not None
            and self.speech_capture_sample_rate_hz <= 0
        ):
            raise ValueError("speech capture sample rate must be positive")


@dataclass(frozen=True, slots=True)
class RegistrationUiState:
    phase: str
    role: str
    seat: str
    completed_roles: tuple[str, ...]
    face_samples: int
    face_target: int
    voice_samples: int
    voice_target: int
    voice_active: bool
    prompt_playing: bool
    speech_enabled: bool
    alert_title: str | None
    alert_detail: str | None
    microphone_live: bool = False
    microphone_level: float = 0.0
    microphone_callback_blocks: int = 0
    simulated_roles: tuple[str, ...] = ()


def validate_live_perception_assets(
    config: LivePerceptionConfig,
) -> Mapping[str, object]:
    """Hash-check offline assets without opening a camera or microphone."""

    identity = FaceIdentityConfig.from_json(config.identity_config)
    gesture = GesturePilotConfig.from_json(config.gesture_config)
    speech = SpeechPilotConfig.from_json(config.speech_config)
    speaker = SpeakerVerificationConfig.from_json(config.speaker_config)
    attribution = ActorAttributionConfig.from_json(config.attribution_config)
    card = CardPilotConfig.from_json(config.card_config)
    geometry = CardSlotGeometryConfig.from_json(config.card_geometry_config)
    if config.target_geometry_validated and not geometry.target_geometry_validated:
        raise ValueError(
            "runtime profile cannot validate an unvalidated card-slot geometry"
        )
    identity_hashes = identity.verify_assets()
    gesture_hash = gesture.verify_model_asset()
    attribution.verify_pose_asset()
    card_hashes = card.verify_assets()
    speech_hash = speech.verify_model_asset() if config.speech_enabled else None
    speaker_hash = (
        speaker.verify_model_asset() if config.speech_enabled else None
    )
    return {
        "identity_hashes": list(identity_hashes),
        "gesture_hash": gesture_hash,
        "pose_hash": attribution.pose_asset_sha256,
        "card_hashes": list(card_hashes),
        "card_binding_mode": geometry.binding_mode,
        "logical_card_slot_count": len(VisionSlot),
        "card_pixel_roi_count": len(geometry.slots),
        "card_slot_count": len(geometry.slots),
        "card_geometry_calibration_id": geometry.calibration_id,
        "card_target_geometry_validated": geometry.target_geometry_validated,
        "speech_hash": speech_hash,
        "speaker_hash": speaker_hash,
        "speech_capture_sample_rate_hz": (
            config.speech_capture_sample_rate_hz
            or int(speech.audio["sample_rate_hz"])
        ),
        "speech_model_sample_rate_hz": int(speech.audio["sample_rate_hz"]),
        "speech_resampling_enabled": (
            config.speech_capture_sample_rate_hz is not None
            and config.speech_capture_sample_rate_hz
            != int(speech.audio["sample_rate_hz"])
        ),
        "runtime_calibration_id": config.runtime_calibration_id,
        "runtime_downloads": False,
        "frames_saved": False,
        "audio_saved": False,
    }


class InteractiveOpenCVFrameSource:
    """Read and display exactly one shared camera frame per runtime iteration."""

    def __init__(
        self,
        camera: OpenCVCamera,
        *,
        display: bool = True,
        window_name: str = "Poker Dealer - Unified Live Runtime",
        registration_observer: object | None = None,
    ) -> None:
        self.camera = camera
        self.display = display
        self.window_name = window_name
        self.registration_observer = registration_observer
        self.camera_epoch = 0
        self._last_key: int | None = None
        self._quit = False
        self._status_lines: tuple[str, ...] = ()
        self._registration_ui: RegistrationUiState | None = None
        self._face_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self._face_status: str | None = None
        self._action_marker: tuple[float, float, str, float] | None = None
        self._window_initialized = False

    def open(self) -> None:
        if not self.camera.is_open:
            self.camera.open()

    def set_status(self, *lines: str) -> None:
        self._status_lines = tuple(line for line in lines if line)

    def set_registration_status(
        self,
        *,
        phase: str,
        role: str,
        seat: str,
        completed_roles: tuple[str, ...],
        face_samples: int,
        face_target: int,
        voice_samples: int,
        voice_target: int,
        voice_active: bool,
        prompt_playing: bool,
        speech_enabled: bool,
        alert_title: str | None,
        alert_detail: str | None,
        microphone_live: bool = False,
        microphone_level: float = 0.0,
        microphone_callback_blocks: int = 0,
        simulated_roles: tuple[str, ...] = (),
    ) -> None:
        self._registration_ui = RegistrationUiState(
            phase=phase,
            role=role,
            seat=seat,
            completed_roles=completed_roles,
            face_samples=face_samples,
            face_target=face_target,
            voice_samples=voice_samples,
            voice_target=voice_target,
            voice_active=voice_active,
            prompt_playing=prompt_playing,
            speech_enabled=speech_enabled,
            alert_title=alert_title,
            alert_detail=alert_detail,
            microphone_live=microphone_live,
            microphone_level=microphone_level,
            microphone_callback_blocks=microphone_callback_blocks,
            simulated_roles=simulated_roles,
        )
        observer = self.registration_observer
        if observer is not None:
            observer.publish_registration_status(self._registration_ui)

    def set_face_detections(
        self,
        boxes: tuple[tuple[int, int, int, int], ...],
        *,
        status: str | None,
    ) -> None:
        self._face_boxes = boxes
        self._face_status = status
        observer = self.registration_observer
        if observer is not None:
            observer.publish_face_detections(boxes, status=status)

    def set_action_marker(
        self,
        marker: tuple[float, float, str, float] | None,
    ) -> None:
        self._action_marker = marker
        observer = self.registration_observer
        if observer is not None:
            observer.publish_action_marker(marker)

    def read(self) -> FrameRead:
        observer = self.registration_observer
        if observer is not None and observer.consume_quit_request():
            self._quit = True
        if self._quit:
            return FrameRead(
                FrameReadState.DISCONNECTED,
                time.monotonic_ns(),
                None,
                self.camera_epoch,
                "operator_exit",
            )
        read = self.camera.read()
        self.camera_epoch = self.camera.network_reconnects
        state = {
            CameraReadStatus.OK: FrameReadState.OK,
            CameraReadStatus.MISSING: FrameReadState.MISSING,
            CameraReadStatus.DISCONNECTED: FrameReadState.DISCONNECTED,
        }[read.status]
        if state is FrameReadState.OK and read.frame is not None:
            if observer is not None:
                observer.publish_frame(
                    read.frame.image,
                    observed_at_ns=read.observed_at_ns,
                )
        if state is FrameReadState.OK and read.frame is not None and self.display:
            if cv2 is None:
                raise RuntimeError("OpenCV UI is unavailable")
            display = self._render_display(read.frame.image)
            self._ensure_window()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            self._last_key = None if key == 255 else key
            if self._last_key in (ord("q"), 27):
                self._quit = True
            elif cv2.getWindowProperty(
                self.window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                self._quit = True
        return FrameRead(
            state,
            read.observed_at_ns,
            read.frame,
            self.camera_epoch,
            read.reason,
        )

    def _render_display(self, image: np.ndarray) -> np.ndarray:
        if self._registration_ui is not None:
            return self._render_registration_dashboard(
                image, self._registration_ui
            )
        display = image.copy()
        if self._action_marker is not None:
            marker_x, marker_y, _action, _confidence = self._action_marker
            height, width = display.shape[:2]
            center = (
                int(round(marker_x * width)),
                int(round(marker_y * height)),
            )
            cv2.circle(display, center, 9, (92, 214, 137), -1, cv2.LINE_AA)
            cv2.circle(display, center, 12, (8, 12, 8), 2, cv2.LINE_AA)
        for index, line in enumerate(self._status_lines):
            cv2.putText(
                display,
                line,
                (18, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return display

    def _render_registration_dashboard(
        self,
        image: np.ndarray,
        state: RegistrationUiState,
    ) -> np.ndarray:
        canvas_width, canvas_height = 1280, 720
        background = (22, 18, 14)
        panel = (34, 29, 24)
        panel_light = (44, 38, 31)
        border = (68, 59, 49)
        text = (238, 238, 235)
        muted = (157, 157, 151)
        accent = (184, 213, 55)
        accent_soft = (92, 102, 40)
        success = (137, 214, 92)
        canvas = np.full(
            (canvas_height, canvas_width, 3), background, dtype=np.uint8
        )

        self._text(canvas, "POKER DEALER", (28, 37), 0.52, accent, 1)
        self._text(canvas, "PLAYER REGISTRATION", (28, 70), 0.94, text, 2)
        enrollment_label = (
            "Face + voice enrollment"
            if state.voice_target > 0
            else "Face-only enrollment · voice commands available in game"
        )
        self._text(canvas, enrollment_label, (358, 67), 0.5, muted, 1)
        cv2.circle(canvas, (1012, 46), 6, success, -1, cv2.LINE_AA)
        self._text(canvas, "CAMERA LIVE", (1027, 52), 0.42, text, 1)
        mic_color = success if state.speech_enabled else muted
        cv2.circle(canvas, (1141, 46), 6, mic_color, -1, cv2.LINE_AA)
        self._text(
            canvas,
            "MIC ON" if state.speech_enabled else "MIC OFF",
            (1156, 52),
            0.42,
            text if state.speech_enabled else muted,
            1,
        )

        video_x, video_y, video_w, video_h = 28, 86, 844, 552
        cv2.rectangle(
            canvas,
            (video_x - 2, video_y - 2),
            (video_x + video_w + 2, video_y + video_h + 2),
            border,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (video_x, video_y),
            (video_x + video_w, video_y + video_h),
            panel,
            -1,
        )
        image_height, image_width = image.shape[:2]
        scale = min(video_w / image_width, video_h / image_height)
        scaled_width = max(1, int(round(image_width * scale)))
        scaled_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        image_x = video_x + (video_w - scaled_width) // 2
        image_y = video_y + (video_h - scaled_height) // 2
        canvas[
            image_y : image_y + scaled_height,
            image_x : image_x + scaled_width,
        ] = resized
        for x, y, width, height in self._face_boxes:
            x1 = image_x + int(round(x * scale))
            y1 = image_y + int(round(y * scale))
            x2 = image_x + int(round((x + width) * scale))
            y2 = image_y + int(round((y + height) * scale))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), accent, 3, cv2.LINE_AA)
            label = self._face_status or "FACE DETECTED"
            label_width = min(max(150, 10 + len(label) * 10), max(150, x2 - x1))
            label_y = max(image_y + 26, y1)
            cv2.rectangle(
                canvas,
                (x1, label_y - 25),
                (x1 + label_width, label_y),
                accent,
                -1,
            )
            self._text(
                canvas,
                label,
                (x1 + 7, label_y - 7),
                0.42,
                background,
                1,
            )
        cv2.rectangle(
            canvas,
            (video_x + 16, video_y + 16),
            (video_x + 188, video_y + 47),
            panel,
            -1,
        )
        cv2.circle(canvas, (video_x + 31, video_y + 31), 5, success, -1)
        self._text(
            canvas,
            "LIVE  /  NO RECORDING",
            (video_x + 43, video_y + 37),
            0.38,
            text,
            1,
        )

        side_x, side_y, side_w, side_h = 896, 86, 356, 552
        cv2.rectangle(
            canvas,
            (side_x, side_y),
            (side_x + side_w, side_y + side_h),
            panel,
            -1,
        )
        cv2.rectangle(
            canvas,
            (side_x, side_y),
            (side_x + side_w, side_y + side_h),
            border,
            1,
            cv2.LINE_AA,
        )
        stage, instruction, detail, stage_color = self._registration_copy(state)
        cv2.rectangle(
            canvas,
            (side_x + 22, side_y + 22),
            (side_x + side_w - 22, side_y + 58),
            stage_color,
            -1,
        )
        self._text(
            canvas, stage, (side_x + 36, side_y + 47), 0.5, background, 2
        )
        self._text(
            canvas, "CURRENT PLAYER", (side_x + 22, side_y + 94), 0.39, muted, 1
        )
        role_label = state.role.replace("_", " ").upper()
        self._text(canvas, role_label, (side_x + 22, side_y + 133), 0.8, text, 2)
        self._text(
            canvas,
            state.seat.replace("_", " ").upper(),
            (side_x + side_w - 104, side_y + 130),
            0.43,
            muted,
            1,
        )
        cv2.line(
            canvas,
            (side_x + 22, side_y + 154),
            (side_x + side_w - 22, side_y + 154),
            border,
            1,
        )
        self._text(
            canvas, "CURRENT STEP", (side_x + 22, side_y + 188), 0.39, muted, 1
        )
        self._text(
            canvas, instruction, (side_x + 22, side_y + 226), 0.66, text, 2
        )
        self._text(canvas, detail, (side_x + 22, side_y + 256), 0.43, muted, 1)

        if state.voice_active:
            progress_value, progress_total = state.voice_samples, state.voice_target
        else:
            progress_value, progress_total = state.face_samples, state.face_target
        self._progress_bar(
            canvas,
            (side_x + 22, side_y + 278),
            side_w - 44,
            progress_value,
            progress_total,
            accent,
            panel_light,
        )
        self._text(
            canvas,
            f"{progress_value} / {progress_total}",
            (side_x + side_w - 70, side_y + 310),
            0.4,
            muted,
            1,
        )

        self._text(
            canvas, "TABLE ROSTER", (side_x + 22, side_y + 332), 0.39, muted, 1
        )
        roles = (
            ("button", "BTN"),
            ("small_blind", "SB"),
            ("big_blind", "BB"),
            ("under_the_gun", "UTG"),
        )
        for index, (role, short_label) in enumerate(roles):
            row_y = side_y + 354 + index * 45
            completed = role in state.completed_roles
            simulated = role in state.simulated_roles
            current = role == state.role
            row_color = panel_light if not current else accent_soft
            cv2.rectangle(
                canvas,
                (side_x + 22, row_y),
                (side_x + side_w - 22, row_y + 36),
                row_color,
                -1,
            )
            dot_color = success if completed else accent if current else muted
            cv2.circle(canvas, (side_x + 41, row_y + 18), 6, dot_color, -1)
            self._text(
                canvas,
                short_label,
                (side_x + 58, row_y + 24),
                0.47,
                text,
                1,
            )
            status = (
                "SIMULATED"
                if simulated
                else "COMPLETE"
                if completed
                else "ACTIVE"
                if current
                else "PENDING"
            )
            self._text(
                canvas,
                status,
                (side_x + side_w - 102, row_y + 24),
                0.37,
                dot_color,
                1,
            )

        footer_y = 666
        self._text(canvas, "CONTROLS", (28, footer_y + 25), 0.4, muted, 1)
        controls = (
            ("E", "Capture face"),
            ("S", "Start session"),
            ("X", "Clear roster"),
            ("Q", "Quit"),
        )
        enabled_controls = {
            "E": (
                state.phase == RegistrationPhase.READY_FOR_FACE.value
                and not state.voice_active
            ),
            "S": (
                state.phase == RegistrationPhase.READY_TO_START.value
                and not state.voice_active
            ),
            "X": state.phase != RegistrationPhase.STARTED.value,
            "Q": True,
        }
        cursor_x = 126
        for key, label in controls:
            cv2.rectangle(
                canvas,
                (cursor_x, footer_y),
                (cursor_x + 34, footer_y + 34),
                panel_light,
                -1,
            )
            cv2.rectangle(
                canvas,
                (cursor_x, footer_y),
                (cursor_x + 34, footer_y + 34),
                border,
                1,
            )
            control_color = accent if enabled_controls[key] else muted
            self._text(
                canvas,
                key,
                (cursor_x + 11, footer_y + 24),
                0.48,
                control_color,
                2,
            )
            self._text(
                canvas,
                label,
                (cursor_x + 45, footer_y + 24),
                0.43,
                text if enabled_controls[key] else muted,
                1,
            )
            cursor_x += 196 if key != "S" else 210
        self._text(
            canvas,
            "ESC also quits",
            (canvas_width - 125, footer_y + 24),
            0.36,
            muted,
            1,
        )
        return canvas

    def _ensure_window(self) -> None:
        if self._window_initialized:
            return
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        initial_width, initial_height = 1280, 720
        try:
            screen_width = int(ctypes.windll.user32.GetSystemMetrics(0))
            screen_height = int(ctypes.windll.user32.GetSystemMetrics(1))
            scale = min(
                1.0,
                max(0.5, (screen_width - 80) / initial_width),
                max(0.5, (screen_height - 140) / initial_height),
            )
            initial_width = int(round(initial_width * scale))
            initial_height = int(round(initial_height * scale))
        except (AttributeError, OSError, ValueError):
            pass
        cv2.resizeWindow(self.window_name, initial_width, initial_height)
        self._window_initialized = True

    def _registration_copy(
        self,
        state: RegistrationUiState,
    ) -> tuple[str, str, str, tuple[int, int, int]]:
        phrases = ("CHECK", "CALL", "RAISE")
        if state.alert_title is not None:
            return (
                "DUPLICATE PLAYER",
                state.alert_title,
                state.alert_detail or "Press E to try again",
                (79, 181, 246),
            )
        if state.voice_active:
            if state.prompt_playing:
                return (
                    "PLAYING PROMPT",
                    "Listen to the prompt",
                    "The microphone will resume automatically",
                    (79, 181, 246),
                )
            phrase_index = min(state.voice_samples, len(phrases) - 1)
            return (
                "RECORDING VOICE",
                f"Say {phrases[phrase_index]}",
                "Speak clearly, then pause",
                (184, 213, 55),
            )
        if state.phase == RegistrationPhase.CAPTURING_FACE.value:
            return (
                "CAPTURING FACE",
                "Look at the camera",
                self._face_status or "Keep one face centered and hold still",
                (184, 213, 55),
            )
        if state.phase == RegistrationPhase.READY_TO_START.value:
            return (
                "ROSTER READY",
                "Press S to continue",
                "All four players are registered",
                (137, 214, 92),
            )
        return (
            "READY FOR FACE",
            "Press E to begin",
            self._face_status or "Look at the camera before you start",
            (137, 214, 92),
        )

    @staticmethod
    def _progress_bar(
        canvas: np.ndarray,
        origin: tuple[int, int],
        width: int,
        value: int,
        total: int,
        foreground: tuple[int, int, int],
        background: tuple[int, int, int],
    ) -> None:
        x, y = origin
        cv2.rectangle(canvas, (x, y), (x + width, y + 10), background, -1)
        ratio = min(1.0, max(0.0, value / max(1, total)))
        if ratio > 0:
            cv2.rectangle(
                canvas,
                (x, y),
                (x + int(round(width * ratio)), y + 10),
                foreground,
                -1,
            )

    @staticmethod
    def _text(
        canvas: np.ndarray,
        value: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(
            canvas,
            value,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def consume_key(self, *keys: int) -> int | None:
        if self._last_key not in keys:
            return None
        key = self._last_key
        self._last_key = None
        return key

    def peek_key(self) -> int | None:
        return self._last_key

    def close(self) -> None:
        if self.display and cv2 is not None and self._window_initialized:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._window_initialized = False


class LiveKeyboardControlSource:
    def __init__(self, frame_source: InteractiveOpenCVFrameSource) -> None:
        from poker_dealer.domain import LaptopControlAdapter

        self.frame_source = frame_source
        self.adapter = LaptopControlAdapter()

    def poll_controls(self, observed_at_ns: int):
        key = self.frame_source.peek_key()
        if key is None:
            return ()
        observation = self.adapter.process_key(key, observed_at_ns)
        if observation is None:
            return ()
        self.frame_source.consume_key(key)
        return (observation,)


class LivePerceptionSession:
    """Registration, identity, action and card adapters sharing one frame."""

    def __init__(
        self,
        config: LivePerceptionConfig,
        frame_source: InteractiveOpenCVFrameSource,
        speech_playback_gate: SpeechPlaybackGate | None = None,
    ) -> None:
        self.config = config
        self.frame_source = frame_source
        self.identity_config = FaceIdentityConfig.from_json(config.identity_config)
        self.gesture_config = GesturePilotConfig.from_json(config.gesture_config)
        self.speech_config = SpeechPilotConfig.from_json(config.speech_config)
        self.speaker_config = SpeakerVerificationConfig.from_json(
            config.speaker_config
        )
        self.attribution_config = ActorAttributionConfig.from_json(
            config.attribution_config
        )
        self.card_config = CardPilotConfig.from_json(config.card_config)
        self.card_geometry = CardSlotGeometryConfig.from_json(
            config.card_geometry_config
        )
        if (
            config.target_geometry_validated
            and not self.card_geometry.target_geometry_validated
        ):
            raise ValueError(
                "runtime profile cannot validate an unvalidated card-slot geometry"
            )
        self.gesture_config = replace(
            self.gesture_config,
            model=replace(
                self.gesture_config.model,
                num_hands=self.attribution_config.max_hands,
            ),
            calibration_version=(
                f"{self.gesture_config.calibration_version}:"
                f"{config.runtime_calibration_id}"
            ),
        )
        self.card_config = replace(
            self.card_config,
            calibration_version=(
                f"{self.card_config.calibration_version}:"
                f"{config.runtime_calibration_id}"
            ),
        )
        self.gallery: SessionFaceGallery | None = None
        self.speaker_gallery: SessionSpeakerGallery | None = None
        self.face_model: OpenCvFaceIdentityAdapter | None = None
        self.gesture_model: MediaPipeGestureAdapter | None = None
        self.pose_model: MediaPipePoseAdapter | None = None
        self.card_model: OpenCvCardRecognitionAdapter | None = None
        self.identity_temporal = FaceIdentityTemporalAdapter(self.identity_config)
        self.gesture_temporal = GestureTemporalAdapter(self.gesture_config)
        self.card_temporal = CardObservationPromoter(self.card_config)
        self._runtime_controls: tuple[ControlObservation, ...] = ()
        self.actor_lease = ActorBindingLease(
            lease_ms=self.attribution_config.actor_lease_ms
        )
        self.person_tracker = TargetPersonTracker(self.attribution_config)
        self.multimodal = MultimodalActionWindow(
            decision_wait_ms=500,
            max_skew_ms=3000,
            allow_speech_single_source=True,
        )
        self.visual_settle = VisualSettleGate()
        self._visual_started = False
        self._identity_context: tuple[int, Seat] | None = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=int(self.speech_config.audio["queue_max_blocks"])
        )
        self._audio_stream = None
        self._audio_event_sink: RuntimeEventSink | None = None
        self._audio_health = AudioInputHealth()
        self._audio_disconnected = False
        self._audio_unavailable_reported = False
        self._audio_last_reconnect_attempt_ns: int | None = None
        self._audio_last_status_events = 0
        self._audio_stale_after_ms = 2000
        self._audio_reconnect_cooldown_ms = 2000
        self._audio_capture_rate_hz = (
            config.speech_capture_sample_rate_hz
            or int(self.speech_config.audio["sample_rate_hz"])
        )
        self._audio_resampler = StreamingPcm16Resampler(
            self._audio_capture_rate_hz,
            int(self.speech_config.audio["sample_rate_hz"]),
        )
        self._speech_playback_gate = speech_playback_gate
        self._speech_was_suppressed = False
        self._last_speech_feedback_ns: int | None = None
        self._speech_recognizer: VoskSpeechRecognizer | None = None
        self._speech_adapter = SpeechObservationAdapter(self.speech_config)
        self._accepted_speech_confidence: float | None = None
        self._accepted_speech_player_id: str | None = None

    def open(self, session_id: str) -> None:
        if not self.config.consent_confirmed:
            raise PermissionError("explicit face-enrollment consent is required")
        self.gallery = SessionFaceGallery(self.identity_config, session_id)
        self.speaker_gallery = SessionSpeakerGallery(
            session_id,
            minimum_samples=self.speaker_config.minimum_samples,
            minimum_speaker_frames=self.speaker_config.minimum_speaker_frames,
            minimum_similarity=self.speaker_config.minimum_similarity,
            minimum_margin=self.speaker_config.minimum_margin,
        )
        self.face_model = OpenCvFaceIdentityAdapter(self.identity_config)
        self.gesture_model = MediaPipeGestureAdapter(self.gesture_config)
        self.pose_model = MediaPipePoseAdapter(self.attribution_config)
        self.card_model = OpenCvCardRecognitionAdapter(self.card_config)
        if self.config.speech_enabled:
            if sd is None:
                raise RuntimeError("sounddevice is unavailable")
            self._speech_recognizer = VoskSpeechRecognizer(
                self.speech_config, self.speaker_config
            )
            self._start_audio_stream()

    def _start_audio_stream(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is unavailable")
        model_rate_hz = int(self.speech_config.audio["sample_rate_hz"])
        model_blocksize = int(self.speech_config.audio["blocksize_frames"])
        capture_blocksize = max(
            1,
            round(
                model_blocksize
                * self._audio_capture_rate_hz
                / model_rate_hz
            ),
        )
        self._audio_resampler.reset()
        self._audio_health.reset_opened()

        def callback(indata: bytes, frames: int, _time, status) -> None:
            raw_pcm = bytes(indata)
            samples = np.frombuffer(raw_pcm, dtype="<i2").astype(
                np.float32, copy=False
            )
            if len(samples):
                rms_level = min(
                    1.0,
                    float(np.sqrt(np.mean(samples * samples))) / 32768.0,
                )
                peak_level = min(
                    1.0,
                    float(np.max(np.abs(samples))) / 32768.0,
                )
            else:
                rms_level = 0.0
                peak_level = 0.0
            self._audio_health.record_callback(
                frames,
                status,
                rms_level=rms_level,
                peak_level=peak_level,
            )
            if (
                self._speech_playback_gate is not None
                and self._speech_playback_gate.is_suppressed()
            ):
                return
            pcm = self._audio_resampler.process(raw_pcm)
            if not pcm:
                return
            try:
                self._audio_queue.put_nowait(pcm)
            except queue.Full:
                pass

        self._audio_stream = sd.RawInputStream(
            samplerate=self._audio_capture_rate_hz,
            blocksize=capture_blocksize,
            device=self.config.speech_device,
            dtype=str(self.speech_config.audio["dtype"]),
            channels=1,
            callback=callback,
        )
        self._audio_stream.start()

    def close(self) -> None:
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        if self.gesture_model is not None:
            self.gesture_model.close()
        if self.pose_model is not None:
            self.pose_model.close()
        if self.gallery is not None:
            self.gallery.clear()
        if self.speaker_gallery is not None:
            self.speaker_gallery.clear()
        self.actor_lease.clear()
        self.person_tracker.clear()
        self.multimodal.clear()

    def set_runtime_event_sink(self, event_sink: RuntimeEventSink) -> None:
        self._audio_event_sink = event_sink
        if self.config.speech_enabled:
            self._emit_audio_event(
                "audio_input_opened",
                observed_at_ns=time.monotonic_ns(),
                payload={
                    "device": self.config.speech_device,
                    "capture_sample_rate_hz": self._audio_capture_rate_hz,
                    "model_sample_rate_hz": int(
                        self.speech_config.audio["sample_rate_hz"]
                    ),
                    "resampling_enabled": (
                        self._audio_capture_rate_hz
                        != int(self.speech_config.audio["sample_rate_hz"])
                    ),
                },
            )

    def _emit_audio_event(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        if self._audio_event_sink is not None:
            self._audio_event_sink.emit(
                kind,
                observed_at_ns=observed_at_ns,
                payload=payload or {},
            )

    def _emit_speech_feedback(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object] | None = None,
        cooldown_ms: int = 1500,
    ) -> None:
        if (
            self._last_speech_feedback_ns is not None
            and observed_at_ns - self._last_speech_feedback_ns
            < cooldown_ms * 1_000_000
        ):
            return
        self._last_speech_feedback_ns = observed_at_ns
        self._emit_audio_event(
            kind,
            observed_at_ns=observed_at_ns,
            payload=payload,
        )

    def _restart_audio_stream(self) -> None:
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
            except Exception:
                pass
            try:
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        self._discard_audio_queue()
        self._start_audio_stream()

    def _audio_input_is_healthy(self, observed_at_ns: int) -> bool:
        if not self.config.speech_enabled:
            return True
        snapshot = self._audio_health.snapshot()
        try:
            stream_active = bool(
                self._audio_stream is not None
                and getattr(self._audio_stream, "active", False)
            )
        except Exception:
            stream_active = False
        if snapshot.status_events > self._audio_last_status_events:
            self._audio_last_status_events = snapshot.status_events
            self._emit_audio_event(
                "microphone_status",
                observed_at_ns=observed_at_ns,
                payload={
                    "status_events": snapshot.status_events,
                    "last_status": snapshot.last_status,
                    "capture_sample_rate_hz": self._audio_capture_rate_hz,
                },
            )
        stale = snapshot.is_stale(
            observed_at_ns, self._audio_stale_after_ms
        )
        if stream_active and not stale:
            if self._audio_disconnected:
                self._emit_audio_event(
                    "audio_link_restored",
                    observed_at_ns=observed_at_ns,
                    payload={
                        "capture_sample_rate_hz": self._audio_capture_rate_hz,
                        "model_sample_rate_hz": int(
                            self.speech_config.audio["sample_rate_hz"]
                        ),
                    },
                )
            self._audio_disconnected = False
            self._audio_unavailable_reported = False
            return True
        if not self._audio_disconnected:
            self._emit_audio_event(
                "audio_link_lost",
                observed_at_ns=observed_at_ns,
                payload={
                    "stream_active": stream_active,
                    "stale": stale,
                    "last_callback_at_ns": snapshot.last_callback_at_ns,
                },
            )
            self._audio_disconnected = True
        if (
            self._audio_last_reconnect_attempt_ns is not None
            and observed_at_ns - self._audio_last_reconnect_attempt_ns
            < self._audio_reconnect_cooldown_ms * 1_000_000
        ):
            return False
        self._audio_last_reconnect_attempt_ns = observed_at_ns
        try:
            self._restart_audio_stream()
        except Exception as exc:
            if not self._audio_unavailable_reported:
                self._emit_audio_event(
                    "microphone_unavailable",
                    observed_at_ns=observed_at_ns,
                    payload={"reason": type(exc).__name__},
                )
                self._audio_unavailable_reported = True
            raise RuntimeError("speech input is unavailable") from exc
        return False

    def acquire_roster(
        self,
        *,
        frame_source: FrameSource,
        control_source: ControlSource,
        event_sink: RuntimeEventSink,
        session_id: str,
        button: Seat,
        deadline_ns: int,
        simulated_seats: Mapping[Seat, str] | None = None,
    ) -> FrozenSessionRoster:
        if (
            self.gallery is None
            or self.face_model is None
            or self.speaker_gallery is None
        ):
            raise RuntimeError("live perception session is not open")
        registration = RegistrationRuntime(session_id, button)
        simulated_participants = dict(simulated_seats or {})

        def restore_simulated_participants(observed_at_ns: int) -> None:
            for seat, participant_id in simulated_participants.items():
                participant = registration.add_simulated_participant(
                    seat=seat,
                    participant_id=participant_id,
                )
                event_sink.emit(
                    "registration_simulated_participant_added",
                    observed_at_ns=observed_at_ns,
                    payload={
                        "player_id": participant.participant_id,
                        "seat": participant.seat.value,
                        "role": participant.initial_role.value,
                        "simulated": True,
                        "face_enrolled": False,
                        "voice_enrolled": False,
                    },
                )

        restore_simulated_participants(time.monotonic_ns())
        samples = []
        last_sample_ns: int | None = None
        last_face_preview_ns: int | None = None
        alert_title: str | None = None
        alert_detail: str | None = None
        camera_outage_started_ns: int | None = None
        last_camera_frame_ns: int | None = None
        last_camera_epoch = frame_source.camera_epoch

        def advance_registration_role() -> None:
            remaining = [
                role
                for role in ROLE_ORDER
                if role_seats(button)[role] not in registration.registered_seats
            ]
            if remaining:
                registration.select_role(remaining[0])

        def update_registration_ui() -> None:
            completed_roles = tuple(
                participant.initial_role.value
                for participant in registration.participants
                if participant.simulated or participant.face_sample_count > 0
            )
            simulated_roles = tuple(
                participant.initial_role.value
                for participant in registration.participants
                if participant.simulated
            )
            if self.config.speech_enabled:
                audio_snapshot = self._audio_health.snapshot()
                microphone_live = bool(
                    self._audio_stream is not None
                    and getattr(self._audio_stream, "active", False)
                    and not audio_snapshot.is_stale(
                        time.monotonic_ns(), self._audio_stale_after_ms
                    )
                )
                microphone_level = audio_snapshot.peak_level
                microphone_callback_blocks = audio_snapshot.callback_blocks
            else:
                microphone_live = False
                microphone_level = 0.0
                microphone_callback_blocks = 0
            self.frame_source.set_registration_status(
                phase=registration.phase.value,
                role=registration.focus_role.value,
                seat=registration.focus_seat.value,
                completed_roles=completed_roles,
                face_samples=len(samples),
                face_target=self.identity_config.minimum_samples,
                voice_samples=0,
                voice_target=0,
                voice_active=False,
                prompt_playing=False,
                speech_enabled=self.config.speech_enabled,
                alert_title=alert_title,
                alert_detail=alert_detail,
                microphone_live=microphone_live,
                microphone_level=microphone_level,
                microphone_callback_blocks=microphone_callback_blocks,
                simulated_roles=simulated_roles,
            )
            if registration.phase in {
                RegistrationPhase.READY_TO_START,
                RegistrationPhase.STARTED,
            }:
                self.frame_source.set_face_detections((), status=None)

        while time.monotonic_ns() < deadline_ns:
            update_registration_ui()
            read = frame_source.read()
            if read.camera_epoch != last_camera_epoch:
                event_sink.emit(
                    "camera_reconnected",
                    observed_at_ns=read.observed_at_ns,
                    payload={
                        "previous_camera_epoch": last_camera_epoch,
                        "camera_epoch": read.camera_epoch,
                        "reconnect_count": read.camera_epoch,
                    },
                )
                last_camera_epoch = read.camera_epoch
            if read.state is FrameReadState.DISCONNECTED:
                event_sink.emit(
                    "camera_disconnected",
                    observed_at_ns=read.observed_at_ns,
                    payload={
                        "reason": read.reason or "unknown",
                        "camera_epoch": read.camera_epoch,
                    },
                )
                raise RuntimeError(read.reason or "camera disconnected during registration")
            if read.state is FrameReadState.MISSING:
                if camera_outage_started_ns is None:
                    camera_outage_started_ns = read.observed_at_ns
                    event_sink.emit(
                        "camera_link_lost",
                        observed_at_ns=read.observed_at_ns,
                        payload={
                            "reason": read.reason or "unknown",
                            "camera_epoch": read.camera_epoch,
                        },
                    )
                continue
            if read.frame is None:
                continue
            if camera_outage_started_ns is not None:
                event_sink.emit(
                    "camera_link_restored",
                    observed_at_ns=read.observed_at_ns,
                    payload={
                        "outage_ms": max(
                            0,
                            (read.observed_at_ns - camera_outage_started_ns)
                            // 1_000_000,
                        ),
                        "camera_epoch": read.camera_epoch,
                    },
                )
                camera_outage_started_ns = None
            if (
                last_camera_frame_ns is not None
                and read.observed_at_ns - last_camera_frame_ns >= 500_000_000
            ):
                event_sink.emit(
                    "camera_frame_gap",
                    observed_at_ns=read.observed_at_ns,
                    payload={
                        "gap_ms": (
                            read.observed_at_ns - last_camera_frame_ns
                        )
                        // 1_000_000,
                        "dropped_before": read.frame.dropped_before,
                        "camera_epoch": read.camera_epoch,
                    },
                )
            last_camera_frame_ns = read.observed_at_ns
            frame = read.frame
            for control in control_source.poll_controls(read.observed_at_ns):
                outcome = registration.accept_control(control)
                event_sink.emit(
                    "registration_control",
                    observed_at_ns=control.observed_at_ns,
                    payload={
                        "observation_id": control.observation_id,
                        "intent": control.intent.value,
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "focus_role": registration.focus_role.value,
                    },
                )
                if control.intent is ControlIntent.CLEAR and outcome.accepted:
                    self.gallery.clear()
                    self.speaker_gallery.clear()
                    restore_simulated_participants(control.observed_at_ns)
                    samples = []
                    last_sample_ns = None
                    alert_title = None
                    alert_detail = None
                if control.intent is ControlIntent.CANCEL and outcome.accepted:
                    for sample in samples:
                        embedding = getattr(sample, "embedding", None)
                        if embedding is not None:
                            embedding.fill(0.0)
                    samples = []
                    last_sample_ns = None
                if (
                    control.intent is ControlIntent.CONFIRM
                    and outcome.accepted
                ):
                    alert_title = None
                    alert_detail = None
                    self.frame_source.set_face_detections(
                        (), status="SEARCHING FOR ONE FACE"
                    )
                if outcome.roster is not None:
                    return outcome.roster
            if (
                registration.phase is RegistrationPhase.READY_FOR_FACE
                and (
                    last_face_preview_ns is None
                    or frame.captured_at_ns - last_face_preview_ns
                    >= 100_000_000
                )
            ):
                preview = self.face_model.preview(frame)
                last_face_preview_ns = frame.captured_at_ns
                if preview.detected_face_count == 0:
                    preview_status = "NO FACE - MOVE INTO VIEW"
                elif preview.detected_face_count > 1:
                    preview_status = "ONE PERSON ONLY"
                elif alert_title is not None:
                    preview_status = "CHANGE PLAYER - PRESS E"
                else:
                    preview_status = "FACE READY - PRESS E"
                self.frame_source.set_face_detections(
                    preview.boxes_xywh,
                    status=preview_status,
                )
            if (
                registration.phase is RegistrationPhase.CAPTURING_FACE
            ):
                evidence = self.face_model.analyze(frame)
                can_sample = (
                    evidence.detected_face_count == 1
                    and len(evidence.features) == 1
                    and (
                        last_sample_ns is None
                        or evidence.observed_at_ns - last_sample_ns >= 150_000_000
                    )
                )
                if can_sample:
                    samples.append(evidence.features[0])
                    last_sample_ns = evidence.observed_at_ns
                if evidence.detected_face_count == 0:
                    face_status = "NO FACE - MOVE INTO VIEW"
                elif evidence.detected_face_count > 1:
                    face_status = "ONE PERSON ONLY"
                elif len(evidence.features) != 1:
                    face_status = "MOVE CLOSER TO THE CAMERA"
                else:
                    face_status = (
                        f"FACE DETECTED  "
                        f"{len(samples)} / {self.identity_config.minimum_samples}"
                    )
                self.frame_source.set_face_detections(
                    tuple(feature.bbox_xywh for feature in evidence.features),
                    status=face_status,
                )
                if len(samples) >= self.identity_config.minimum_samples:
                    player_id = registration.participant_id
                    seat = registration.focus_seat
                    try:
                        self.gallery.enroll(
                            player_id,
                            seat,
                            samples,
                            consent_granted=self.config.consent_confirmed,
                        )
                    except DuplicateFaceEnrollmentError as exc:
                        existing_role = next(
                            role
                            for role, mapped_seat in role_seats(button).items()
                            if mapped_seat is exc.existing_seat
                        )
                        attempted_role = registration.focus_role
                        registration.reject_face_enrollment()
                        samples = []
                        last_sample_ns = None
                        alert_title = (
                            "Already registered as "
                            f"{existing_role.value.replace('_', ' ').title()}"
                        )
                        alert_detail = (
                            f"{attempted_role.value.replace('_', ' ').title()} "
                            "requires a different player"
                        )
                        self.frame_source.set_face_detections(
                            tuple(
                                feature.bbox_xywh
                                for feature in evidence.features
                            ),
                            status=(
                                "ALREADY REGISTERED: "
                                f"{existing_role.value.replace('_', ' ').upper()}"
                            ),
                        )
                        event_sink.emit(
                            "registration_face_rejected",
                            observed_at_ns=evidence.observed_at_ns,
                            payload={
                                "reason": "duplicate_face",
                                "role": attempted_role.value,
                                "seat": seat.value,
                                "existing_role": existing_role.value,
                                "existing_seat": exc.existing_seat.value,
                                "similarity": round(exc.similarity, 6),
                                "threshold": exc.threshold,
                                "retryable": True,
                                "frames_saved": False,
                                "embeddings_logged": False,
                            },
                        )
                        continue
                    participant = registration.complete_face_enrollment(len(samples))
                    event_sink.emit(
                        "registration_enrolled",
                        observed_at_ns=evidence.observed_at_ns,
                        payload={
                            "player_id": participant.participant_id,
                            "seat": participant.seat.value,
                            "role": participant.initial_role.value,
                            "sample_count": participant.face_sample_count,
                            "speaker_enrollment_required": False,
                        },
                    )
                    samples = []
                    last_sample_ns = None
                    event_sink.emit(
                        "speaker_enrollment_skipped",
                        observed_at_ns=evidence.observed_at_ns,
                        payload={
                            "player_id": participant.participant_id,
                            "seat": participant.seat.value,
                            "reason": "temporarily_disabled",
                            "voice_enrolled": False,
                            "embeddings_logged": False,
                            "audio_saved": False,
                        },
                    )
                    advance_registration_role()
        raise TimeoutError("registration deadline expired")

    def accept_runtime_controls(
        self,
        controls: tuple[ControlObservation, ...],
        context: RuntimeObservationContext,
    ) -> None:
        del context
        self._runtime_controls = controls

    def _consume_runtime_intent(self, *intents: ControlIntent) -> bool:
        matched = any(item.intent in intents for item in self._runtime_controls)
        self._runtime_controls = tuple(
            item for item in self._runtime_controls if item.intent not in intents
        )
        return matched

    def reset_visual_settle(self, context: RuntimeObservationContext) -> None:
        del context
        self.visual_settle.clear()
        self._visual_started = False

    def visual_is_settled(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> bool | None:
        del observed_at_ns
        if frame is None:
            return None
        if not self._visual_started:
            self.visual_settle.begin(
                started_at_ns=frame.captured_at_ns,
                sequence_watermark=max(0, frame.sequence_id - 1),
                camera_epoch=context.camera_epoch,
            )
            self._visual_started = True
        observation = self.visual_settle.observe(
            frame, camera_epoch=context.camera_epoch
        )
        if observation.state is VisualSettleState.TIMED_OUT:
            return False
        return observation.state is VisualSettleState.SETTLED

    @staticmethod
    def _runtime_seat_label(seat: Seat) -> str:
        return seat.value.removeprefix("seat_").upper()

    def _publish_runtime_face(
        self,
        face: object,
        *,
        status: str,
    ) -> None:
        setter = getattr(self.frame_source, "set_face_detections", None)
        if setter is None:
            return
        features = getattr(face, "features", ())
        setter(
            tuple(feature.bbox_xywh for feature in features),
            status=status,
        )

    def _identity_ui_status(
        self,
        state: FaceIdentityState,
        seat: Seat,
    ) -> str:
        label = self._runtime_seat_label(seat)
        if state is FaceIdentityState.NO_FACE:
            return f"LOOK AT CAMERA · VERIFYING {label}"
        if state is FaceIdentityState.MULTIPLE_FACES:
            return "ONE PLAYER ONLY"
        if state is FaceIdentityState.SEAT_MISMATCH:
            return f"WRONG PLAYER · EXPECTING {label}"
        if state in {
            FaceIdentityState.UNKNOWN,
            FaceIdentityState.AMBIGUOUS,
        }:
            return f"PLAYER NOT RECOGNIZED · EXPECTING {label}"
        if state is FaceIdentityState.LOW_QUALITY:
            return "MOVE CLOSER · HOLD STILL"
        return f"VERIFYING {label}"

    def observe_identity(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> FaceIdentityObservation | None:
        del observed_at_ns
        if frame is None or context.focus_seat is None:
            return None
        if self.gallery is None or self.face_model is None or self.pose_model is None:
            raise RuntimeError("live perception session is not open")
        key = (context.state_version, context.focus_seat)
        if key != self._identity_context:
            self._reset_action_context()
            self._identity_context = key
        face = self.face_model.analyze(frame)
        pose = self.pose_model.recognize(frame)
        match = self.gallery.match_expected_seat(face, context.focus_seat)
        observation = self.identity_temporal.process(
            match,
            face.observed_at_ns,
            FaceIdentityContext(
                context.session_id,
                context.state_version,
                context.focus_seat,
            ),
        )
        self.frame_source.set_action_marker(None)
        self._publish_runtime_face(
            face,
            status=self._identity_ui_status(
                observation.identity_state,
                context.focus_seat,
            ),
        )
        if observation.identity_state is FaceIdentityState.MATCHED:
            if len(face.features) != 1:
                self._publish_runtime_face(face, status="ONE PLAYER ONLY")
                return replace(
                    observation,
                    identity_state=FaceIdentityState.LOW_QUALITY,
                    player_id=None,
                    registered_seat=None,
                    quality_flags=(*observation.quality_flags, "one_face_required_for_actor_binding"),
                )
            track = self.person_tracker.acquire(
                pose.poses,
                face_bbox_xywh=face.features[0].bbox_xywh,
                frame_width=frame.width,
                frame_height=frame.height,
                observed_at_ns=face.observed_at_ns,
            )
            if track is None:
                self._publish_runtime_face(
                    face,
                    status="FACE FOUND · HOLD STILL",
                )
                return replace(
                    observation,
                    identity_state=FaceIdentityState.LOW_QUALITY,
                    player_id=None,
                    registered_seat=None,
                    quality_flags=(*observation.quality_flags, "target_pose_not_acquired"),
                )
            self.actor_lease.open(
                observation,
                hand_id=context.hand_id,
                person_track_id=track.track_id,
                camera_epoch=context.camera_epoch,
            )
            self._publish_runtime_face(
                face,
                status=(
                    f"{self._runtime_seat_label(context.focus_seat)} "
                    "VERIFIED · LISTENING"
                ),
            )
        self.frame_source.set_status(
            f"IDENTITY: expected {context.focus_seat.value}",
            f"state={observation.identity_state.value}",
        )
        return observation

    def observe_action(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        observed_at_ns: int,
    ) -> ActionEvidence | None:
        if frame is None or context.focus_seat is None:
            return None
        if any(
            item is None
            for item in (
                self.gallery,
                self.face_model,
                self.gesture_model,
                self.pose_model,
            )
        ):
            raise RuntimeError("live perception session is not open")
        assert self.gallery is not None
        assert self.face_model is not None
        assert self.gesture_model is not None
        assert self.pose_model is not None
        face = self.face_model.analyze(frame)
        pose = self.pose_model.recognize(frame)
        guard_match = self.gallery.match_expected_seat(face, context.focus_seat)
        guard = self.identity_temporal.process(
            guard_match,
            face.observed_at_ns,
            FaceIdentityContext(
                context.session_id,
                context.state_version,
                context.focus_seat,
            ),
        )
        self.actor_lease.observe_identity(guard)
        track = self.person_tracker.update(
            pose.poses, observed_at_ns=pose.observed_at_ns
        )
        binding = self.actor_lease.binding
        if (
            binding is None
            or track is None
            or not self.actor_lease.is_valid_at(
                face.observed_at_ns, camera_epoch=context.camera_epoch
            )
        ):
            self.frame_source.set_action_marker(None)
            self._publish_runtime_face(
                face,
                status=(
                    "IDENTITY LOST · VERIFYING "
                    f"{self._runtime_seat_label(context.focus_seat)}"
                ),
            )
            observation = self._unknown_action(
                context, observed_at_ns, self.actor_lease.last_reason
            )
            return ActionEvidence(
                observation,
                identity_revocation_reason=self.actor_lease.last_reason,
            )
        self._publish_runtime_face(
            face,
            status=(
                f"{self._runtime_seat_label(context.focus_seat)} "
                "VERIFIED · LISTENING"
            ),
        )
        raw_hands = self.gesture_model.recognize_all(frame)
        attributed = attribute_hands_to_target(
            raw_hands,
            pose.poses,
            target_pose_detector_index=track.pose.detector_index,
            config=self.attribution_config,
        )
        selected_hand = attributed.selected_hand
        if (
            selected_hand is not None
            and selected_hand.gesture_label in self.gesture_config.gesture_to_action
            and selected_hand.gesture_score is not None
            and selected_hand.gesture_score
            >= self.gesture_config.confirmation.minimum_score
            and selected_hand.wrist_x is not None
            and selected_hand.wrist_y is not None
        ):
            self.frame_source.set_action_marker(
                (
                    round(min(1.0, max(0.0, selected_hand.wrist_x)), 3),
                    round(min(1.0, max(0.0, selected_hand.wrist_y)), 3),
                    self.gesture_config.gesture_to_action[
                        selected_hand.gesture_label
                    ].value,
                    round(selected_hand.gesture_score, 3),
                )
            )
        else:
            self.frame_source.set_action_marker(None)
        gesture_observation = self.gesture_temporal.process(
            attributed.temporal_evidence(frame.captured_at_ns),
            ActionObservationContext(
                context.hand_id,
                context.state_version,
                context.focus_seat,
            ),
        )
        fused = self.multimodal.add(gesture_observation)
        recognized_speech_action: PlayerActionType | None = None
        if self._speech_recognizer is not None:
            if self.speaker_gallery is None:
                raise RuntimeError("speaker gallery is unavailable")
            if not self._audio_input_is_healthy(observed_at_ns):
                return None
            if self._speech_input_suppressed(observed_at_ns):
                return None
            while True:
                try:
                    pcm = self._audio_queue.get_nowait()
                except queue.Empty:
                    break
                speech = self._speech_recognizer.accept_audio(pcm, observed_at_ns)
                if speech is None:
                    continue
                intent = classify_speech_intent(speech, self.speech_config)
                speaker = (
                    self.speaker_gallery.match(
                        speech.speaker_embedding,
                        speaker_frames=speech.speaker_frames,
                    )
                    if speech.speaker_embedding is not None
                    else None
                )
                speaker_matches_expected = bool(
                    speaker is not None
                    and speaker.state is SpeakerVerificationState.MATCHED
                    and speaker.player_id == binding.player_id
                )
                if speaker is None:
                    speaker_state = "embedding_missing"
                else:
                    speaker_state = speaker.state.value
                if intent.kind is SpeechIntentKind.UNKNOWN:
                    if speech.confidence is None:
                        rejection_reason = "speech_confidence_missing"
                    elif speech.confidence < self.speech_config.minimum_confidence:
                        rejection_reason = "speech_confidence_below_threshold"
                    else:
                        rejection_reason = "speech_command_not_understood"
                else:
                    rejection_reason = None
                self._emit_audio_event(
                    "speech_recognition_result",
                    observed_at_ns=speech.observed_at_ns,
                    payload={
                        "seat": context.focus_seat.value,
                        "expected_state_version": context.state_version,
                        "command": speech.canonical_transcript[:32],
                        "confidence": (
                            round(speech.confidence, 6)
                            if speech.confidence is not None
                            else None
                        ),
                        "intent": intent.kind.value,
                        "candidate_action": (
                            intent.action.value if intent.action is not None else None
                        ),
                        "speaker_state": speaker_state,
                        "speaker_matches_expected_player": speaker_matches_expected,
                        "speaker_similarity": (
                            round(speaker.similarity, 6)
                            if speaker is not None
                            and speaker.similarity is not None
                            else None
                        ),
                        "second_best_speaker_similarity": (
                            round(speaker.second_best_similarity, 6)
                            if speaker is not None
                            and speaker.second_best_similarity is not None
                            else None
                        ),
                        "speaker_frames": speech.speaker_frames,
                        "accepted_for_action": intent.kind is SpeechIntentKind.ACTION,
                        "confirmation_required": False,
                        "rejection_reason": rejection_reason,
                        "speaker_verification_enforced": False,
                        "speaker_verification_advisory_only": True,
                        "audio_saved": False,
                        "embedding_logged": False,
                    },
                )
                if intent.kind is SpeechIntentKind.UNKNOWN:
                    if speech.is_final:
                        self._emit_speech_feedback(
                            "speech_command_not_understood",
                            observed_at_ns=speech.observed_at_ns,
                        )
                    continue
                if intent.kind is SpeechIntentKind.ACTION:
                    self._accepted_speech_confidence = speech.confidence
                    self._accepted_speech_player_id = binding.player_id
                    speech_observation = self._speech_adapter.process(
                        speech,
                        ActionObservationContext(
                            context.hand_id,
                            context.state_version,
                            context.focus_seat,
                        ),
                    )
                    recognized_speech_action = intent.action
                    self._emit_speech_feedback(
                        "speech_action_recognized",
                        observed_at_ns=speech.observed_at_ns,
                        payload={"action": intent.action.value},
                        cooldown_ms=0,
                    )
                    candidate = self.multimodal.add(speech_observation)
                    if candidate is not None:
                        fused = candidate
                elif intent.kind is SpeechIntentKind.CANCEL:
                    self.multimodal.cancel_pending_speech()
                    self._accepted_speech_confidence = None
                    self._accepted_speech_player_id = None
                    self._emit_speech_feedback(
                        "speech_action_cancelled",
                        observed_at_ns=speech.observed_at_ns,
                        cooldown_ms=0,
                    )
        if self._consume_runtime_intent(
            ControlIntent.CANCEL, ControlIntent.CLEAR
        ):
            self.multimodal.cancel_pending_speech()
            self._accepted_speech_confidence = None
            self._accepted_speech_player_id = None
        if fused is None:
            fused = self.multimodal.poll(observed_at_ns)
        self._publish_runtime_face(
            face,
            status=(
                (
                    f"{recognized_speech_action.value.upper()} HEARD · SUBMITTING"
                )
                if recognized_speech_action is not None
                else (
                    f"{self._runtime_seat_label(context.focus_seat)} "
                    "VERIFIED · LISTENING"
                )
            ),
        )
        self.frame_source.set_status(
            f"ACTION: {context.focus_seat.value}",
            "gesture or clear English voice",
            f"legal: {','.join(action.value for action in context.legal_actions)}",
        )
        if fused is None or fused.evidence_state is not ActionEvidenceState.CANDIDATE:
            return None
        fusion_flag = next(
            (
                flag
                for flag in fused.quality_flags
                if flag.startswith("fusion_sources:")
            ),
            "fusion_sources:gesture",
        )
        has_speech = "speech" in fusion_flag
        has_gesture = "gesture" in fusion_flag
        if has_speech and self._accepted_speech_player_id != binding.player_id:
            return None
        if has_speech and has_gesture:
            values = (
                attributed.attribution_confidence,
                self._accepted_speech_confidence,
            )
            attribution_confidence = (
                min(value for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            )
        elif has_speech:
            attribution_confidence = self._accepted_speech_confidence
        else:
            attribution_confidence = attributed.attribution_confidence
        if attribution_confidence is None:
            return None
        return ActionEvidence(
            fused,
            binding,
            (
                "face_bound_speech_recognition"
                if has_speech and not has_gesture
                else "face_pose_wrist_multimodal"
                if has_speech
                else "face_pose_wrist"
            ),
            attribution_confidence,
            (
                fusion_flag,
                *(("speaker_verification_advisory_only",) if has_speech else ()),
            ),
        )

    def observe_card(
        self,
        frame: FramePacket | None,
        context: RuntimeObservationContext,
        slot: VisionSlot,
        observed_at_ns: int,
    ) -> CardObservation | None:
        if frame is None or self.card_model is None:
            return None
        if context.hand_phase is HandPhase.DEALING_HOLE:
            raise RuntimeError(
                "hole-card slots are completed by a successful dispense ACK, "
                "not by live card perception"
            )
        full_frame = self.card_geometry.binding_mode == "state_directed_full_frame"
        self.frame_source.set_status(
            f"FACE-UP CARD: {slot.value}",
            (
                "show only the current target card; YOLO scans the full frame"
                if full_frame
                else "place/reveal one card inside the active fixed ROI"
            ),
        )
        inference_frame = frame
        if not full_frame:
            inference_frame, _pixel_roi = crop_fixed_card_roi(
                frame, self.card_geometry.roi_for(slot), slot
            )
        evidence = self.card_model.analyze(inference_frame)
        return self.card_temporal.process(slot, evidence)

    def _reset_action_context(self) -> None:
        self.identity_temporal = FaceIdentityTemporalAdapter(self.identity_config)
        self.gesture_temporal = GestureTemporalAdapter(self.gesture_config)
        self._speech_adapter = SpeechObservationAdapter(self.speech_config)
        self.actor_lease.clear()
        self.person_tracker.clear()
        self.multimodal.clear()
        self._accepted_speech_confidence = None
        self._accepted_speech_player_id = None
        self.frame_source.set_action_marker(None)
        if self._speech_recognizer is not None:
            self._speech_recognizer.reset_window()
        self._discard_audio_queue()

    def _discard_audio_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _speech_input_suppressed(self, observed_at_ns: int) -> bool:
        suppressed = (
            self._speech_playback_gate is not None
            and self._speech_playback_gate.is_suppressed(observed_at_ns)
        )
        if suppressed:
            self._discard_audio_queue()
            if not self._speech_was_suppressed and self._speech_recognizer is not None:
                self._speech_recognizer.reset_window()
            self._speech_was_suppressed = True
            return True
        if self._speech_was_suppressed:
            self._discard_audio_queue()
            if self._speech_recognizer is not None:
                self._speech_recognizer.reset_window()
            self._speech_was_suppressed = False
        return False

    @staticmethod
    def _unknown_action(
        context: RuntimeObservationContext,
        observed_at_ns: int,
        reason: str,
    ) -> PlayerActionObservation:
        assert context.focus_seat is not None
        return PlayerActionObservation(
            observation_id=f"live-action-revoked:{context.state_version}:{observed_at_ns}",
            hand_id=context.hand_id,
            expected_state_version=context.state_version,
            window_started_at_ns=observed_at_ns,
            observed_at_ns=observed_at_ns,
            focus_seat=context.focus_seat,
            evidence_state=ActionEvidenceState.UNKNOWN,
            candidate_action=None,
            confidence=None,
            stable_duration_ms=0,
            stable_frames=1,
            model_version="unified-live-action-guard@1",
            calibration_version="identity-lease-v1",
            quality_flags=(reason,),
        )

    def _unknown_card(
        self, slot: VisionSlot, observed_at_ns: int, reason: str
    ) -> CardObservation:
        return CardObservation(
            observation_id=f"live-card-unknown:{slot.value}:{observed_at_ns}",
            slot_id=slot,
            observed_at_ns=observed_at_ns,
            status=ObservationStatus.UNKNOWN,
            card=None,
            confidence=None,
            model_version=self.card_config.model_version,
            calibration_version=self.card_config.calibration_version,
            stable_frames=1,
            quality_flags=(reason,),
        )


__all__ = [
    "InteractiveOpenCVFrameSource",
    "LiveKeyboardControlSource",
    "LivePerceptionConfig",
    "LivePerceptionSession",
    "RegistrationUiState",
    "validate_live_perception_assets",
]
