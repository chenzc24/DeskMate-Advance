"""Shared-frame live adapters composed from the existing Stage 2 pilots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import queue
import time
from typing import Mapping

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
    SpeechConfirmationController,
    SpeechConfirmationStatus,
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
    FaceIdentityConfig,
    FaceIdentityContext,
    FaceIdentityObservation,
    FaceIdentityState,
    FaceIdentityTemporalAdapter,
    OpenCvFaceIdentityAdapter,
    SessionFaceGallery,
)

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
    runtime_calibration_id: str
    target_geometry_validated: bool = False
    operator_face_down_confirmation: bool = True

    def __post_init__(self) -> None:
        if not self.runtime_calibration_id.strip():
            raise ValueError("runtime calibration ID is required")


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
        "card_slot_count": len(geometry.slots),
        "card_geometry_calibration_id": geometry.calibration_id,
        "card_target_geometry_validated": geometry.target_geometry_validated,
        "speech_hash": speech_hash,
        "speaker_hash": speaker_hash,
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
    ) -> None:
        self.camera = camera
        self.display = display
        self.window_name = window_name
        self.camera_epoch = 0
        self._last_key: int | None = None
        self._quit = False
        self._status_lines: tuple[str, ...] = ()

    def open(self) -> None:
        if not self.camera.is_open:
            self.camera.open()

    def set_status(self, *lines: str) -> None:
        self._status_lines = tuple(line for line in lines if line)

    def read(self) -> FrameRead:
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
        if state is FrameReadState.OK and read.frame is not None and self.display:
            if cv2 is None:
                raise RuntimeError("OpenCV UI is unavailable")
            display = read.frame.image.copy()
            for index, line in enumerate(self._status_lines):
                cv2.putText(
                    display,
                    line,
                    (18, 30 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            self._last_key = None if key == 255 else key
            if self._last_key in (ord("q"), 27):
                self._quit = True
        return FrameRead(
            state,
            read.observed_at_ns,
            read.frame,
            self.camera_epoch,
            read.reason,
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
        if self.display and cv2 is not None:
            cv2.destroyWindow(self.window_name)


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
            allow_speech_single_source=False,
        )
        self.visual_settle = VisualSettleGate()
        self._visual_started = False
        self._identity_context: tuple[int, Seat] | None = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=int(self.speech_config.audio["queue_max_blocks"])
        )
        self._audio_stream = None
        self._speech_recognizer: VoskSpeechRecognizer | None = None
        self._speech_adapter = SpeechObservationAdapter(self.speech_config)
        self._speech_confirmation = SpeechConfirmationController(
            confirmation_timeout_ms=self.speaker_config.confirmation_timeout_ms,
            require_speaker_match=True,
        )
        self._verified_speech_similarity: float | None = None
        self._verified_speech_player_id: str | None = None

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

            def callback(indata: bytes, _frames: int, _time, _status) -> None:
                try:
                    self._audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass

            self._audio_stream = sd.RawInputStream(
                samplerate=int(self.speech_config.audio["sample_rate_hz"]),
                blocksize=int(self.speech_config.audio["blocksize_frames"]),
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
        self._speech_confirmation.clear()

    def acquire_roster(
        self,
        *,
        frame_source: FrameSource,
        control_source: ControlSource,
        event_sink: RuntimeEventSink,
        session_id: str,
        button: Seat,
        deadline_ns: int,
    ) -> FrozenSessionRoster:
        if (
            self.gallery is None
            or self.face_model is None
            or self.speaker_gallery is None
        ):
            raise RuntimeError("live perception session is not open")
        registration = RegistrationRuntime(session_id, button)
        samples = []
        last_sample_ns: int | None = None
        voice_player_id: str | None = None
        voice_seat: Seat | None = None
        voice_samples = []

        def advance_registration_role() -> None:
            remaining = [
                role
                for role in ROLE_ORDER
                if role_seats(button)[role] not in registration.registered_seats
            ]
            if remaining:
                registration.select_role(remaining[0])

        self.frame_source.set_status(
            "REGISTER: E/Enter capture current role | S start | X clear | Q quit",
            f"Current role: {registration.focus_role.value}",
        )
        while time.monotonic_ns() < deadline_ns:
            read = frame_source.read()
            if read.state is FrameReadState.DISCONNECTED:
                raise RuntimeError(read.reason or "camera disconnected during registration")
            if read.frame is None:
                continue
            frame = read.frame
            for control in control_source.poll_controls(read.observed_at_ns):
                if voice_player_id is not None and control.intent is ControlIntent.START:
                    event_sink.emit(
                        "registration_control",
                        observed_at_ns=control.observed_at_ns,
                        payload={
                            "intent": control.intent.value,
                            "accepted": False,
                            "reason": "voice_enrollment_pending",
                            "focus_role": registration.focus_role.value,
                        },
                    )
                    continue
                outcome = registration.accept_control(control)
                event_sink.emit(
                    "registration_control",
                    observed_at_ns=control.observed_at_ns,
                    payload={
                        "intent": control.intent.value,
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "focus_role": registration.focus_role.value,
                    },
                )
                if control.intent is ControlIntent.CLEAR and outcome.accepted:
                    self.gallery.clear()
                    self.speaker_gallery.clear()
                    samples = []
                    voice_player_id = None
                    voice_seat = None
                    for sample in voice_samples:
                        sample.fill(0.0)
                    voice_samples = []
                    last_sample_ns = None
                if control.intent is ControlIntent.CANCEL and outcome.accepted:
                    for sample in samples:
                        embedding = getattr(sample, "embedding", None)
                        if embedding is not None:
                            embedding.fill(0.0)
                    samples = []
                    last_sample_ns = None
                if outcome.roster is not None:
                    return outcome.roster
            if (
                registration.phase is RegistrationPhase.CAPTURING_FACE
                and voice_player_id is None
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
                if len(samples) >= self.identity_config.minimum_samples:
                    player_id = registration.participant_id
                    seat = registration.focus_seat
                    self.gallery.enroll(
                        player_id,
                        seat,
                        samples,
                        consent_granted=self.config.consent_confirmed,
                    )
                    participant = registration.complete_face_enrollment(len(samples))
                    event_sink.emit(
                        "registration_enrolled",
                        observed_at_ns=evidence.observed_at_ns,
                        payload={
                            "player_id": participant.participant_id,
                            "seat": participant.seat.value,
                            "role": participant.initial_role.value,
                            "sample_count": participant.face_sample_count,
                        },
                    )
                    samples = []
                    last_sample_ns = None
                    if self.config.speech_enabled:
                        if self._speech_recognizer is None:
                            raise RuntimeError("speech recognizer is unavailable")
                        voice_player_id = participant.participant_id
                        voice_seat = participant.seat
                        voice_samples = []
                        self._speech_recognizer.reset_window()
                        while not self._audio_queue.empty():
                            try:
                                self._audio_queue.get_nowait()
                            except queue.Empty:
                                break
                    else:
                        advance_registration_role()
            if voice_player_id is not None:
                assert voice_seat is not None and self._speech_recognizer is not None
                while len(voice_samples) < self.speaker_config.minimum_samples:
                    try:
                        pcm = self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    voice = self._speech_recognizer.accept_audio(
                        pcm, time.monotonic_ns()
                    )
                    if (
                        voice is None
                        or voice.speaker_embedding is None
                        or voice.speaker_frames
                        < self.speaker_config.minimum_speaker_frames
                    ):
                        continue
                    voice_samples.append(voice.speaker_embedding.copy())
                if len(voice_samples) >= self.speaker_config.minimum_samples:
                    self.speaker_gallery.enroll(voice_player_id, voice_samples)
                    registration.mark_voice_enrolled(voice_seat)
                    event_sink.emit(
                        "speaker_enrollment_completed",
                        observed_at_ns=time.monotonic_ns(),
                        payload={
                            "player_id": voice_player_id,
                            "seat": voice_seat.value,
                            "sample_count": len(voice_samples),
                            "embeddings_logged": False,
                            "audio_saved": False,
                        },
                    )
                    for sample in voice_samples:
                        sample.fill(0.0)
                    voice_samples = []
                    voice_player_id = None
                    voice_seat = None
                    self._speech_recognizer.reset_window()
                    advance_registration_role()
            self.frame_source.set_status(
                "REGISTER: E/Enter capture current role | S start | X clear | Q quit",
                f"Current role: {registration.focus_role.value}",
                f"Faces: {len(registration.registered_seats)}/4 | samples: {len(samples)}",
                (
                    f"VOICE: say a command {len(voice_samples)}/"
                    f"{self.speaker_config.minimum_samples}"
                    if voice_player_id is not None
                    else ""
                ),
            )
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
        if observation.identity_state is FaceIdentityState.MATCHED:
            if len(face.features) != 1:
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
            observation = self._unknown_action(
                context, observed_at_ns, self.actor_lease.last_reason
            )
            return ActionEvidence(
                observation,
                identity_revocation_reason=self.actor_lease.last_reason,
            )
        raw_hands = self.gesture_model.recognize_all(frame)
        attributed = attribute_hands_to_target(
            raw_hands,
            pose.poses,
            target_pose_detector_index=track.pose.detector_index,
            config=self.attribution_config,
        )
        gesture_observation = self.gesture_temporal.process(
            attributed.temporal_evidence(frame.captured_at_ns),
            ActionObservationContext(
                context.hand_id,
                context.state_version,
                context.focus_seat,
            ),
        )
        fused = self.multimodal.add(gesture_observation)
        speaker_confirmed = False
        if self._speech_confirmation.expire(observed_at_ns):
            self.multimodal.cancel_pending_speech()
            self._verified_speech_similarity = None
            self._verified_speech_player_id = None
        if self._speech_recognizer is not None:
            if self.speaker_gallery is None:
                raise RuntimeError("speaker gallery is unavailable")
            while True:
                try:
                    pcm = self._audio_queue.get_nowait()
                except queue.Empty:
                    break
                speech = self._speech_recognizer.accept_audio(pcm, observed_at_ns)
                if speech is None:
                    continue
                speaker = (
                    self.speaker_gallery.match(
                        speech.speaker_embedding,
                        speaker_frames=speech.speaker_frames,
                    )
                    if speech.speaker_embedding is not None
                    else None
                )
                if (
                    speaker is None
                    or speaker.state is not SpeakerVerificationState.MATCHED
                    or speaker.player_id != binding.player_id
                ):
                    continue
                self._verified_speech_similarity = speaker.similarity
                self._verified_speech_player_id = speaker.player_id
                speech_observation = self._speech_adapter.process(
                    speech,
                    ActionObservationContext(
                        context.hand_id,
                        context.state_version,
                        context.focus_seat,
                    ),
                )
                intent = classify_speech_intent(speech, self.speech_config)
                if intent.kind is SpeechIntentKind.ACTION:
                    pending = self._speech_confirmation.offer_action(
                        speech_observation,
                        binding,
                        speaker_player_id=speaker.player_id,
                    )
                    if pending.status is SpeechConfirmationStatus.PENDING:
                        candidate = self.multimodal.add(speech_observation)
                        if candidate is not None:
                            fused = candidate
                elif intent.kind in {
                    SpeechIntentKind.CONFIRM,
                    SpeechIntentKind.CANCEL,
                }:
                    confirmation = self._speech_confirmation.handle_control(
                        intent,
                        binding,
                        speaker_player_id=speaker.player_id,
                    )
                    if confirmation.status is SpeechConfirmationStatus.CONFIRMED:
                        fused = confirmation.observation
                        self.multimodal.cancel_pending_speech()
                        speaker_confirmed = True
                    elif confirmation.status is SpeechConfirmationStatus.CANCELLED:
                        self.multimodal.cancel_pending_speech()
        if self._consume_runtime_intent(ControlIntent.CONFIRM):
            pending = self._speech_confirmation.pending
            if (
                pending is not None
                and pending.player_id == binding.player_id
                and self._verified_speech_player_id == binding.player_id
            ):
                confirmed = self.multimodal.confirm_pending_speech(observed_at_ns)
                if confirmed is not None:
                    fused = replace(
                        confirmed,
                        quality_flags=tuple(
                            dict.fromkeys(
                                confirmed.quality_flags
                                + (
                                    "speaker_verified_same_actor",
                                    "speech_ui_confirmed_after_speaker_verification",
                                )
                            )
                        ),
                    )
                    self._speech_confirmation.clear()
                    speaker_confirmed = True
        if self._consume_runtime_intent(
            ControlIntent.CANCEL, ControlIntent.CLEAR
        ):
            self.multimodal.cancel_pending_speech()
            self._speech_confirmation.clear()
            self._verified_speech_similarity = None
            self._verified_speech_player_id = None
        if fused is None:
            fused = self.multimodal.poll(observed_at_ns)
        self.frame_source.set_status(
            f"ACTION: {context.focus_seat.value}",
            "gesture or English voice | C confirms speech | Backspace cancels",
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
        if has_speech and self._verified_speech_player_id != binding.player_id:
            return None
        if has_speech and has_gesture:
            values = (
                attributed.attribution_confidence,
                self._verified_speech_similarity,
            )
            attribution_confidence = (
                min(value for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            )
        elif has_speech:
            attribution_confidence = self._verified_speech_similarity
        else:
            attribution_confidence = attributed.attribution_confidence
        if attribution_confidence is None:
            return None
        if has_speech:
            self._speech_confirmation.clear()
        return ActionEvidence(
            fused,
            binding,
            (
                "session_speaker_verification_ui_confirm"
                if speaker_confirmed
                else (
                    "session_speaker_verification"
                    if has_speech and not has_gesture
                    else "face_pose_wrist_multimodal"
                    if has_speech
                    else "face_pose_wrist"
                )
            ),
            attribution_confidence,
            (
                fusion_flag,
                *(("speaker_verified_same_actor",) if has_speech else ()),
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
            self.frame_source.set_status(
                f"HOLE CARD: {slot.value}",
                "F confirms visible face-down occupancy (development operator fallback)",
            )
            if not self.config.operator_face_down_confirmation:
                return self._unknown_card(
                    slot, observed_at_ns, "face_down_orientation_adapter_unavailable"
                )
            if not self._consume_runtime_intent(ControlIntent.CONFIRM):
                return None
            return CardObservation(
                observation_id=f"live-hole-operator:{slot.value}:{observed_at_ns}",
                slot_id=slot,
                observed_at_ns=observed_at_ns,
                status=ObservationStatus.FACE_DOWN,
                card=None,
                confidence=None,
                model_version="operator-confirmed-hole-orientation@development",
                calibration_version=self.card_config.calibration_version,
                stable_frames=1,
                quality_flags=(
                    "operator_confirmed_face_down",
                    "not_gate_2b_model_evidence",
                ),
            )
        self.frame_source.set_status(
            f"FACE-UP CARD: {slot.value}",
            "place/reveal one card inside the active fixed ROI",
        )
        cropped, _pixel_roi = crop_fixed_card_roi(
            frame, self.card_geometry.roi_for(slot), slot
        )
        evidence = self.card_model.analyze(cropped)
        return self.card_temporal.process(slot, evidence)

    def _reset_action_context(self) -> None:
        self.identity_temporal = FaceIdentityTemporalAdapter(self.identity_config)
        self.gesture_temporal = GestureTemporalAdapter(self.gesture_config)
        self._speech_adapter = SpeechObservationAdapter(self.speech_config)
        self.actor_lease.clear()
        self.person_tracker.clear()
        self.multimodal.clear()
        self._speech_confirmation.clear()
        self._verified_speech_similarity = None
        self._verified_speech_player_id = None
        if self._speech_recognizer is not None:
            self._speech_recognizer.reset_window()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

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
    "validate_live_perception_assets",
]
