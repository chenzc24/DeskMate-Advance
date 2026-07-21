"""Offline English command evidence for the focused-seat speech pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from poker_dealer.domain import (
    ActionEvidenceState,
    PlayerActionObservation,
    PlayerActionType,
)

from .temporal import ActionObservationContext

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
except ImportError:  # pragma: no cover - exercised through the runtime error
    KaldiRecognizer = Model = None  # type: ignore[assignment,misc]
    SetLogLevel = None  # type: ignore[assignment]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SpeechModelConfig:
    model_id: str
    version: str
    asset_path: Path
    tree_sha256: str
    framework: str
    framework_version: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.version.strip():
            raise ValueError("speech model ID and version are required")
        if len(self.tree_sha256) != 64:
            raise ValueError("speech model tree SHA-256 must have 64 digits")
        int(self.tree_sha256, 16)


@dataclass(frozen=True, slots=True)
class SpeechPilotConfig:
    schema_version: str
    pilot_status: str
    model: SpeechModelConfig
    audio: Mapping[str, int | str]
    command_to_action: Mapping[str, PlayerActionType]
    control_commands: frozenset[str]
    recognition_grammar: Mapping[str, str]
    minimum_confidence: float
    candidate_cooldown_ms: int
    calibration_version: str
    seat_attribution: str
    save_audio: bool
    max_seconds_default: int

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported speech pilot schema version")
        if self.pilot_status != "development_feasibility_only":
            raise ValueError("speech pilot must remain development-only")
        if set(self.command_to_action.values()) != set(PlayerActionType):
            raise ValueError("speech commands must cover all five poker actions")
        vocabulary = set(self.command_to_action) | set(self.control_commands)
        if set(self.recognition_grammar) != vocabulary:
            raise ValueError("recognition grammar must cover actions and controls")
        if set(self.command_to_action) & self.control_commands:
            raise ValueError("action and control commands must be disjoint")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum speech confidence must be in [0, 1]")
        if self.candidate_cooldown_ms < 0:
            raise ValueError("speech cooldown must be non-negative")
        if self.seat_attribution != "state_owned_listening_window_only":
            raise ValueError("speech cannot claim biometric seat attribution")
        if self.save_audio:
            raise ValueError("the speech pilot must not save microphone audio")
        if self.max_seconds_default <= 0:
            raise ValueError("speech pilot duration must be positive")
        if int(self.audio["sample_rate_hz"]) != 16000:
            raise ValueError("the Vosk pilot requires 16 kHz PCM")
        if int(self.audio["channels"]) != 1 or self.audio["dtype"] != "int16":
            raise ValueError("the Vosk pilot requires mono int16 PCM")

    @classmethod
    def from_json(cls, path: str | Path) -> SpeechPilotConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parents[2]
        model = value["model"]
        confirmation = value["confirmation"]
        runtime = value["runtime"]
        return cls(
            schema_version=value["schema_version"],
            pilot_status=value["pilot_status"],
            model=SpeechModelConfig(
                model_id=model["model_id"],
                version=model["version"],
                asset_path=project_root / model["asset_path"],
                tree_sha256=model["tree_sha256"].lower(),
                framework=model["framework"],
                framework_version=model["framework_version"],
            ),
            audio=dict(value["audio"]),
            command_to_action={
                command: PlayerActionType(action)
                for command, action in value["command_to_action"].items()
            },
            control_commands=frozenset(value["control_commands"]),
            recognition_grammar=dict(value["recognition_grammar"]),
            minimum_confidence=float(confirmation["minimum_confidence"]),
            candidate_cooldown_ms=int(confirmation["candidate_cooldown_ms"]),
            calibration_version=runtime["calibration_version"],
            seat_attribution=runtime["seat_attribution"],
            save_audio=runtime["save_audio"],
            max_seconds_default=int(runtime["max_seconds_default"]),
        )

    def verify_model_asset(self) -> str:
        if not self.model.asset_path.is_dir():
            raise FileNotFoundError(
                f"speech model asset is missing: {self.model.asset_path}"
            )
        actual = _tree_sha256(self.model.asset_path)
        if actual != self.model.tree_sha256:
            raise ValueError(
                "speech model tree SHA-256 mismatch: "
                f"expected {self.model.tree_sha256}, got {actual}"
            )
        return actual

    def grammar_json(self) -> str:
        phrases = list(self.recognition_grammar.values()) + ["[unk]"]
        return json.dumps(phrases, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class SpeechUtteranceEvidence:
    window_started_at_ns: int
    observed_at_ns: int
    transcript: str
    confidence: float | None
    is_final: bool
    supporting_blocks: int = 1

    def __post_init__(self) -> None:
        if self.window_started_at_ns < 0 or self.observed_at_ns < 0:
            raise ValueError("speech timestamps must be non-negative")
        if self.observed_at_ns < self.window_started_at_ns:
            raise ValueError("speech observation cannot precede its window")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speech confidence must be in [0, 1]")
        if self.supporting_blocks <= 0:
            raise ValueError("supporting audio blocks must be positive")

    @property
    def canonical_transcript(self) -> str:
        return "".join(self.transcript.split())


class SpeechModelError(RuntimeError):
    """Raised when the offline speech model cannot load or decode audio."""


class VoskSpeechRecognizer:
    """Bounded streaming decoder that never stores PCM or free-form text."""

    def __init__(self, config: SpeechPilotConfig) -> None:
        if Model is None or KaldiRecognizer is None:
            raise SpeechModelError(
                "Vosk is unavailable; install the speech-pilot dependency"
            )
        config.verify_model_asset()
        if SetLogLevel is not None:
            SetLogLevel(-1)
        try:
            self._model = Model(str(config.model.asset_path))
            self._recognizer = KaldiRecognizer(
                self._model,
                int(config.audio["sample_rate_hz"]),
                config.grammar_json(),
            )
            self._recognizer.SetWords(True)
        except (RuntimeError, ValueError) as exc:
            raise SpeechModelError(f"failed to load Vosk speech model: {exc}") from exc
        self._window_started_at_ns: int | None = None
        self._window_blocks = 0

    def accept_audio(
        self, pcm_bytes: bytes, observed_at_ns: int
    ) -> SpeechUtteranceEvidence | None:
        if observed_at_ns < 0:
            raise ValueError("audio timestamp must be non-negative")
        if len(pcm_bytes) % 2:
            raise ValueError("int16 PCM byte count must be even")
        if self._window_started_at_ns is None:
            self._window_started_at_ns = observed_at_ns
        self._window_blocks += 1
        try:
            accepted = self._recognizer.AcceptWaveform(pcm_bytes)
        except (RuntimeError, ValueError) as exc:
            raise SpeechModelError(f"Vosk audio decoding failed: {exc}") from exc
        if not accepted:
            return None
        return self._consume_result(self._recognizer.Result(), observed_at_ns)

    def partial_text(self) -> str:
        payload = json.loads(self._recognizer.PartialResult())
        return str(payload.get("partial", ""))

    def flush(self, observed_at_ns: int) -> SpeechUtteranceEvidence | None:
        if self._window_started_at_ns is None:
            return None
        return self._consume_result(self._recognizer.FinalResult(), observed_at_ns)

    def reset_window(self) -> None:
        """Discard decoder history when state opens a new listening window."""

        try:
            self._recognizer.Reset()
        except (RuntimeError, ValueError) as exc:
            raise SpeechModelError(f"failed to reset Vosk listening window: {exc}") from exc
        self._window_started_at_ns = None
        self._window_blocks = 0

    def _consume_result(
        self, raw_result: str, observed_at_ns: int
    ) -> SpeechUtteranceEvidence | None:
        payload: dict[str, Any] = json.loads(raw_result)
        transcript = str(payload.get("text", ""))
        words = payload.get("result", [])
        confidences = [
            float(word["conf"])
            for word in words
            if isinstance(word, dict) and "conf" in word
        ]
        confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )
        started_at_ns = self._window_started_at_ns
        supporting_blocks = self._window_blocks
        self._window_started_at_ns = None
        self._window_blocks = 0
        if not transcript.strip():
            return None
        assert started_at_ns is not None
        return SpeechUtteranceEvidence(
            window_started_at_ns=started_at_ns,
            observed_at_ns=observed_at_ns,
            transcript=transcript,
            confidence=confidence,
            is_final=True,
            supporting_blocks=supporting_blocks,
        )


class SpeechObservationAdapter:
    """Map final closed-vocabulary utterances into model-neutral evidence."""

    def __init__(self, config: SpeechPilotConfig) -> None:
        self.config = config
        self._last_timestamp_ns: int | None = None
        self._last_candidate_at_ns: int | None = None
        self._last_candidate_action: PlayerActionType | None = None
        self._sequence = 0

    def process(
        self,
        evidence: SpeechUtteranceEvidence,
        context: ActionObservationContext,
    ) -> PlayerActionObservation:
        if (
            self._last_timestamp_ns is not None
            and evidence.observed_at_ns < self._last_timestamp_ns
        ):
            raise ValueError("speech evidence timestamps must be monotonic")
        self._last_timestamp_ns = evidence.observed_at_ns
        self._sequence += 1
        transcript = evidence.canonical_transcript
        state = ActionEvidenceState.UNKNOWN
        candidate_action: PlayerActionType | None = None
        flags = ["evidence_source:speech"]

        if not evidence.is_final:
            state = ActionEvidenceState.ACTION_START
            flags.append("speech_partial")
        elif not transcript:
            state = ActionEvidenceState.NO_ACTION
            flags.append("speech_empty")
        elif transcript in self.config.control_commands:
            flags.append(f"speech_control:{transcript}")
        elif transcript not in self.config.command_to_action:
            flags.append("speech_outside_closed_vocabulary")
        elif (
            evidence.confidence is None
            or evidence.confidence < self.config.minimum_confidence
        ):
            flags.append("speech_below_confidence_threshold")
        else:
            action = self.config.command_to_action[transcript]
            in_cooldown = (
                self._last_candidate_at_ns is not None
                and self._last_candidate_action is action
                and evidence.observed_at_ns - self._last_candidate_at_ns
                < self.config.candidate_cooldown_ms * 1_000_000
            )
            if in_cooldown:
                state = ActionEvidenceState.NO_ACTION
                flags.append("speech_candidate_cooldown_active")
            else:
                state = ActionEvidenceState.CANDIDATE
                candidate_action = action
                self._last_candidate_at_ns = evidence.observed_at_ns
                self._last_candidate_action = action
                flags.append(f"speech_command:{transcript}")

        duration_ms = max(
            0,
            (evidence.observed_at_ns - evidence.window_started_at_ns)
            // 1_000_000,
        )
        return PlayerActionObservation(
            observation_id=(
                f"speech-pilot:{context.hand_id}:{self._sequence}:"
                f"{evidence.observed_at_ns}"
            ),
            hand_id=context.hand_id,
            expected_state_version=context.expected_state_version,
            window_started_at_ns=evidence.window_started_at_ns,
            observed_at_ns=evidence.observed_at_ns,
            focus_seat=context.focus_seat,
            evidence_state=state,
            candidate_action=candidate_action,
            confidence=evidence.confidence,
            stable_duration_ms=duration_ms,
            stable_frames=evidence.supporting_blocks,
            model_version=f"{self.config.model.model_id}@{self.config.model.version}",
            calibration_version=self.config.calibration_version,
            quality_flags=tuple(flags),
        )
