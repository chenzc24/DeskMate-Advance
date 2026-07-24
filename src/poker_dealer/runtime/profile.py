"""Validated dependency profiles for the single live runtime entry point."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Mapping

from poker_dealer.domain import ControlSource
from poker_dealer.io.camera import CameraConfig
from poker_dealer.runtime.network import NetworkEndpoints


class RuntimeProfileId(StrEnum):
    LAPTOP = "laptop"
    ROBOT_CAMERA = "robot_camera"
    ROBOT_HARDWARE = "robot_hardware"


class RuntimeCameraKind(StrEnum):
    LOCAL = "local"
    MJPEG = "mjpeg"


class DealerAdapterKind(StrEnum):
    SIMULATED = "simulated"
    REAL = "real"


@dataclass(frozen=True, slots=True)
class RuntimeCameraProfile:
    kind: RuntimeCameraKind
    source_id: str
    device_index: int = 0
    stream_url: str | None = None
    stream_endpoint: str | None = None
    backend: str = "auto"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 2000

    def __post_init__(self) -> None:
        if self.kind is RuntimeCameraKind.LOCAL and (
            self.stream_url is not None or self.stream_endpoint is not None
        ):
            raise ValueError("local camera profile cannot contain a network stream")
        if self.kind is RuntimeCameraKind.MJPEG and self.stream_url is None:
            raise ValueError("MJPEG camera profile requires stream_url")
        if self.stream_endpoint is not None and not self.stream_endpoint.strip():
            raise ValueError("camera stream endpoint must not be blank")
        self.to_camera_config()

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            device_index=self.device_index,
            stream_url=self.stream_url,
            source_id=self.source_id,
            backend=self.backend,
            width=self.width,
            height=self.height,
            fps=self.fps,
            open_timeout_ms=self.open_timeout_ms,
            read_timeout_ms=self.read_timeout_ms,
        )


@dataclass(frozen=True, slots=True)
class RuntimeDealerProfile:
    adapter: DealerAdapterKind
    device_id: str
    protocol_version: str
    physical_motion: bool
    enabled: bool
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.protocol_version.strip():
            raise ValueError("dealer device and protocol version are required")
        if self.adapter is DealerAdapterKind.SIMULATED and self.physical_motion:
            raise ValueError("simulated dealer cannot request physical motion")
        if self.adapter is DealerAdapterKind.REAL and not self.physical_motion:
            raise ValueError("real dealer profile must declare physical motion")
        if not self.enabled and not (self.unavailable_reason or "").strip():
            raise ValueError("disabled dealer profile requires unavailable_reason")


@dataclass(frozen=True, slots=True)
class RuntimePerceptionProfile:
    identity_config: Path
    gesture_config: Path
    speech_config: Path
    speaker_config: Path
    attribution_config: Path
    card_config: Path
    card_geometry_config: Path
    calibration_id: str
    target_geometry_validated: bool

    def __post_init__(self) -> None:
        for value in (
            self.identity_config,
            self.gesture_config,
            self.speech_config,
            self.speaker_config,
            self.attribution_config,
            self.card_config,
            self.card_geometry_config,
        ):
            if value.is_absolute() or not value.parts or ".." in value.parts:
                raise ValueError("perception config paths must be safe project-relative paths")
            if value.parts[:2] != ("configs", "perception"):
                raise ValueError("perception configs must stay under configs/perception/")
        if not self.calibration_id.strip():
            raise ValueError("perception calibration ID is required")

    def resolved(self, project_root: Path) -> Mapping[str, Path]:
        root = project_root.resolve()
        return {
            "identity_config": (root / self.identity_config).resolve(),
            "gesture_config": (root / self.gesture_config).resolve(),
            "speech_config": (root / self.speech_config).resolve(),
            "speaker_config": (root / self.speaker_config).resolve(),
            "attribution_config": (root / self.attribution_config).resolve(),
            "card_config": (root / self.card_config).resolve(),
            "card_geometry_config": (root / self.card_geometry_config).resolve(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    schema_version: str
    profile_id: RuntimeProfileId
    camera: RuntimeCameraProfile
    dealer: RuntimeDealerProfile
    perception: RuntimePerceptionProfile
    controls: tuple[ControlSource, ...]
    speech_enabled: bool
    speech_device: int | str | None
    speech_capture_sample_rate_hz: int | None
    log_root: Path

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported runtime profile schema version")
        if not self.controls:
            raise ValueError("at least one control source is required")
        if self.log_root.is_absolute() or not self.log_root.parts:
            raise ValueError("runtime log root must be a non-empty relative path")
        if self.log_root.parts[0].lower() != "runs":
            raise ValueError("runtime log root must stay under ignored runs/")
        if (
            self.speech_capture_sample_rate_hz is not None
            and self.speech_capture_sample_rate_hz <= 0
        ):
            raise ValueError("speech capture sample rate must be positive")
        if self.profile_id is RuntimeProfileId.LAPTOP:
            if self.camera.kind is not RuntimeCameraKind.LOCAL:
                raise ValueError("laptop profile requires a local camera")
            if self.dealer.adapter is not DealerAdapterKind.SIMULATED:
                raise ValueError("laptop profile requires the simulated dealer")
        elif self.profile_id is RuntimeProfileId.ROBOT_CAMERA:
            if self.camera.kind is not RuntimeCameraKind.MJPEG:
                raise ValueError("robot-camera profile requires an MJPEG stream")
            if self.dealer.adapter is not DealerAdapterKind.SIMULATED:
                raise ValueError("robot-camera profile requires the simulated dealer")
        elif self.profile_id is RuntimeProfileId.ROBOT_HARDWARE:
            if self.camera.kind is not RuntimeCameraKind.MJPEG:
                raise ValueError("robot-hardware profile requires an MJPEG stream")
            if self.dealer.adapter is not DealerAdapterKind.REAL:
                raise ValueError("robot-hardware profile requires a real dealer adapter")

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        network_endpoints_path: Path | None = None,
    ) -> RuntimeProfile:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("runtime profile root must be an object")
        camera = value.get("camera")
        endpoint_name = (
            camera.get("stream_endpoint")
            if isinstance(camera, Mapping)
            else None
        )
        endpoints = None
        if endpoint_name is not None:
            endpoint_path = network_endpoints_path or path.with_name(
                "network_endpoints.json"
            )
            endpoints = NetworkEndpoints.from_json(endpoint_path)
        return cls.from_mapping(value, network_endpoints=endpoints)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        network_endpoints: NetworkEndpoints | None = None,
    ) -> RuntimeProfile:
        cls._reject_unknown(
            value,
            {"schema_version", "profile_id", "camera", "dealer", "perception", "controls", "speech", "logging"},
            "runtime profile",
        )
        camera = cls._object(value, "camera")
        dealer = cls._object(value, "dealer")
        perception = cls._object(value, "perception")
        speech = cls._object(value, "speech")
        logging = cls._object(value, "logging")
        cls._reject_unknown(
            camera,
            {
                "kind", "source_id", "device_index", "stream_url",
                "stream_endpoint", "backend", "width", "height", "fps",
                "open_timeout_ms", "read_timeout_ms",
            },
            "camera",
        )
        cls._reject_unknown(
            dealer,
            {
                "adapter", "device_id", "protocol_version", "physical_motion",
                "enabled", "unavailable_reason",
            },
            "dealer",
        )
        cls._reject_unknown(
            perception,
            {
                "identity_config", "gesture_config", "speech_config", "speaker_config",
                "attribution_config", "card_config", "card_geometry_config", "calibration_id",
                "target_geometry_validated",
            },
            "perception",
        )
        cls._reject_unknown(
            speech,
            {"enabled", "device", "capture_sample_rate_hz"},
            "speech",
        )
        cls._reject_unknown(logging, {"root"}, "logging")
        controls = value.get("controls")
        if not isinstance(controls, list) or not all(
            isinstance(item, str) for item in controls
        ):
            raise ValueError("controls must be a list of control-source strings")
        stream_url_value = camera.get("stream_url")
        stream_endpoint_value = camera.get("stream_endpoint")
        if stream_url_value is not None and stream_endpoint_value is not None:
            raise ValueError(
                "camera stream_url and stream_endpoint are mutually exclusive"
            )
        stream_endpoint = (
            str(stream_endpoint_value)
            if stream_endpoint_value is not None
            else None
        )
        if stream_endpoint is not None:
            if network_endpoints is None:
                raise ValueError(
                    "camera stream_endpoint requires network endpoints"
                )
            stream_url = network_endpoints.camera_stream_url(stream_endpoint)
        else:
            stream_url = (
                str(stream_url_value)
                if stream_url_value is not None
                else None
            )
        return cls(
            schema_version=str(value.get("schema_version", "")),
            profile_id=RuntimeProfileId(str(value.get("profile_id", ""))),
            camera=RuntimeCameraProfile(
                kind=RuntimeCameraKind(str(camera.get("kind", ""))),
                source_id=str(camera.get("source_id", "")),
                device_index=int(camera.get("device_index", 0)),
                stream_url=stream_url,
                stream_endpoint=stream_endpoint,
                backend=str(camera.get("backend", "auto")),
                width=cls._optional_int(camera.get("width"), "camera.width"),
                height=cls._optional_int(camera.get("height"), "camera.height"),
                fps=cls._optional_float(camera.get("fps"), "camera.fps"),
                open_timeout_ms=int(camera.get("open_timeout_ms", 5000)),
                read_timeout_ms=int(camera.get("read_timeout_ms", 2000)),
            ),
            dealer=RuntimeDealerProfile(
                adapter=DealerAdapterKind(str(dealer.get("adapter", ""))),
                device_id=str(dealer.get("device_id", "")),
                protocol_version=str(dealer.get("protocol_version", "")),
                physical_motion=cls._bool(
                    dealer.get("physical_motion"), "dealer.physical_motion"
                ),
                enabled=cls._bool(dealer.get("enabled"), "dealer.enabled"),
                unavailable_reason=(
                    str(dealer["unavailable_reason"])
                    if dealer.get("unavailable_reason") is not None
                    else None
                ),
            ),
            perception=RuntimePerceptionProfile(
                identity_config=Path(str(perception.get("identity_config", ""))),
                gesture_config=Path(str(perception.get("gesture_config", ""))),
                speech_config=Path(str(perception.get("speech_config", ""))),
                speaker_config=Path(str(perception.get("speaker_config", ""))),
                attribution_config=Path(str(perception.get("attribution_config", ""))),
                card_config=Path(str(perception.get("card_config", ""))),
                card_geometry_config=Path(
                    str(perception.get("card_geometry_config", ""))
                ),
                calibration_id=str(perception.get("calibration_id", "")),
                target_geometry_validated=cls._bool(
                    perception.get("target_geometry_validated"),
                    "perception.target_geometry_validated",
                ),
            ),
            controls=tuple(ControlSource(str(item)) for item in controls),
            speech_enabled=cls._bool(speech.get("enabled"), "speech.enabled"),
            speech_device=cls._speech_device(speech.get("device")),
            speech_capture_sample_rate_hz=cls._optional_int(
                speech.get("capture_sample_rate_hz"),
                "speech.capture_sample_rate_hz",
            ),
            log_root=Path(str(logging.get("root", ""))),
        )

    def with_camera_override(
        self, *, device_index: int | None = None, stream_url: str | None = None
    ) -> RuntimeProfile:
        if device_index is not None and stream_url is not None:
            raise ValueError("camera index and stream URL overrides are exclusive")
        if device_index is not None:
            if self.profile_id is not RuntimeProfileId.LAPTOP:
                raise ValueError("camera index override is only valid for laptop")
            camera = replace(
                self.camera,
                kind=RuntimeCameraKind.LOCAL,
                device_index=device_index,
                stream_url=None,
                stream_endpoint=None,
            )
        elif stream_url is not None:
            if self.profile_id is RuntimeProfileId.LAPTOP:
                raise ValueError("stream URL override is not valid for laptop")
            camera = replace(
                self.camera,
                kind=RuntimeCameraKind.MJPEG,
                stream_url=stream_url,
                stream_endpoint=None,
            )
        else:
            return self
        return replace(self, camera=camera)

    def with_speech_override(
        self, *, enabled: bool, device: int | str | None = None
    ) -> RuntimeProfile:
        return replace(self, speech_enabled=enabled, speech_device=device)

    def resolved_log_root(self, project_root: Path) -> Path:
        root = project_root.resolve()
        resolved = (root / self.log_root).resolve()
        try:
            resolved.relative_to(root / "runs")
        except ValueError as exc:
            raise ValueError("runtime logs must resolve inside project runs/") from exc
        return resolved

    @staticmethod
    def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        item = value.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"{key} must be an object")
        return item

    @staticmethod
    def _reject_unknown(
        value: Mapping[str, Any], allowed: set[str], label: str
    ) -> None:
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown {label} fields: {sorted(unknown)}")

    @staticmethod
    def _optional_int(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer or null")
        return value

    @staticmethod
    def _optional_float(value: Any, label: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric or null")
        return float(value)

    @staticmethod
    def _bool(value: Any, label: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean")
        return value

    @staticmethod
    def _speech_device(value: Any) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("speech.device must be an integer, string or null")
        if isinstance(value, int) and value < 0:
            raise ValueError("speech.device index must be non-negative")
        if isinstance(value, str) and not value.strip():
            raise ValueError("speech.device name must not be blank")
        return value


__all__ = [
    "DealerAdapterKind",
    "RuntimeCameraKind",
    "RuntimeCameraProfile",
    "RuntimeDealerProfile",
    "RuntimeProfile",
    "RuntimeProfileId",
    "RuntimePerceptionProfile",
]
