"""Composition root shared by laptop and robot-camera live profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import time
from typing import Mapping

from poker_dealer.domain import DealerAck, DealerCommand, Seat
from poker_dealer.game import CoreGameConfig, FixedLimitRules
from poker_dealer.io.camera import CameraReadStatus, OpenCVCamera
from poker_dealer.robotics.dealer import (
    DealerPort,
    DealerUnavailableError,
    SimulatedDealerAdapter,
    UnavailableDealerAdapter,
)

from .hand_runtime import HandRuntime
from .profile import DealerAdapterKind, RuntimeCameraKind, RuntimeProfile
from .registration import FrozenSessionRoster
from .session_runtime import SessionRuntime
from .resource_lock import RuntimeResourceLocks


@dataclass(frozen=True, slots=True)
class RuntimePreflight:
    profile_id: str
    ready: bool
    camera_kind: str
    camera_source: str
    dealer_adapter: str
    dealer_available: bool
    physical_motion: bool
    speech_enabled: bool
    controls: tuple[str, ...]
    resources: tuple[str, ...]
    perception_calibration_id: str
    target_geometry_validated: bool
    reason: str | None
    development_live_available: bool = False
    full_live_hand_integrated: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CameraSmokeResult:
    profile_id: str
    requested_frames: int
    ok_frames: int
    missing_frames: int
    disconnected_frames: int
    elapsed_ms: int
    physical_motion: bool = False

    @property
    def passed(self) -> bool:
        return self.ok_frames > 0 and self.disconnected_frames == 0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


class LiveHandApplication:
    """Own live dependencies while leaving game authority in ``HandRuntime``."""

    def __init__(self, project_root: Path, profile: RuntimeProfile) -> None:
        self.project_root = project_root.resolve()
        self.profile = profile
        self.game_config = CoreGameConfig.from_json(
            self.project_root / "configs" / "game" / "core_v1.json"
        )
        self.camera = OpenCVCamera(profile.camera.to_camera_config())
        self.dealer = self._build_dealer(profile)
        self._locks = RuntimeResourceLocks(
            self.project_root / "runs" / "runtime" / "locks",
            self.resource_ids,
        )
        self._opened = False
        self._camera_opened = False

    @property
    def resource_ids(self) -> tuple[str, ...]:
        camera = self.profile.camera
        if camera.kind is RuntimeCameraKind.LOCAL:
            camera_id = f"camera:local:{camera.device_index}"
        else:
            assert camera.stream_url is not None
            digest = hashlib.sha256(camera.stream_url.encode("utf-8")).hexdigest()[:16]
            camera_id = f"camera:mjpeg:{digest}"
        resources = [camera_id]
        if self.profile.speech_enabled:
            device = self.profile.speech_device
            resources.append(f"microphone:{'default' if device is None else device}")
        if self.profile.dealer.physical_motion:
            resources.append(f"dealer:{self.profile.dealer.device_id}")
        return tuple(resources)

    def preflight(self) -> RuntimePreflight:
        health = self.dealer.health()
        reason = None
        ready = self.profile.dealer.enabled and health.available
        if not self.profile.dealer.enabled:
            reason = self.profile.dealer.unavailable_reason
        elif not health.available:
            reason = health.reason
        camera_source = (
            str(self.profile.camera.device_index)
            if self.profile.camera.kind is RuntimeCameraKind.LOCAL
            else str(self.profile.camera.stream_url)
        )
        return RuntimePreflight(
            profile_id=self.profile.profile_id.value,
            ready=ready,
            camera_kind=self.profile.camera.kind.value,
            camera_source=camera_source,
            dealer_adapter=self.profile.dealer.adapter.value,
            dealer_available=health.available,
            physical_motion=health.physical_motion,
            speech_enabled=self.profile.speech_enabled,
            controls=tuple(item.value for item in self.profile.controls),
            resources=self.resource_ids,
            perception_calibration_id=self.profile.perception.calibration_id,
            target_geometry_validated=(
                self.profile.perception.target_geometry_validated
            ),
            reason=reason,
            development_live_available=(
                ready and not health.physical_motion
            ),
        )

    def event_log_path(self, *, session_id: str, hand_id: str) -> Path:
        """Return an isolated ignored JSONL path without creating it."""

        for label, value in (("session_id", session_id), ("hand_id", hand_id)):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError(
                    f"{label} must contain only letters, digits, dot, dash or underscore"
                )
        return (
            self.profile.resolved_log_root(self.project_root)
            / session_id
            / f"{hand_id}.jsonl"
        )

    def reserve_event_log(self, *, session_id: str, hand_id: str) -> Path:
        """Create one new hand log exclusively; existing evidence is never overwritten."""

        path = self.event_log_path(session_id=session_id, hand_id=hand_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        return path

    def open(self, *, open_camera: bool = True) -> LiveHandApplication:
        if self._opened:
            return self
        preflight = self.preflight()
        if not preflight.ready:
            raise DealerUnavailableError(preflight.reason or "runtime profile unavailable")
        self._locks.acquire()
        try:
            self.dealer.open()
            if open_camera:
                self.camera.open()
                self._camera_opened = True
            self._opened = True
        except Exception:
            self.dealer.close()
            self._locks.release()
            raise
        return self

    def camera_smoke(
        self, *, requested_frames: int = 10, max_seconds: float = 10.0
    ) -> CameraSmokeResult:
        if requested_frames <= 0 or max_seconds <= 0:
            raise ValueError("camera smoke limits must be positive")
        if not self._opened or not self._camera_opened:
            raise RuntimeError("application camera is not open")
        started = time.monotonic_ns()
        deadline = started + int(max_seconds * 1_000_000_000)
        counts = {status: 0 for status in CameraReadStatus}
        for _ in range(requested_frames):
            if time.monotonic_ns() >= deadline:
                break
            observation = self.camera.read()
            counts[observation.status] += 1
            if observation.status is CameraReadStatus.DISCONNECTED:
                break
        return CameraSmokeResult(
            profile_id=self.profile.profile_id.value,
            requested_frames=requested_frames,
            ok_frames=counts[CameraReadStatus.OK],
            missing_frames=counts[CameraReadStatus.MISSING],
            disconnected_frames=counts[CameraReadStatus.DISCONNECTED],
            elapsed_ms=(time.monotonic_ns() - started) // 1_000_000,
        )

    def create_hand(
        self,
        *,
        hand_id: str,
        roster: FrozenSessionRoster,
        stacks: Mapping[Seat, int] | None = None,
        rules: FixedLimitRules | None = None,
    ) -> HandRuntime:
        if not self._opened:
            raise RuntimeError("application must be open before starting a hand")
        return HandRuntime.from_roster(
            hand_id=hand_id,
            roster=roster,
            stacks=stacks or self.game_config.default_stacks(),
            rules=rules or self.game_config.rules,
        )

    def create_session(
        self,
        *,
        roster: FrozenSessionRoster,
        stacks: Mapping[Seat, int] | None = None,
    ) -> SessionRuntime:
        if not self._opened:
            raise RuntimeError("application must be open before starting a session")
        return SessionRuntime(roster, self.game_config, stacks=stacks)

    def execute_dealer_command(self, command: DealerCommand) -> DealerAck:
        if not self._opened:
            raise RuntimeError("application must be open before dealer commands")
        return self.dealer.execute(command)

    def close(self) -> None:
        if self._camera_opened:
            self.camera.close()
            self._camera_opened = False
        self.dealer.close()
        self._locks.release()
        self._opened = False

    def __enter__(self) -> LiveHandApplication:
        return self.open()

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _build_dealer(profile: RuntimeProfile) -> DealerPort:
        dealer = profile.dealer
        if dealer.adapter is DealerAdapterKind.SIMULATED:
            return SimulatedDealerAdapter(dealer.device_id)
        return UnavailableDealerAdapter(
            dealer.device_id,
            dealer.unavailable_reason or "real dealer adapter is not integrated",
        )


__all__ = ["CameraSmokeResult", "LiveHandApplication", "RuntimePreflight"]
