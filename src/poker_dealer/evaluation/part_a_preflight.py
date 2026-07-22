"""Read-only Part A environment, asset and device preflight checks."""

from __future__ import annotations

from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Callable

from poker_dealer.io.camera.diagnostics import probe_camera_indices
from poker_dealer.perception.actions import GesturePilotConfig, SpeechPilotConfig
from poker_dealer.perception.identity import FaceIdentityConfig

from .four_player_acceptance import load_acceptance_protocol


JsonObject = dict[str, Any]


def _check(check_id: str, operation: Callable[[], object]) -> JsonObject:
    try:
        detail = operation()
    except Exception as exc:  # preflight must retain every failing check
        return {
            "check_id": check_id,
            "status": "FAIL",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {"check_id": check_id, "status": "PASS", "detail": detail}


def _package_version(name: str) -> str:
    return metadata.version(name)


def _manifest_summary(path: Path) -> JsonObject:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported model manifest schema")
    if manifest.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("model manifest models must be a list")
    required = {
        "player-action-mediapipe-canned-gesture",
        "player-action-vosk-small-en-us",
        "face-identity-opencv-yunet",
        "face-identity-opencv-sface",
        "player-action-landmark-tcn",
    }
    by_id = {str(model.get("model_id")): model for model in models}
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"model manifest missing {missing}")
    non_development = sorted(
        model_id for model_id in required if by_id[model_id].get("state") != "development"
    )
    if non_development:
        raise ValueError(f"unvalidated models are not development: {non_development}")
    return {"models": sorted(required), "runtime_downloads_allowed": False}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_sha256(path: Path, expected: str) -> str:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"expected {expected}, got {actual}")
    return actual


def _microphone_report(device: int | str) -> JsonObject:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is unavailable") from exc
    selected = sd.query_devices(device, "input")
    if int(selected["max_input_channels"]) < 1:
        raise ValueError("selected microphone has no input channel")
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=4000,
        device=device,
        dtype="int16",
        channels=1,
    ) as stream:
        block, overflowed = stream.read(4000)
    if len(block) != 8000:
        raise ValueError(f"microphone returned {len(block)} bytes, expected 8000")
    return {
        "requested": device,
        "name": str(selected["name"]),
        "max_input_channels": int(selected["max_input_channels"]),
        "default_samplerate": float(selected["default_samplerate"]),
        "validated_samplerate": 16000,
        "validated_frames": 4000,
        "overflowed": bool(overflowed),
    }


def run_part_a_preflight(
    root: Path,
    *,
    include_devices: bool,
    camera_index: int = 0,
    camera_backend: str = "dshow",
    speech_device: int | str = 1,
    minimum_free_gib: float = 1.0,
) -> JsonObject:
    root = root.resolve()
    gesture_path = root / "configs/perception/actions_laptop_pilot.json"
    speech_path = root / "configs/perception/actions_speech_pilot.json"
    identity_path = root / "configs/perception/face_identity_session.json"
    protocol_path = root / "configs/evaluation/stage2a_four_player_live_acceptance_v1.json"

    checks: list[JsonObject] = []
    checks.append(
        _check(
            "python_version",
            lambda: (
                platform.python_version()
                if (3, 11) <= sys.version_info[:2] < (3, 14)
                else (_ for _ in ()).throw(
                    ValueError(f"unsupported Python {platform.python_version()}")
                )
            ),
        )
    )
    for package in ("numpy", "opencv-contrib-python", "mediapipe", "vosk", "sounddevice"):
        checks.append(_check(f"package:{package}", lambda name=package: _package_version(name)))
    checks.append(
        _check(
            "gesture_asset_sha256",
            lambda: GesturePilotConfig.from_json(gesture_path).verify_model_asset(),
        )
    )
    checks.append(
        _check(
            "speech_asset_tree_sha256",
            lambda: SpeechPilotConfig.from_json(speech_path).verify_model_asset(),
        )
    )
    checks.append(
        _check(
            "face_assets_sha256",
            lambda: FaceIdentityConfig.from_json(identity_path).verify_assets(),
        )
    )
    checks.append(
        _check(
            "hand_landmarker_asset_sha256",
            lambda: _verify_file_sha256(
                root / "models/assets/hand_landmarker.task",
                "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
            ),
        )
    )
    checks.append(
        _check(
            "acceptance_protocol",
            lambda: {
                "protocol_id": load_acceptance_protocol(protocol_path)["protocol_id"],
                "cases": 9,
            },
        )
    )
    checks.append(
        _check("model_manifest", lambda: _manifest_summary(root / "models/manifest.yaml"))
    )

    def disk_report() -> JsonObject:
        usage = shutil.disk_usage(root)
        free_gib = usage.free / (1024**3)
        if free_gib < minimum_free_gib:
            raise ValueError(
                f"only {free_gib:.3f} GiB free; {minimum_free_gib:.3f} GiB required"
            )
        return {"free_gib": round(free_gib, 3), "minimum_gib": minimum_free_gib}

    checks.append(_check("disk_free", disk_report))

    if include_devices:
        def camera_report() -> JsonObject:
            report = probe_camera_indices(
                [camera_index],
                backend=camera_backend,
                width=1280,
                height=720,
                fps=30.0,
            )[0]
            if report.get("read_status") != "ok":
                raise ValueError(report.get("error") or "camera did not return a frame")
            return report

        checks.append(_check("camera_read", camera_report))
        checks.append(_check("microphone_input", lambda: _microphone_report(speech_device)))
    else:
        checks.extend(
            (
                {"check_id": "camera_read", "status": "SKIP", "detail": "devices disabled"},
                {
                    "check_id": "microphone_input",
                    "status": "SKIP",
                    "detail": "devices disabled",
                },
            )
        )

    failures = [item for item in checks if item["status"] == "FAIL"]
    return {
        "schema_version": "1.0",
        "preflight_id": "stage2a-part-a-preflight-v1",
        "result": "PASS" if not failures else "FAIL",
        "root": str(root),
        "include_devices": include_devices,
        "checks": checks,
        "privacy": {
            "frames_saved": 0,
            "audio_saved": False,
            "embeddings_persisted": False,
        },
        "physical_robot_connected": False,
    }
