"""Microphone device discovery and no-save smoke diagnostics."""

from __future__ import annotations

import math

import numpy as np

from .adapter import MicrophoneConfig, MicrophoneError, SoundDeviceMicrophone, sd


def list_input_devices() -> list[dict[str, int | float | str]]:
    """Return available input devices without opening or recording from them."""

    if sd is None:
        raise MicrophoneError("sounddevice is not installed")
    host_apis = sd.query_hostapis()
    devices: list[dict[str, int | float | str]] = []
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) <= 0:
            continue
        host_api_index = int(device["hostapi"])
        devices.append(
            {
                "device_index": index,
                "name": str(device["name"]),
                "host_api": str(host_apis[host_api_index]["name"]),
                "max_input_channels": int(device["max_input_channels"]),
                "default_sample_rate_hz": float(device["default_samplerate"]),
            }
        )
    return devices


def check_input_config(config: MicrophoneConfig) -> dict[str, object]:
    """Validate a requested format without capturing samples."""

    if sd is None:
        raise MicrophoneError("sounddevice is not installed")
    device = sd.query_devices(config.device_index)
    try:
        sd.check_input_settings(
            device=config.device_index,
            samplerate=config.sample_rate_hz,
            channels=config.channel_count,
            dtype="float32",
        )
    except Exception as error:
        return {
            "device_index": config.device_index,
            "device_name": str(device["name"]),
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "device_index": config.device_index,
        "device_name": str(device["name"]),
        "sample_rate_hz": config.sample_rate_hz,
        "channel_count": config.channel_count,
        "block_frames": config.block_frames,
        "supported": True,
        "error": None,
    }


def smoke_read(config: MicrophoneConfig) -> dict[str, object]:
    """Read one bounded block and return only aggregate metadata, never samples."""

    with SoundDeviceMicrophone(config) as microphone:
        result = microphone.read()
        report: dict[str, object] = microphone.negotiated_properties()
        report.update(
            {
                "read_status": result.status.value,
                "observed_at_ns": result.observed_at_ns,
                "error": result.reason,
            }
        )
        if result.packet is not None:
            rms = float(np.sqrt(np.mean(np.square(result.packet.samples), dtype=np.float64)))
            report["rms"] = rms
            report["dbfs"] = 20.0 * math.log10(max(rms, 1e-12))
            report["input_overflowed"] = result.packet.input_overflowed
        return report
