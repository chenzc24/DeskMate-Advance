"""Validated network endpoints shared by runtime profiles and web services."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class MobileWebEndpoint:
    bind_host: str
    advertised_host: str
    port: int

    def __post_init__(self) -> None:
        _validate_host(self.bind_host, "mobile_web_console.bind_host")
        _validate_host(
            self.advertised_host,
            "mobile_web_console.advertised_host",
            allow_wildcard=False,
        )
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("mobile_web_console.port must be between 1 and 65535")

    @property
    def browser_url(self) -> str:
        host = self.advertised_host
        try:
            is_ipv6 = ipaddress.ip_address(host).version == 6
        except ValueError:
            is_ipv6 = False
        rendered_host = f"[{host}]" if is_ipv6 else host
        return f"http://{rendered_host}:{self.port}/"


@dataclass(frozen=True, slots=True)
class NetworkEndpoints:
    schema_version: str
    mobile_web_console: MobileWebEndpoint
    camera_streams: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported network-endpoints schema version")
        if not self.camera_streams:
            raise ValueError("camera_streams must contain at least one endpoint")
        for name, url in self.camera_streams.items():
            if not name.strip():
                raise ValueError("camera stream endpoint names must not be blank")
            _validate_http_url(url, f"camera_streams.{name}.url")

    @classmethod
    def from_json(cls, path: Path) -> NetworkEndpoints:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("network endpoints root must be an object")
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NetworkEndpoints:
        _reject_unknown(
            value,
            {"schema_version", "mobile_web_console", "camera_streams"},
            "network endpoints",
        )
        mobile = _object(value, "mobile_web_console")
        streams = _object(value, "camera_streams")
        _reject_unknown(
            mobile,
            {"bind_host", "advertised_host", "port"},
            "mobile web endpoint",
        )
        parsed_streams: dict[str, str] = {}
        for name, raw_endpoint in streams.items():
            if not isinstance(name, str) or not isinstance(raw_endpoint, Mapping):
                raise ValueError("camera stream entries must be named objects")
            _reject_unknown(raw_endpoint, {"url"}, f"camera stream {name}")
            url = raw_endpoint.get("url")
            if not isinstance(url, str):
                raise ValueError(f"camera_streams.{name}.url must be a string")
            parsed_streams[name] = url
        port = mobile.get("port")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError("mobile_web_console.port must be an integer")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            mobile_web_console=MobileWebEndpoint(
                bind_host=str(mobile.get("bind_host", "")),
                advertised_host=str(mobile.get("advertised_host", "")),
                port=port,
            ),
            camera_streams=parsed_streams,
        )

    def camera_stream_url(self, endpoint: str) -> str:
        try:
            return self.camera_streams[endpoint]
        except KeyError as exc:
            raise ValueError(f"unknown camera stream endpoint: {endpoint}") from exc


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be an object")
    return item


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


def _validate_host(value: str, label: str, *, allow_wildcard: bool = True) -> None:
    host = value.strip()
    if not host or host != value or any(character.isspace() for character in host):
        raise ValueError(f"{label} must be a non-blank host")
    if any(character in host for character in "/?#@"):
        raise ValueError(f"{label} must be a host without scheme, path or credentials")
    if not allow_wildcard and host in {"0.0.0.0", "::"}:
        raise ValueError(f"{label} must be a phone-reachable host, not a wildcard")
    parsed = urlsplit(f"//{host}")
    if parsed.hostname is None or parsed.port is not None:
        raise ValueError(f"{label} must not contain a port")


def _validate_http_url(value: str, label: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    if parsed.query:
        raise ValueError(f"{label} must not contain query parameters")
    if parsed.fragment:
        raise ValueError(f"{label} must not contain a fragment")


__all__ = ["MobileWebEndpoint", "NetworkEndpoints"]
