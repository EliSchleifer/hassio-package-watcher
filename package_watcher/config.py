"""YAML configuration loading with ${ENV_VAR} interpolation for secrets."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .detector import DetectorConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise KeyError(
                    f"config references ${{{var}}} but it is not set in the environment")
            return os.environ[var]
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


@dataclass
class CameraConfig:
    name: str
    source: str                       # rtsp(s):// URL or a video file path
    zone: Optional[list[tuple[float, float]]] = None  # normalized polygon
    sample_fps: float = 1.0


@dataclass
class UnifiConfig:
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None   # newer Protect integrations use an API key
    port: int = 443
    verify_ssl: bool = False
    trigger_types: list[str] = field(default_factory=lambda: ["person"])
    attention_seconds: float = 120.0
    # Maps watcher camera name -> Protect camera display name.
    camera_map: dict[str, str] = field(default_factory=dict)


@dataclass
class SinkConfig:
    jsonl_path: Optional[str] = None
    webhook_url: Optional[str] = None
    stdout: bool = True


@dataclass
class AppConfig:
    cameras: list[CameraConfig]
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    unifi: Optional[UnifiConfig] = None
    sinks: SinkConfig = field(default_factory=SinkConfig)
    events_dir: str = "./events"


def _dataclass_from(cls, data: dict[str, Any]):
    fields = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str, require_cameras: bool = True) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _interpolate(raw)

    cameras_raw = raw.get("cameras") or []
    # The authoring UI only needs the `unifi` block, so it loads with
    # require_cameras=False — a config with just credentials is valid there.
    if not cameras_raw and require_cameras:
        raise ValueError("config must define at least one camera")
    cameras = []
    for cam in cameras_raw:
        zone = cam.get("zone")
        if zone is not None:
            zone = [tuple(pt) for pt in zone]
            cam = {**cam, "zone": zone}
        cameras.append(_dataclass_from(CameraConfig, cam))

    detector = _dataclass_from(DetectorConfig, raw.get("detector") or {})
    unifi = (_dataclass_from(UnifiConfig, raw["unifi"])
             if raw.get("unifi") else None)
    sinks = _dataclass_from(SinkConfig, raw.get("sinks") or {})

    return AppConfig(
        cameras=cameras,
        detector=detector,
        unifi=unifi,
        sinks=sinks,
        events_dir=raw.get("events_dir", "./events"),
    )
