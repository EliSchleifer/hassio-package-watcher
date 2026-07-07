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
    # In person_gated mode: how long a person event keeps the camera in
    # "person present" after its last websocket update. Ongoing events keep
    # refreshing it; the hold papers over sparse updates near the end.
    presence_hold_seconds: float = 10.0
    # Maps watcher camera name -> Protect camera display name.
    camera_map: dict[str, str] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """Semantic verification of candidate crops with a local vision model.

    The CV stages find "something new and static"; the verifier answers
    *what is it* — the role a human labeler plays in fixtures, played live
    by a model. Off by default: it needs the `[verify]` extra installed."""

    backend: str = "off"             # "off" | "florence"
    # The florence-community repos are the official conversions for native
    # transformers support (microsoft/* still targets trust_remote_code).
    model: str = "florence-community/Florence-2-base"
    cache_dir: Optional[str] = None  # model download cache (default: HF cache)
    # Caption keywords that count as a delivered item.
    accept: list[str] = field(default_factory=lambda: [
        "package", "box", "parcel", "carton", "envelope", "crate"])
    crop_margin: float = 0.35        # context margin around the bbox crop
    suppress_rejected: bool = False  # drop events whose crop isn't accepted


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
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    sinks: SinkConfig = field(default_factory=SinkConfig)
    events_dir: str = "./events"
    # Per-camera watch zones (normalized polygons), keyed by Protect camera
    # id and/or display name. Maintained by the UI's zone drawer in
    # zones.yaml next to the config file; an inline `zones:` block wins.
    zones: dict[str, Any] = field(default_factory=dict)


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
    verifier = _dataclass_from(VerifierConfig, raw.get("verifier") or {})
    sinks = _dataclass_from(SinkConfig, raw.get("sinks") or {})

    # Watch zones: zones.yaml next to the config file (written by the UI's
    # zone drawer), overridden by an inline `zones:` block.
    zones: dict[str, Any] = {}
    sibling = os.path.join(os.path.dirname(os.path.abspath(path)), "zones.yaml")
    if os.path.isfile(sibling):
        with open(sibling, "r", encoding="utf-8") as f:
            zones.update(yaml.safe_load(f) or {})
    zones.update(raw.get("zones") or {})

    return AppConfig(
        cameras=cameras,
        detector=detector,
        unifi=unifi,
        verifier=verifier,
        sinks=sinks,
        events_dir=raw.get("events_dir", "./events"),
        zones=zones,
    )
