"""Discover camera entities from the Home Assistant Core API.

When this service runs as a Home Assistant add-on with `homeassistant_api:
true`, the Supervisor injects a `SUPERVISOR_TOKEN` granting access to the Core
API at http://supervisor/core/api. We use it to enumerate the `camera.*`
entities the user already has, so the fixture UI can list them without a
separate `unifi` credential block.

Note this only *discovers* cameras. Pulling a recorded clip for a past time
range is NVR-specific (see `protect.py`); a plain HA camera entity exposes
live snapshots/streams, not arbitrary historical footage.

Stdlib-only (urllib) so it adds no dependency and works even in the minimal
add-on image.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = "http://supervisor/core/api"


def _token() -> Optional[str]:
    # SUPERVISOR_TOKEN is the current name; HASSIO_TOKEN is the legacy alias.
    return os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HASSIO_TOKEN")


def _base_url() -> str:
    return os.environ.get("PACKAGE_WATCHER_HA_URL", DEFAULT_BASE_URL).rstrip("/")


def available() -> bool:
    """True when a Supervisor token is present to reach the HA Core API."""
    return bool(_token())


def _get(path: str, timeout: float = 10.0) -> Any:
    token = _token()
    if not token:
        raise RuntimeError(
            "no SUPERVISOR_TOKEN — not running as a Home Assistant add-on "
            "(or homeassistant_api is not enabled)")
    req = urllib.request.Request(
        f"{_base_url()}{path}",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def list_cameras() -> list[dict[str, Any]]:
    """Return the HA `camera.*` entities as [{id, name, state}], name-sorted."""
    states = _get("/states")
    cams = []
    for s in states:
        eid = s.get("entity_id", "")
        if not eid.startswith("camera."):
            continue
        attrs = s.get("attributes") or {}
        cams.append({
            "id": eid,
            "name": attrs.get("friendly_name") or eid,
            "state": s.get("state"),
        })
    return sorted(cams, key=lambda c: (c["name"] or "").lower())
