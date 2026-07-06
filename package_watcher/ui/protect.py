"""Pull recorded clips from Unifi Protect for a begin/end time range.

Works for both a direct Unifi Protect NVR and one surfaced through Home
Assistant, since HA just proxies the same Protect backend — point the
`unifi` config block at whichever host exposes the Protect API.

Kept separate and import-light so the UI runs even without `uiprotect`
installed (camera listing / clip pull simply report as unavailable).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Optional

from ..config import UnifiConfig


def available() -> bool:
    try:
        import uiprotect  # noqa: F401
        return True
    except ImportError:
        return False


async def _with_client(cfg: UnifiConfig, fn):
    from uiprotect import ProtectApiClient

    client = ProtectApiClient(
        cfg.host, cfg.port, cfg.username, cfg.password,
        api_key=cfg.api_key, verify_ssl=cfg.verify_ssl)
    try:
        await client.update()
        return await fn(client)
    finally:
        try:
            await client.close_session()
        except Exception:  # noqa: BLE001
            pass


def list_cameras(cfg: UnifiConfig) -> list[dict[str, Any]]:
    async def _fn(client):
        cams = []
        for cam in client.bootstrap.cameras.values():
            cams.append({
                "id": cam.id,
                "name": cam.name,
                "is_recording": getattr(cam, "is_recording", None),
            })
        return sorted(cams, key=lambda c: c["name"] or "")
    return asyncio.run(_with_client(cfg, _fn))


def pull_clip(cfg: UnifiConfig, camera_id: str, start: datetime,
              end: datetime, output_path: str) -> str:
    """Download recorded footage [start, end] for a camera to output_path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    async def _fn(client):
        camera = client.bootstrap.cameras.get(camera_id)
        if camera is None:
            raise ValueError(f"no Protect camera with id {camera_id!r}")
        # uiprotect writes the mp4 directly to output_file.
        await camera.get_video(start, end, output_file=output_path)
        return output_path

    return asyncio.run(_with_client(cfg, _fn))
