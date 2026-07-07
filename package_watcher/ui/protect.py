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
import threading
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


class _Session:
    """A Protect client kept warm on a background event loop.

    Scrubbing fetches one frame per seek, so paying the bootstrap/login cost
    on every call (as ``_with_client`` does) would make the timeline sluggish.
    This bootstraps once and reuses the connection for subsequent snapshots;
    it re-connects automatically if credentials change or a call fails.
    """

    def __init__(self) -> None:
        self._loop = None
        self._client = None
        self._key = None
        self._lock = threading.Lock()

    def _ensure_loop(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, daemon=True).start()
        return self._loop

    async def _client_for(self, cfg: UnifiConfig):
        key = (cfg.host, cfg.port, cfg.username, cfg.api_key, cfg.verify_ssl)
        if self._client is not None and self._key == key:
            return self._client
        if self._client is not None:
            try:
                await self._client.close_session()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        from uiprotect import ProtectApiClient

        client = ProtectApiClient(
            cfg.host, cfg.port, cfg.username, cfg.password,
            api_key=cfg.api_key, verify_ssl=cfg.verify_ssl)
        await client.update()
        self._client, self._key = client, key
        return client

    def snapshot(self, cfg: UnifiConfig, camera_id: str,
                 dt: Optional[datetime],
                 width: Optional[int]) -> Optional[bytes]:
        with self._lock:
            loop = self._ensure_loop()

            async def _fn():
                client = await self._client_for(cfg)
                cam = client.bootstrap.cameras.get(camera_id)
                if cam is None:
                    raise ValueError(f"no Protect camera with id {camera_id!r}")
                return await cam.get_snapshot(width=width, dt=dt)

            fut = asyncio.run_coroutine_threadsafe(_fn(), loop)
            try:
                return fut.result(timeout=30)
            except Exception:
                # Drop the client so the next call reconnects cleanly.
                self._client = self._key = None
                raise


_session = _Session()


def snapshot_at(cfg: UnifiConfig, camera_id: str, dt: Optional[datetime],
                width: Optional[int] = 640) -> Optional[bytes]:
    """One frame at ``dt`` (or the LIVE view when dt is None).

    width=640 keeps interactive scrubbing snappy; pass width=None for the
    camera's native resolution — detection/verification paths should, since
    they are cron-cadence work where quality beats latency. Recorded footage
    at "now" does not exist yet on the NVR, so callers wanting the current
    view must pass dt=None."""
    return _session.snapshot(cfg, camera_id, dt, width)


def _events_to_windows(events, camera_id: str, start: datetime,
                       end: datetime) -> list[tuple[float, float]]:
    """Reduce Protect events to clip-relative (start_s, end_s) windows for
    one camera, clamped to [0, clip length]. Pure, for testability."""
    clip_len = (end - start).total_seconds()
    windows: list[tuple[float, float]] = []
    for ev in events:
        if getattr(ev, "camera_id", None) != camera_id:
            continue
        ev_start = getattr(ev, "start", None)
        if ev_start is None:
            continue
        ev_end = getattr(ev, "end", None) or end  # ongoing -> clip end
        a = max(0.0, (ev_start - start).total_seconds())
        b = min(clip_len, (ev_end - start).total_seconds())
        if b > a:
            windows.append((round(a, 1), round(b, 1)))
    windows.sort()
    # Merge overlaps so the UI shows clean, disjoint windows.
    merged: list[tuple[float, float]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def camera_zones(cfg: UnifiConfig, camera_id: str) -> list[dict[str, Any]]:
    """The zones already configured on the camera in UniFi Protect
    (smart-detection and motion zones), as normalized polygons — so a watch
    zone can be imported instead of drawn."""
    async def _fn(client):
        cam = client.bootstrap.cameras.get(camera_id)
        if cam is None:
            raise ValueError(f"no Protect camera with id {camera_id!r}")
        out = []
        for kind, zones in (("smart", cam.smart_detect_zones or []),
                            ("motion", cam.motion_zones or [])):
            for z in zones:
                pts = [[float(x), float(y)] for x, y in (z.points or [])]
                if len(pts) >= 3:
                    out.append({"name": z.name, "kind": kind, "points": pts})
        return out
    return asyncio.run(_with_client(cfg, _fn))


def person_windows(cfg: UnifiConfig, camera_id: str, start: datetime,
                   end: datetime) -> list[tuple[float, float]]:
    """Person smart-detect windows over [start, end], clip-relative seconds.

    Lets a pulled clip arrive already labeled with when a person was in
    frame — the ground truth the person-gated detector mode needs."""
    async def _fn(client):
        from uiprotect.data.types import EventType, SmartDetectObjectType
        events = await client.get_events(
            start=start, end=end,
            types=[EventType.SMART_DETECT],
            smart_detect_types=[SmartDetectObjectType.PERSON])
        return _events_to_windows(events, camera_id, start, end)
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
