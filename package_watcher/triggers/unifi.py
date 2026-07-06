"""Unifi Protect smart-detection triggers via the `uiprotect` websocket.

Protect's NVR already runs person/vehicle smart detection on-device. We
subscribe to those events and use them as *attention* signals: a person at
the door makes a subsequent new-stationary-object far more likely to be a
delivery, so the detector lowers its persistence bar for a configurable
window after the trigger.

This module is optional — it only imports `uiprotect` when actually started,
so the core watcher runs without the dependency installed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from ..config import UnifiConfig

log = logging.getLogger(__name__)

# Callback signature: (watcher_camera_name, trigger_kind, epoch_ts)
TriggerCallback = Callable[[str, str, float], None]


class UnifiTriggerListener:
    """Runs the uiprotect websocket client on a daemon thread."""

    def __init__(self, cfg: UnifiConfig, on_trigger: TriggerCallback):
        self.cfg = cfg
        self.on_trigger = on_trigger
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Reverse map: Protect camera display name -> watcher camera name.
        self._name_map = {v: k for k, v in cfg.camera_map.items()}

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, name="unifi-triggers", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # noqa: BLE001 - triggers are best-effort
            log.error("unifi trigger listener died: %s", exc)

    async def _run(self) -> None:
        try:
            from uiprotect import ProtectApiClient
        except ImportError:
            log.error(
                "uiprotect is not installed; run `pip install "
                "package-watcher[unifi]` to enable Protect triggers")
            return

        while not self._stop.is_set():
            client = ProtectApiClient(
                self.cfg.host, self.cfg.port, self.cfg.username,
                self.cfg.password, verify_ssl=self.cfg.verify_ssl)
            try:
                await client.update()
                unsub = client.subscribe_websocket(self._on_message(client))
                log.info("subscribed to Unifi Protect events at %s",
                         self.cfg.host)
                while not self._stop.is_set():
                    await asyncio.sleep(1.0)
                unsub()
            except Exception as exc:  # noqa: BLE001
                log.warning("unifi connection error (%s); retrying in 30s", exc)
                await asyncio.sleep(30)
            finally:
                try:
                    await client.close_session()
                except Exception:  # noqa: BLE001
                    pass

    def _on_message(self, client):
        wanted = {t.lower() for t in self.cfg.trigger_types}

        def handler(msg) -> None:
            try:
                obj = getattr(msg, "new_obj", None)
                if obj is None or type(obj).__name__ != "Event":
                    return
                smart_types = [
                    getattr(t, "value", str(t))
                    for t in (getattr(obj, "smart_detect_types", None) or [])
                ]
                hit = next((t for t in smart_types if t.lower() in wanted), None)
                if hit is None:
                    return
                camera = getattr(obj, "camera", None)
                protect_name = getattr(camera, "name", None) if camera else None
                if protect_name is None:
                    return
                watcher_name = self._name_map.get(protect_name)
                if watcher_name is None:
                    return  # not a camera we watch
                start = getattr(obj, "start", None)
                ts = start.timestamp() if start is not None else None
                import time as _time
                self.on_trigger(watcher_name, hit, ts or _time.time())
            except Exception as exc:  # noqa: BLE001
                log.debug("ignoring malformed protect message: %s", exc)

        return handler
