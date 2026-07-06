"""Event sinks: stdout, JSONL log, and webhook POST."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from typing import Any

from .config import SinkConfig

log = logging.getLogger(__name__)


class EventSinks:
    def __init__(self, cfg: SinkConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        if cfg.jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(cfg.jsonl_path)),
                        exist_ok=True)

    def emit(self, event: dict[str, Any]) -> None:
        if self.cfg.stdout:
            print(json.dumps(event), flush=True)
        if self.cfg.jsonl_path:
            with self._lock, open(self.cfg.jsonl_path, "a",
                                  encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        if self.cfg.webhook_url:
            threading.Thread(
                target=self._post, args=(event,), daemon=True).start()

    def _post(self, event: dict[str, Any]) -> None:
        try:
            req = urllib.request.Request(
                self.cfg.webhook_url,
                data=json.dumps(event).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as exc:  # noqa: BLE001 - sink failures must not kill the watcher
            log.warning("webhook delivery failed: %s", exc)
