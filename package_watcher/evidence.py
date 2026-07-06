"""Evidence writer — the "show your work" half of the watcher.

For every event we save a small bundle next to the JSON so a human or an LLM
can audit the call:

    events/<camera>/<event-id>/
        event.json      the full event payload
        annotated.jpg   native frame with the region outlined + labeled
        crop.jpg        close-up of the region (with margin)
        baseline.jpg    slow-background snapshot from before the object
        mask.png        the binary diff mask that produced the blob
        raw.jpg         untouched native frame at report time
"""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np

from .detector import NewObjectReport

_BOX_COLOR = (60, 220, 60)   # BGR green
_CROP_MARGIN = 0.35          # extra context around the bbox in the crop


def _clamp_box(x: int, y: int, w: int, h: int, fw: int, fh: int,
               margin: float = 0.0) -> tuple[int, int, int, int]:
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(fw, x + w + mx)
    y1 = min(fh, y + h + my)
    return x0, y0, x1, y1


def write_evidence(event: dict[str, Any], report: NewObjectReport,
                   base_dir: str) -> dict[str, str]:
    """Write the evidence bundle; return relative paths keyed by kind."""
    event_dir = os.path.join(base_dir, event["camera"], event["id"])
    os.makedirs(event_dir, exist_ok=True)

    frame = report.frame
    fh, fw = frame.shape[:2]
    x, y, w, h = report.bbox

    paths: dict[str, str] = {}

    def _save(name: str, image: np.ndarray) -> None:
        path = os.path.join(event_dir, name)
        cv2.imwrite(path, image)
        paths[os.path.splitext(name)[0]] = os.path.relpath(path, base_dir)

    _save("raw.jpg", frame)

    annotated = frame.copy()
    thickness = max(2, fw // 640)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), _BOX_COLOR, thickness)
    label = f"new object {event['signals']['confidence']:.2f}"
    cv2.putText(annotated, label, (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(1, fw // 640),
                _BOX_COLOR, thickness)
    _save("annotated.jpg", annotated)

    cx0, cy0, cx1, cy1 = _clamp_box(x, y, w, h, fw, fh, _CROP_MARGIN)
    if cx1 > cx0 and cy1 > cy0:
        _save("crop.jpg", frame[cy0:cy1, cx0:cx1])

    if report.baseline.size:
        _save("baseline.jpg", report.baseline)
    if report.mask.size:
        _save("mask.png", report.mask)

    event["evidence"] = paths
    with open(os.path.join(event_dir, "event.json"), "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    paths["event"] = os.path.relpath(
        os.path.join(event_dir, "event.json"), base_dir)
    return paths
