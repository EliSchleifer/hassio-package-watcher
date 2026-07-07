"""Event model: what the watcher tells the world when it finds something.

The JSON is deliberately self-describing and LLM-ready: coordinates in both
pixel and normalized form, the heuristic signals that produced the call, and
paths to the evidence images. A verifier (human or LLM) should be able to
judge the detection from this payload alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .detector import NewObjectReport


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class TriggerInfo:
    kind: str          # e.g. "person", "vehicle", "manual"
    source: str        # e.g. "unifi-protect"
    at: float          # epoch seconds


LLM_PROMPT_TEMPLATE = (
    "A CPU-based motion watcher believes a new stationary object appeared on "
    "the '{camera}' camera at {when}. The attached images are: (1) the full "
    "frame with the region outlined, (2) a close-up crop of the region, "
    "(3) the scene baseline from before the object appeared, and (4) the "
    "pixel-difference mask that drove the call. The region is at pixels "
    "x={x}, y={y}, w={w}, h={h} in a {fw}x{fh} frame. Question: does the "
    "highlighted region contain a package or delivered item? Answer with "
    "yes/no/unsure and a one-sentence justification."
)


def build_event(camera: str, report: NewObjectReport,
                trigger: Optional[TriggerInfo] = None,
                evidence_paths: Optional[dict[str, str]] = None,
                verification: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    x, y, w, h = report.bbox
    fw, fh = report.frame_size
    stamp = datetime.fromtimestamp(report.reported_at, tz=timezone.utc)
    event_id = f"{camera}-{stamp.strftime('%Y%m%dT%H%M%SZ')}-{report.candidate_id}"
    event: dict[str, Any] = {
        "id": event_id,
        "kind": "new_static_object",
        "camera": camera,
        "first_seen": _iso(report.first_seen),
        "reported_at": _iso(report.reported_at),
        "samples_persisted": report.samples_persisted,
        "frame_size": {"width": fw, "height": fh},
        "bbox_pixels": {"x": x, "y": y, "w": w, "h": h},
        "bbox_normalized": {
            "x": round(report.bbox_norm[0], 4),
            "y": round(report.bbox_norm[1], 4),
            "w": round(report.bbox_norm[2], 4),
            "h": round(report.bbox_norm[3], 4),
        },
        "signals": {
            "area_fraction": round(report.area_fraction, 5),
            "contrast": round(report.contrast, 4),
            "confidence": round(report.confidence, 3),
        },
        "trigger": (
            {"kind": trigger.kind, "source": trigger.source, "at": _iso(trigger.at)}
            if trigger else None
        ),
        # Verdict from the local vision model (verify.py), when enabled:
        # {accepted, label, caption, backend, model}.
        "verification": verification,
        "evidence": evidence_paths or {},
        "llm_verification": {
            "suggested_prompt": LLM_PROMPT_TEMPLATE.format(
                camera=camera, when=_iso(report.reported_at),
                x=x, y=y, w=w, h=h, fw=fw, fh=fh),
        },
    }
    return event


def event_json(event: dict[str, Any]) -> str:
    return json.dumps(event, indent=2, sort_keys=False)
