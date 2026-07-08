"""Replay recorded person events as labelable before/after cards.

The production trigger already exists outside this tool (the camera fires
on person-leaving and a model is asked about packages). What this module
adds is the DATA side of that loop: Protect remembers every person event,
so for any day we can reconstruct each visit's (before, after) pair, run
the standard localize→delineate→name pipeline on it, and hand a human a
card to label — "package" or "nothing". The labels accumulate into a
training set (and hard negatives) for whatever model eventually replaces
or augments the caption stage.

Same engine shape as backtest.py: injectable snapshot_fn, deterministic,
testable without an NVR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np

from .backtest import SnapshotFn, _SnapshotComparer, _decode
from .detector import DetectorConfig

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    verification: Optional[dict[str, Any]] = None


@dataclass
class EventReview:
    index: int
    start: datetime                     # person event window (from Protect)
    end: datetime
    before_at: Optional[datetime] = None
    after_at: Optional[datetime] = None
    before: Optional[np.ndarray] = None  # clean frame preceding the visit
    after: Optional[np.ndarray] = None   # settled frame after the visit
    candidates: list[Candidate] = field(default_factory=list)
    scene_flip: bool = False
    error: Optional[str] = None


def merge_events(events: list[tuple[datetime, datetime]],
                 gap_s: float) -> list[tuple[datetime, datetime]]:
    """Visits separated by less than gap_s are one visit — their brackets
    would otherwise collide (the 'after' of one inside the next)."""
    merged: list[tuple[datetime, datetime]] = []
    for a, b in sorted(events):
        if merged and (a - merged[-1][1]).total_seconds() < gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def review_person_events(snapshot_fn: SnapshotFn,
                         events: list[tuple[datetime, datetime]],
                         *,
                         before_margin_s: float = 20.0,
                         settle_s: float = 45.0,
                         detector: Optional[DetectorConfig] = None,
                         zone: Optional[list[tuple[float, float]]] = None,
                         verifier=None,
                         progress: Optional[Callable[[int, int], None]] = None,
                         on_event: Optional[Callable[[EventReview], None]] = None,
                         ) -> list[EventReview]:
    """One before/after comparison per person visit.

    before = margin before the visit started; after = settle after it ended,
    clamped so it never runs into the next visit. Candidates carry the usual
    shape/zone gating plus optional model verification.
    """
    cfg = detector or DetectorConfig()
    comparer = _SnapshotComparer(cfg, zone=zone)
    events = merge_events(events, gap_s=before_margin_s + settle_s)
    out: list[EventReview] = []

    for i, (start, end) in enumerate(events):
        if progress:
            progress(i, len(events))
        rv = EventReview(index=i, start=start, end=end)
        before_at = start - timedelta(seconds=before_margin_s)
        after_at = end + timedelta(seconds=settle_s)
        if i + 1 < len(events):
            latest = events[i + 1][0] - timedelta(seconds=5)
            after_at = min(after_at, max(latest, end))
        rv.before_at, rv.after_at = before_at, after_at

        before = _decode(snapshot_fn(before_at))
        after = _decode(snapshot_fn(after_at))
        if before is None or after is None:
            rv.error = "missing footage for the before/after pair"
            out.append(rv)
            if on_event:
                on_event(rv)
            continue
        rv.before, rv.after = before, after

        blobs, flipped = comparer.compare(before, after)
        rv.scene_flip = flipped
        for bbox, contrast, blob_mask in blobs:
            report = comparer._blob_report(
                candidate_id=len(rv.candidates) + 1, bbox=bbox,
                contrast=contrast, mask=blob_mask, baseline=np.empty(0),
                frame=after, first_seen=end.timestamp(),
                ts=after_at.timestamp(), persisted=1, triggered=True)
            verdict = None
            if verifier is not None:
                try:
                    verdict = verifier.verify(after, report.bbox)
                except Exception as exc:  # noqa: BLE001
                    verdict = {"error": str(exc)}
            rv.candidates.append(Candidate(
                bbox=report.bbox, bbox_norm=report.bbox_norm,
                confidence=report.confidence, verification=verdict))
        out.append(rv)
        if on_event:
            on_event(rv)

    if progress:
        progress(len(events), len(events))
    return out
