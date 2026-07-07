"""Backtest a camera's recorded day: sparse sampling + optional verification.

The production question is not "watch 1 fps forever" — it is "every X
minutes, is there maybe a package, and should a higher-level model look?".
This module answers it against history:

  for each sample time across the window (default every 10 min):
      skip it if a person was in frame (Protect person events)
      fetch one historical snapshot (cheap JPEG, no video download)
      diff it against the previous person-free snapshot
      package-shaped blobs -> candidate hits ("maybe a package, HERE")
      optionally: Florence captions the crop -> accepted / rejected

Comparing *consecutive* person-free samples keeps the lighting delta small
(10 minutes of sun, not hours) and re-baselines automatically: a package
shows up as a hit on the first sample after it arrives, then becomes part
of the scene. A global-change guard skips sunrise/сunset flips.

The engine takes an injectable `snapshot_fn` so it runs identically against
uiprotect or generated frames in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .config import UnifiConfig
from .detector import DetectorConfig, _BlobDetectorBase

log = logging.getLogger(__name__)

# snapshot_fn(dt) -> JPEG bytes | BGR ndarray | None (no footage at dt)
SnapshotFn = Callable[[datetime], Optional[Any]]


@dataclass
class Hit:
    at: datetime
    bbox: tuple[int, int, int, int]           # native pixels
    bbox_norm: tuple[float, float, float, float]
    confidence: float                          # CV heuristic
    frame: np.ndarray = field(repr=False)      # native frame at the hit
    crop: np.ndarray = field(repr=False)       # margined crop for the model
    # The comparison pair: this hit exists because `frame` differed from
    # `baseline` (the previous person-free snapshot, taken at baseline_at).
    baseline: Optional[np.ndarray] = field(default=None, repr=False)
    baseline_at: Optional[datetime] = None
    verification: Optional[dict[str, Any]] = None


@dataclass
class BacktestResult:
    camera_id: str
    start: datetime
    end: datetime
    interval_s: float
    samples_total: int = 0
    samples_skipped_person: int = 0
    samples_missing: int = 0
    scene_flips: int = 0                       # global-change guard trips
    hits: list[Hit] = field(default_factory=list)


class _SnapshotComparer(_BlobDetectorBase):
    """Reuses the shared prep + blob shape priors for pairwise comparison."""

    def _reset_models(self) -> None:  # no continuous state to reset
        pass

    def compare(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray
                ) -> tuple[list, bool]:
        """Blobs new in cur vs prev; True flag = global scene flip."""
        prev_gray = self._prepare(prev_bgr)
        cur_gray = self._prepare(cur_bgr)
        if prev_gray.shape != cur_gray.shape:
            return [], True
        diff = cv2.absdiff(cur_gray, prev_gray)
        changed = float(np.count_nonzero(
            diff > self.cfg.diff_threshold)) / diff.size
        if changed > self.cfg.global_change_frac:
            return [], True
        mask = (diff > self.cfg.diff_threshold).astype(np.uint8) * 255
        if self._zone_mask is not None:
            mask = cv2.bitwise_and(mask, self._zone_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self._extract_blobs(mask, diff), False


def _decode(snap: Any) -> Optional[np.ndarray]:
    if snap is None:
        return None
    if isinstance(snap, np.ndarray):
        return snap
    img = cv2.imdecode(np.frombuffer(snap, np.uint8), cv2.IMREAD_COLOR)
    return img


def _in_person_window(t_s: float, windows: list[tuple[float, float]],
                      pad_s: float) -> bool:
    return any(a - pad_s <= t_s <= b + pad_s for a, b in windows)


def run_backtest(snapshot_fn: SnapshotFn,
                 camera_id: str,
                 start: datetime,
                 end: datetime,
                 interval_s: float = 600.0,
                 person_windows: Optional[list[tuple[float, float]]] = None,
                 detector: Optional[DetectorConfig] = None,
                 zone: Optional[list[tuple[float, float]]] = None,
                 verifier=None,
                 person_pad_s: float = 30.0,
                 progress: Optional[Callable[[int, int], None]] = None,
                 on_hit: Optional[Callable[[Hit], None]] = None,
                 ) -> BacktestResult:
    """Sample [start, end] every interval_s and report candidate packages.

    person_windows are seconds relative to `start` (as produced by
    protect.person_windows). Samples inside a window (± pad) are skipped —
    both to avoid captioning people and because the interesting comparison
    is clean-before vs clean-after a visit.
    """
    cfg = detector or DetectorConfig()
    comparer = _SnapshotComparer(cfg, zone=zone)
    windows = person_windows or []
    result = BacktestResult(camera_id=camera_id, start=start, end=end,
                            interval_s=interval_s)

    times: list[datetime] = []
    t = start
    while t <= end:
        times.append(t)
        t += timedelta(seconds=interval_s)
    result.samples_total = len(times)

    prev_frame: Optional[np.ndarray] = None
    prev_at: Optional[datetime] = None
    for i, at in enumerate(times):
        if progress:
            progress(i, len(times))
        rel = (at - start).total_seconds()
        if _in_person_window(rel, windows, person_pad_s):
            result.samples_skipped_person += 1
            continue
        frame = _decode(snapshot_fn(at))
        if frame is None:
            result.samples_missing += 1
            continue
        if prev_frame is not None:
            blobs, flipped = comparer.compare(prev_frame, frame)
            if flipped:
                result.scene_flips += 1
            for bbox, contrast, blob_mask in blobs:
                report = comparer._blob_report(
                    candidate_id=len(result.hits) + 1, bbox=bbox,
                    contrast=contrast, mask=blob_mask,
                    baseline=np.empty(0), frame=frame,
                    first_seen=at.timestamp(), ts=at.timestamp(),
                    persisted=1, triggered=False)
                from .verify import crop_with_margin
                crop = crop_with_margin(frame, report.bbox, 0.35)
                verdict = None
                if verifier is not None:
                    try:
                        verdict = verifier.verify(frame, report.bbox)
                    except Exception as exc:  # noqa: BLE001
                        verdict = {"error": str(exc)}
                hit = Hit(
                    at=at, bbox=report.bbox, bbox_norm=report.bbox_norm,
                    confidence=report.confidence, frame=frame, crop=crop,
                    baseline=prev_frame, baseline_at=prev_at,
                    verification=verdict)
                result.hits.append(hit)
                if on_hit:
                    on_hit(hit)  # lets the UI stream hits while scanning
        # The current person-free sample becomes the next baseline whether or
        # not anything was found — arrivals report once, then join the scene.
        prev_frame = frame
        prev_at = at

    if progress:
        progress(len(times), len(times))
    return result


def protect_snapshot_fn(unifi: UnifiConfig, camera_id: str,
                        width: int = 640) -> SnapshotFn:
    """A SnapshotFn backed by the Protect NVR (warm persistent session)."""
    from .ui import protect

    def fn(at: datetime):
        try:
            return protect.snapshot_at(unifi, camera_id, at, width=width)
        except Exception as exc:  # noqa: BLE001 - a gap in recording is data,
            log.debug("no snapshot at %s: %s", at, exc)  # not a crash
            return None
    return fn
