"""Backtest a camera's recorded day: sparse sampling + optional verification.

The production question is not "watch 1 fps forever" — it is "every X
minutes, is there maybe a package, and should a higher-level model look?".
This module answers it against history:

  for each sample time across the window (default every 10 min):
      skip it if a person was in frame (Protect person events)
      fetch one historical snapshot (cheap JPEG, no video download)
      diff it against the REFERENCE (the held clean scene)
      blobs must PERSIST at the same spot across confirm_samples consecutive
      comparisons before becoming hits ("maybe a package, HERE")
      optionally: Florence captions the crop -> accepted / rejected

This is the fast/slow idea at sparse cadence: the reference rolls forward
every clean sample (so gradual lighting tracks), but the moment candidate
blobs appear it HOLDS while they are re-checked on the next sample(s). A
package is pixel-stable and confirms; a moving shadow lands somewhere else
each sample, never matches itself, and is dropped. A global-change guard
skips sunrise/sunset flips, and the hold is capped so continuously-moving
light can't pin the reference forever.

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
from .detector import DetectorConfig, _BlobDetectorBase, _iou

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
        """Blobs that APPEARED in cur vs prev; True flag = global flip.

        absdiff is symmetric: a region where something *left* (or where a
        shadow used to be in the held reference) lights up exactly like an
        arrival. Those ghosts are temporally stable, so persistence cannot
        kill them — instead we require the blob's edges to live in the
        CURRENT frame: an object sitting there now has a boundary in `cur`;
        a ghost's boundary exists only in the reference."""
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
        blobs = [b for b in self._extract_blobs(mask, diff)
                 if self._appeared(b[2], cur_gray, prev_gray)]
        return blobs, False

    @staticmethod
    def _appeared(blob_mask: np.ndarray, cur_gray: np.ndarray,
                  ref_gray: np.ndarray) -> bool:
        """True when the blob's boundary edges exist in the current frame."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        band = cv2.morphologyEx(blob_mask, cv2.MORPH_GRADIENT, kernel) > 0
        if not band.any():
            return True
        e_cur = float(np.abs(cv2.Laplacian(cur_gray, cv2.CV_64F))[band].mean())
        e_ref = float(np.abs(cv2.Laplacian(ref_gray, cv2.CV_64F))[band].mean())
        # Keep arrivals (edges now); drop ghosts (edges only in the ref).
        return e_cur >= e_ref * 0.8


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


def _region_still(cur_gray: np.ndarray, prev_gray: np.ndarray,
                  bbox: tuple[int, int, int, int], thresh: int,
                  max_frac: float = 0.3) -> bool:
    """Is the blob's region pixel-stable between consecutive samples?

    This is the 'fast' half of the fast/slow idea at sparse cadence: a
    package's pixels are identical sample-to-sample (modulo noise); a
    creeping shadow's region visibly shifts even when its bounding boxes
    overlap. Only still candidates may accumulate confirmations."""
    x, y, w, h = bbox
    a = cur_gray[y:y + h, x:x + w]
    b = prev_gray[y:y + h, x:x + w]
    if a.size == 0 or a.shape != b.shape:
        return False
    d = cv2.absdiff(a, b)
    return float(np.count_nonzero(d > thresh)) / d.size <= max_frac


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
                 confirm_samples: int = 2,
                 max_hold_samples: int = 4,
                 progress: Optional[Callable[[int, int], None]] = None,
                 on_hit: Optional[Callable[[Hit], None]] = None,
                 ) -> BacktestResult:
    """Sample [start, end] every interval_s and report candidate packages.

    person_windows are seconds relative to `start` (as produced by
    protect.person_windows). Samples inside a window (± pad) are skipped —
    both to avoid captioning people and because the interesting comparison
    is clean-before vs clean-after a visit.

    A blob must reappear at the same spot (IoU vs the reference) AND be
    pixel-stable vs the previous sample for `confirm_samples` consecutive
    samples before it becomes a hit — a package is still, a creeping shadow
    is not. The reference rolls forward EVERY sample so lighting tracks,
    but the pixels under pending candidates are preserved from before their
    arrival (masked absorption) until they confirm or age out after
    `max_hold_samples` — so a real arrival is never swallowed by an
    unrelated moving shadow elsewhere in the frame.
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

    reference: Optional[np.ndarray] = None    # the "slow": clean scene, with
    reference_at: Optional[datetime] = None   # pre-arrival pixels preserved
    prev_gray: Optional[np.ndarray] = None    # the "fast": previous sample
    pending: list[dict] = []                  # candidates awaiting re-check

    def _native_bbox(bbox, pad=6):
        pw, ph = comparer._proc_size
        nw, nh = comparer._native_size
        sx, sy = nw / pw, nh / ph
        x, y, w, h = bbox
        x0 = max(0, int(x * sx) - pad)
        y0 = max(0, int(y * sy) - pad)
        x1 = min(nw, int((x + w) * sx) + pad)
        y1 = min(nh, int((y + h) * sy) + pad)
        return x0, y0, x1, y1

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
        if reference is None:
            reference, reference_at = frame, at
            prev_gray = comparer._prepare(frame)
            continue

        blobs, flipped = comparer.compare(reference, frame)
        cur_gray = comparer._prepare(frame)
        if flipped:
            result.scene_flips += 1
            reference, reference_at = frame, at
            prev_gray = cur_gray
            pending = []
            continue

        # Match blobs against pending candidates. Confirmation needs BOTH:
        # same spot vs the reference (IoU) AND pixel-stability vs the
        # previous sample — a slow-creeping shadow overlaps its own last
        # position, but its pixels shift; a package's do not.
        matched: list[dict] = []
        for bbox, contrast, blob_mask in blobs:
            cand = next((p for p in pending
                         if _iou(bbox, p["bbox"]) > 0.3), None)
            still = (prev_gray is not None and
                     _region_still(cur_gray, prev_gray, bbox,
                                   cfg.diff_threshold))
            if cand is None or not still:
                cand = {"first_at": at, "first_i": i, "seen": 0}
            cand.update(bbox=bbox, contrast=contrast, mask=blob_mask)
            cand["seen"] += 1
            matched.append(cand)
        prev_gray = cur_gray

        confirmed = [c for c in matched if c["seen"] >= confirm_samples]
        for c in confirmed:
            report = comparer._blob_report(
                candidate_id=len(result.hits) + 1, bbox=c["bbox"],
                contrast=c["contrast"], mask=c["mask"],
                baseline=np.empty(0), frame=frame,
                first_seen=c["first_at"].timestamp(), ts=at.timestamp(),
                persisted=c["seen"], triggered=False)
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
                baseline=reference, baseline_at=reference_at,
                verification=verdict)
            result.hits.append(hit)
            if on_hit:
                on_hit(hit)  # lets the UI stream hits while scanning

        # Carry unconfirmed candidates that haven't aged out; confirmed and
        # stale ones absorb into the scene.
        pending = [c for c in matched
                   if c["seen"] < confirm_samples
                   and i - c["first_i"] < max_hold_samples]

        # Masked absorption: the new reference is the current frame, except
        # that pixels under pending candidates keep their pre-arrival state
        # so the same comparison can repeat next sample.
        new_ref = frame.copy()
        for c in pending:
            x0, y0, x1, y1 = _native_bbox(c["bbox"])
            new_ref[y0:y1, x0:x1] = reference[y0:y1, x0:x1]
        reference, reference_at = new_ref, at

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
