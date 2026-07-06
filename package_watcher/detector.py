"""Static new-object detection with a dual background model.

The idea, in one paragraph: keep two exponential running averages of the
scene — a *fast* background that adapts within a few samples, and a *slow*
background that takes minutes to change. A person walking through differs
from both models (moving object). A package that was set down differs from
the slow model but is quickly absorbed by the fast one. So the mask

    static_new = (|frame - slow_bg| > T)  AND  NOT (|frame - fast_bg| > T)

lights up precisely for things that appeared recently *and stopped moving*.
Blobs in that mask are tracked across samples; one that persists for N
consecutive samples is reported as a candidate package, together with the
evidence needed to verify the call.

Everything is plain numpy/OpenCV arithmetic on downscaled grayscale frames —
no neural networks, comfortably CPU-bound at ~1 fps per camera.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    resize_width: int = 640          # processing resolution (aspect preserved)
    blur_kernel: int = 5             # gaussian blur to suppress sensor noise
    diff_threshold: int = 18         # grayscale delta considered "different"
    fast_alpha: float = 0.15         # fast background learning rate per sample
    slow_alpha: float = 0.004        # slow background learning rate per sample
    min_area_frac: float = 0.0008    # ignore blobs smaller than this
    max_area_frac: float = 0.25      # ignore blobs bigger than this (lighting)
    persist_samples: int = 8         # samples a blob must survive to report
    persist_samples_triggered: int = 4  # lower bar inside an attention window
    miss_limit: int = 6              # drop a candidate after this many misses
    heal_after_reported: int = 20    # samples after report before re-arming
    global_change_frac: float = 0.35  # scene-change guard (lights on/off, PTZ)
    match_iou: float = 0.2           # IoU to match a blob to a known candidate


@dataclass
class Candidate:
    """A blob we are watching to see whether it persists."""

    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in processing coordinates
    first_seen: float
    hits: int = 1
    misses: int = 0
    reported: bool = False
    hits_since_report: int = 0
    contrast: float = 0.0
    baseline: Optional[np.ndarray] = None  # slow bg snapshot when first seen
    mask: Optional[np.ndarray] = None      # latest blob mask (proc resolution)


@dataclass
class NewObjectReport:
    """A detection, expressed in the native coordinate plane of the camera."""

    candidate_id: int
    bbox: tuple[int, int, int, int]           # native pixels
    bbox_norm: tuple[float, float, float, float]
    frame_size: tuple[int, int]               # native (width, height)
    area_fraction: float
    contrast: float                            # 0..1 mean |diff| inside blob
    confidence: float                          # 0..1 heuristic, see below
    first_seen: float
    reported_at: float
    samples_persisted: int
    triggered: bool
    # Evidence, all at processing resolution unless noted:
    frame: np.ndarray = field(repr=False)      # native frame at report time
    mask: np.ndarray = field(repr=False)       # binary blob mask
    baseline: np.ndarray = field(repr=False)   # slow bg before object appeared


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    return inter / float(aw * ah + bw * bh - inter)


class StaticObjectDetector:
    """Per-camera detector. Feed it sampled frames; it returns reports."""

    def __init__(self, config: DetectorConfig | None = None,
                 zone: Optional[list[tuple[float, float]]] = None):
        self.cfg = config or DetectorConfig()
        self._zone_norm = zone  # normalized polygon, or None = whole frame
        self._fast: Optional[np.ndarray] = None
        self._slow: Optional[np.ndarray] = None
        self._zone_mask: Optional[np.ndarray] = None
        self._proc_size: Optional[tuple[int, int]] = None  # (w, h)
        self._native_size: Optional[tuple[int, int]] = None
        self._candidates: list[Candidate] = []
        self._ids = itertools.count(1)
        self.samples_seen = 0
        self.scene_resets = 0

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, ts: Optional[float] = None,
                attention: bool = False) -> list[NewObjectReport]:
        """Process one sampled frame; return zero or more new-object reports.

        `attention` marks samples inside a trigger window (e.g. Unifi just
        saw a person) — the persistence bar is lowered because a delivery is
        more plausible right after someone was at the door.
        """
        ts = time.time() if ts is None else ts
        gray = self._prepare(frame_bgr)
        self.samples_seen += 1

        if self._fast is None:
            self._fast = gray.astype(np.float32)
            self._slow = gray.astype(np.float32)
            return []

        diff_slow = cv2.absdiff(gray, self._slow.astype(np.uint8))
        diff_fast = cv2.absdiff(gray, self._fast.astype(np.uint8))

        # Scene-change guard: if most of the frame differs from the slow
        # model (lights toggled, camera bumped, exposure jump), re-seed the
        # backgrounds instead of reporting the whole world as new.
        changed_frac = float(np.count_nonzero(
            diff_slow > self.cfg.diff_threshold)) / diff_slow.size
        if changed_frac > self.cfg.global_change_frac:
            self._fast = gray.astype(np.float32)
            self._slow = gray.astype(np.float32)
            self._candidates.clear()
            self.scene_resets += 1
            return []

        cv2.accumulateWeighted(gray, self._fast, self.cfg.fast_alpha)
        cv2.accumulateWeighted(gray, self._slow, self.cfg.slow_alpha)

        static_new = cv2.bitwise_and(
            (diff_slow > self.cfg.diff_threshold).astype(np.uint8) * 255,
            (diff_fast <= self.cfg.diff_threshold).astype(np.uint8) * 255,
        )
        if self._zone_mask is not None:
            static_new = cv2.bitwise_and(static_new, self._zone_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        static_new = cv2.morphologyEx(static_new, cv2.MORPH_OPEN, kernel)
        static_new = cv2.morphologyEx(static_new, cv2.MORPH_CLOSE, kernel)

        blobs = self._extract_blobs(static_new, diff_slow)
        reports = self._track(blobs, gray, ts, attention)
        for report in reports:
            report.frame = frame_bgr
        return reports

    # ------------------------------------------------------------------
    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._native_size = (w, h)
        if w > self.cfg.resize_width:
            scale = self.cfg.resize_width / w
            frame_bgr = cv2.resize(
                frame_bgr, (self.cfg.resize_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA)
        ph, pw = frame_bgr.shape[:2]
        if self._proc_size != (pw, ph):
            self._proc_size = (pw, ph)
            self._fast = self._slow = None  # resolution changed; re-seed
            self._zone_mask = self._build_zone_mask(pw, ph)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = self.cfg.blur_kernel
        if k >= 3:
            gray = cv2.GaussianBlur(gray, (k | 1, k | 1), 0)
        return gray

    def _build_zone_mask(self, w: int, h: int) -> Optional[np.ndarray]:
        if not self._zone_norm:
            return None
        pts = np.array([[int(x * w), int(y * h)] for x, y in self._zone_norm],
                       dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _extract_blobs(self, mask: np.ndarray, diff_slow: np.ndarray
                       ) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        """Return (bbox, contrast, blob_mask) for each plausible blob."""
        total = mask.shape[0] * mask.shape[1]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            frac = area / total
            if frac < self.cfg.min_area_frac or frac > self.cfg.max_area_frac:
                continue
            blob = (labels == i)
            contrast = float(diff_slow[blob].mean()) / 255.0
            out.append(((int(x), int(y), int(w), int(h)), contrast,
                        blob.astype(np.uint8) * 255))
        return out

    def _track(self, blobs, gray: np.ndarray, ts: float,
               attention: bool) -> list[NewObjectReport]:
        cfg = self.cfg
        matched: set[int] = set()
        reports: list[NewObjectReport] = []

        for bbox, contrast, blob_mask in blobs:
            best, best_iou = None, cfg.match_iou
            for cand in self._candidates:
                if cand.id in matched:
                    continue
                iou = _iou(bbox, cand.bbox)
                if iou > best_iou:
                    best, best_iou = cand, iou
            if best is None:
                cand = Candidate(
                    id=next(self._ids), bbox=bbox, first_seen=ts,
                    contrast=contrast,
                    baseline=self._slow.astype(np.uint8).copy(),
                    mask=blob_mask)
                self._candidates.append(cand)
                matched.add(cand.id)
                continue
            matched.add(best.id)
            best.bbox = bbox
            best.hits += 1
            best.misses = 0
            best.contrast = contrast
            best.mask = blob_mask
            if best.reported:
                best.hits_since_report += 1
                if best.hits_since_report >= cfg.heal_after_reported:
                    self._heal(best, gray)
                continue
            needed = (cfg.persist_samples_triggered if attention
                      else cfg.persist_samples)
            if best.hits >= needed:
                best.reported = True
                reports.append(self._make_report(best, ts, attention))

        survivors = []
        for cand in self._candidates:
            if cand.id in matched:
                if cand.hits > 0:  # healed candidates get hits set to -1
                    survivors.append(cand)
                continue
            cand.misses += 1
            if cand.misses <= cfg.miss_limit:
                survivors.append(cand)
        self._candidates = survivors
        return reports

    def _heal(self, cand: Candidate, gray: np.ndarray) -> None:
        """Absorb a long-reported object into the slow background so the
        detector re-arms for the *next* object in the same spot."""
        x, y, w, h = cand.bbox
        self._slow[y:y + h, x:x + w] = gray[y:y + h, x:x + w].astype(np.float32)
        cand.hits = -1  # mark for removal in _track's survivor pass

    def _make_report(self, cand: Candidate, ts: float,
                     attention: bool) -> NewObjectReport:
        pw, ph = self._proc_size
        nw, nh = self._native_size
        sx, sy = nw / pw, nh / ph
        x, y, w, h = cand.bbox
        native_bbox = (int(round(x * sx)), int(round(y * sy)),
                       int(round(w * sx)), int(round(h * sy)))
        area_frac = (w * h) / float(pw * ph)
        # Confidence heuristic (documented, not learned): persistence is
        # already satisfied, so blend contrast (how different the region is
        # from the pre-object background) with a mild size prior that favors
        # package-sized blobs over specks or half-frame lighting artifacts.
        size_prior = 1.0 - min(1.0, abs(np.log10(max(area_frac, 1e-6) / 0.01)) / 2.5)
        confidence = float(np.clip(0.35 + 0.45 * min(1.0, cand.contrast * 4)
                                   + 0.20 * size_prior, 0.0, 1.0))
        return NewObjectReport(
            candidate_id=cand.id,
            bbox=native_bbox,
            bbox_norm=(native_bbox[0] / nw, native_bbox[1] / nh,
                       native_bbox[2] / nw, native_bbox[3] / nh),
            frame_size=(nw, nh),
            area_fraction=area_frac,
            contrast=cand.contrast,
            confidence=confidence,
            first_seen=cand.first_seen,
            reported_at=ts,
            samples_persisted=cand.hits,
            triggered=attention,
            frame=np.empty(0),   # filled in by process() with the native frame
            mask=cand.mask if cand.mask is not None else np.empty(0),
            baseline=cand.baseline if cand.baseline is not None else np.empty(0),
        )
