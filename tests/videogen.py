"""Tiny deterministic video files for pipeline tests.

These exercise the real path (mp4 → OpenCV decode → detector) so plumbing
tests run without committed footage. They deliberately make NO claim about
real-world detection quality — that is what real fixture clips, authored in
the UI and graded by the harness, are for.
"""

from __future__ import annotations

import numpy as np

W, H = 320, 240
PKG = (140, 150, 46, 32)                      # pixel bbox of the box
PKG_REGION = (0.40, 0.58, 0.22, 0.22)         # normalized expectation


def frame(rng: np.random.Generator, package: bool = False,
          person: bool = False) -> np.ndarray:
    f = np.full((H, W, 3), 120, dtype=np.float32)
    f[: H // 3] = 160
    f += rng.normal(0, 3, size=f.shape).astype(np.float32)
    if package:
        x, y, w, h = PKG
        f[y:y + h, x:x + w] = (55, 70, 95)
    if person:
        f[96:204, 150:176] = (28, 28, 30)
    return np.clip(f, 0, 255).astype(np.uint8)


def write_clip(path, phases, fps: float = 2.0, seed: int = 7) -> str:
    """phases: list of (n_frames, dict(package=..., person=...))."""
    import cv2

    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    assert writer.isOpened(), f"cannot open VideoWriter for {path}"
    for n, kw in phases:
        for _ in range(n):
            writer.write(frame(rng, **kw))
    writer.release()
    return str(path)


def empty_clip(path, seconds: float = 20, fps: float = 2.0) -> str:
    return write_clip(path, [(int(seconds * fps), {})], fps)


def package_clip(path, warmup_s: float = 8, hold_s: float = 14,
                 fps: float = 2.0) -> str:
    return write_clip(path, [
        (int(warmup_s * fps), {}),
        (int(hold_s * fps), {"package": True})], fps)


def delivery_clip(path, warmup_s: float = 8, visit_s: float = 6,
                  tail_s: float = 12, fps: float = 2.0) -> str:
    """empty → person visits (box lands mid-visit) → box alone.
    Person is in frame for [warmup_s, warmup_s + visit_s]."""
    half = int(visit_s * fps) // 2
    return write_clip(path, [
        (int(warmup_s * fps), {}),
        (half, {"person": True}),
        (int(visit_s * fps) - half, {"person": True, "package": True}),
        (int(tail_s * fps), {"package": True})], fps)
