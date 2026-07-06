"""Synthetic camera footage for deterministic integration tests.

Real delivery clips are the gold standard, but they are big binaries and
they are not reproducible in CI. These scenarios render a fixed 'porch'
scene with controllable events (a package dropped, a person walking
through, lighting drift, a light switch) so the fixture suite can assert
detect / no-detect behavior without any camera, network, or committed
video. Everything is seeded, so a scenario renders identically every run.

Each scenario returns a list of *samples* — one frame per detector sample —
so the harness can treat sample `i` as occurring at `i / fps` seconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Package region as a normalized (x, y, w, h) box — the porch stoop.
DEFAULT_PACKAGE_REGION = (0.44, 0.62, 0.14, 0.13)
_CARDBOARD = (55, 70, 95)      # BGR, a dark parcel
_PERSON = (28, 28, 30)         # BGR, a dark figure


def _base(rng: np.random.Generator, size: tuple[int, int],
          brightness: float = 0.0) -> np.ndarray:
    w, h = size
    frame = np.full((h, w, 3), 120, dtype=np.float32)
    frame[: h // 3] = 160                       # a brighter "sky" band
    frame += brightness
    frame += rng.normal(0, 3, size=(h, w, 3)).astype(np.float32)
    return frame


def _fill(frame: np.ndarray, region_norm: tuple[float, float, float, float],
          color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    x = int(region_norm[0] * w)
    y = int(region_norm[1] * h)
    bw = max(1, int(region_norm[2] * w))
    bh = max(1, int(region_norm[3] * h))
    frame[y:y + bh, x:x + bw] = color


def _done(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame, 0, 255).astype(np.uint8)


def _n(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def render(spec: dict[str, Any]) -> list[np.ndarray]:
    """Render a scenario dict into a list of frames (one per sample).

    Recognized keys: scene, fps, size, warmup_s, event_s, region, plus
    a few scene-specific knobs. Unknown scenes raise ValueError so a typo
    in a fixture manifest fails loudly instead of silently passing.
    """
    scene = spec.get("scene")
    fps = float(spec.get("fps", 2.0))
    size = tuple(spec.get("size", (320, 240)))  # type: ignore[assignment]
    seed = int(spec.get("seed", 1234))
    rng = np.random.default_rng(seed)
    region = tuple(spec.get("region", DEFAULT_PACKAGE_REGION))  # type: ignore

    builder = _SCENES.get(scene)
    if builder is None:
        raise ValueError(
            f"unknown synthetic scene {scene!r}; "
            f"choices: {sorted(_SCENES)}")
    return builder(rng, size, fps, region, spec)


# --- scenarios ---------------------------------------------------------

def _empty(rng, size, fps, region, spec) -> list[np.ndarray]:
    seconds = float(spec.get("seconds", 30))
    return [_done(_base(rng, size)) for _ in range(_n(seconds, fps))]


def _package(rng, size, fps, region, spec) -> list[np.ndarray]:
    warmup = _n(spec.get("warmup_s", 12), fps)
    hold = _n(spec.get("hold_s", 15), fps)
    frames = [_done(_base(rng, size)) for _ in range(warmup)]
    for _ in range(hold):
        f = _base(rng, size)
        _fill(f, region, _CARDBOARD)
        frames.append(_done(f))
    return frames


def _person_only(rng, size, fps, region, spec) -> list[np.ndarray]:
    warmup = _n(spec.get("warmup_s", 10), fps)
    cross = _n(spec.get("cross_s", 6), fps)
    tail = _n(spec.get("tail_s", 10), fps)
    w, h = size
    frames = [_done(_base(rng, size)) for _ in range(warmup)]
    for i in range(cross):
        f = _base(rng, size)
        x = int((0.05 + 0.85 * (i / max(1, cross - 1))) * w)
        pw = int(0.08 * w)
        f[int(0.4 * h):int(0.85 * h), x:x + pw] = _PERSON
        frames.append(_done(f))
    frames += [_done(_base(rng, size)) for _ in range(tail)]
    return frames


def _person_then_package(rng, size, fps, region, spec) -> list[np.ndarray]:
    """A person walks up, leaves, and a package remains — the classic
    delivery sequence. Use with an attention window over the person."""
    frames = _person_only(rng, size, fps, region,
                          {"warmup_s": spec.get("warmup_s", 10),
                           "cross_s": spec.get("cross_s", 6),
                           "tail_s": spec.get("gap_s", 3)})
    hold = _n(spec.get("hold_s", 15), fps)
    for _ in range(hold):
        f = _base(rng, size)
        _fill(f, region, _CARDBOARD)
        frames.append(_done(f))
    return frames


def _lighting_drift(rng, size, fps, region, spec) -> list[np.ndarray]:
    n = _n(spec.get("seconds", 40), fps)
    rate = float(spec.get("rate", 1.0))  # brightness units per sample
    return [_done(_base(rng, size, brightness=i * rate)) for i in range(n)]


def _light_switch(rng, size, fps, region, spec) -> list[np.ndarray]:
    before = _n(spec.get("before_s", 10), fps)
    after = _n(spec.get("after_s", 20), fps)
    delta = float(spec.get("delta", 80))
    frames = [_done(_base(rng, size)) for _ in range(before)]
    frames += [_done(_base(rng, size, brightness=delta)) for _ in range(after)]
    return frames


def _shadow_sweep(rng, size, fps, region, spec) -> list[np.ndarray]:
    """A soft shadow band (cloud/tree) sweeps across a porch that is clean at
    the start and end. It differs from the backgrounds but never dwells in one
    place, so it must NOT be reported. The clean warmup seeds the background
    with the true (unshadowed) scene."""
    w, h = size
    warmup = _n(spec.get("warmup_s", 6), fps)
    tail = _n(spec.get("tail_s", 6), fps)
    sweep = _n(spec.get("sweep_s", 8), fps)
    band = int(0.22 * w)
    frames = [_done(_base(rng, size)) for _ in range(warmup)]
    for i in range(sweep):
        f = _base(rng, size)
        # Travel from fully off-screen left to fully off-screen right in big
        # steps, so no column stays shadowed long enough to look "static".
        x = int(-band + (i / max(1, sweep - 1)) * (w + band))
        x0, x1 = max(0, x), min(w, x + band)
        if x1 > x0:
            f[:, x0:x1] -= 35
        frames.append(_done(f))
    frames += [_done(_base(rng, size)) for _ in range(tail)]
    return frames


_SCENES = {
    "empty": _empty,
    "package": _package,
    "person_only": _person_only,
    "person_then_package": _person_then_package,
    "lighting_drift": _lighting_drift,
    "light_switch": _light_switch,
    "shadow_sweep": _shadow_sweep,
}


def write_clip(path: str, spec: dict[str, Any]) -> str:
    """Render a scenario to an .mp4 (for seeding the UI / manual review)."""
    import cv2

    frames = render(spec)
    h, w = frames[0].shape[:2]
    fps = float(spec.get("fps", 2.0))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
