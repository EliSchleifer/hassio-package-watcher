"""Semantic verification of candidate crops with a local vision model.

The CV pipeline (diff → blobs → shape priors) can only say "something new
and static is here". This stage answers *what is it*: it crops each
candidate region (with context margin), captions it with Florence-2 running
locally on CPU, and decides from the caption whether it is a delivered item
(package / box / parcel / …) or noise (shadow, reflection, cat).

In fixtures a human plays this role by drawing the expected region; live
there is no labeler, so the model is what separates "cardboard box on the
step" from "light artifact on the wall".

Heavy dependencies (torch / transformers) are imported lazily and only when
`backend: florence` is configured — the core watcher never pays for them.
The model itself is loaded on first use and kept for the process lifetime;
a few seconds per crop on CPU is fine at delivery frequency.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from .config import VerifierConfig

log = logging.getLogger(__name__)

# Captions we may see for non-deliveries, used only to give the verdict a
# more useful label than "other" — rejection is driven by the accept list.
_KNOWN_OTHERS = ("person", "newspaper", "cat", "dog", "shadow",
                 "car", "plant", "chair", "bicycle", "door", "wall",
                 "ladder", "railing")


def crop_with_margin(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                     margin: float) -> np.ndarray:
    """Crop bbox plus a context margin (fraction of bbox size), clamped."""
    fh, fw = frame_bgr.shape[:2]
    x, y, w, h = bbox
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
    return frame_bgr[y0:y1, x0:x1]


def decide(caption: str, accept: list[str]) -> dict[str, Any]:
    """Turn a caption into a verdict. Pure, so it is trivially testable.

    Matches whole words only — 'cat' must not match inside 'scattered'."""
    import re

    lowered = caption.lower()

    def has_word(w: str) -> bool:
        return re.search(rf"\b{re.escape(w.lower())}\b", lowered) is not None

    for word in accept:
        if has_word(word):
            return {"accepted": True, "label": word, "caption": caption}
    label = next((w for w in _KNOWN_OTHERS if has_word(w)), "other")
    return {"accepted": False, "label": label, "caption": caption}


class FlorenceVerifier:
    """Caption candidate crops with Florence-2 (local, CPU)."""

    TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._sam = None
        self._sam_processor = None
        self._lock = threading.Lock()

    # -- model plumbing --------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch  # noqa: F401
                from transformers import (AutoProcessor,
                                          Florence2ForConditionalGeneration)
            except ImportError as exc:
                raise RuntimeError(
                    "verifier backend 'florence' needs the [verify] extra "
                    "(and transformers>=4.54 with native Florence-2): "
                    "pip install 'package-watcher[verify]'") from exc
            log.info("loading %s (first use; this can take a minute)",
                     self.cfg.model)
            kwargs: dict[str, Any] = {}
            if self.cfg.cache_dir:
                kwargs["cache_dir"] = self.cfg.cache_dir
            # Native transformers support — no trust_remote_code needed.
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.model, **kwargs)
            self._model = Florence2ForConditionalGeneration.from_pretrained(
                self.cfg.model, **kwargs).eval()
            log.info("verifier model ready")

    def _generate(self, crop_bgr: np.ndarray, task: str,
                  extra_text: str = ""):
        import torch
        from PIL import Image

        self._ensure_loaded()
        image = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR -> RGB
        inputs = self._processor(text=task + extra_text, images=image,
                                 return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(
                **inputs, max_new_tokens=64, num_beams=1, do_sample=False)
        raw = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(
            raw, task=task, image_size=(image.width, image.height))

    def _caption(self, crop_bgr: np.ndarray) -> str:
        """One caption for one crop. Overridable in tests."""
        parsed = self._generate(crop_bgr, self.TASK)
        return str(parsed.get(self.TASK, "")).strip()

    # NOTE on giving Florence "context": its captioner takes no instructions,
    # and its open-vocabulary grounding task is NOT usable as an accept
    # signal — validated on real footage, it "finds" the requested package
    # phrase in doors, ladders, and bare walls. Context lives in the accept
    # vocabulary instead; a true instruction-following VLM backend (Ollama /
    # Claude) is the upgrade path if caption words ever stop being enough.

    def _mask_bbox(self, frame_bgr: np.ndarray,
                   bbox: tuple[int, int, int, int]
                   ) -> Optional[tuple[int, int, int, int]]:
        """SAM 2 crop refinement: prompt the segmenter with the loose diff
        bbox, return the actual object boundary's bbox. None on any failure
        (missing deps, empty mask) — the plain crop is always a safe
        fallback. Overridable in tests."""
        try:
            import torch
            from PIL import Image
            from transformers import Sam2Model, Sam2Processor

            with self._lock:
                if self._sam is None:
                    kwargs = ({"cache_dir": self.cfg.cache_dir}
                              if self.cfg.cache_dir else {})
                    self._sam_processor = Sam2Processor.from_pretrained(
                        self.cfg.refine_model, **kwargs)
                    self._sam = Sam2Model.from_pretrained(
                        self.cfg.refine_model, **kwargs).eval()
            x, y, w, h = bbox
            image = Image.fromarray(frame_bgr[:, :, ::-1])
            inputs = self._sam_processor(
                images=image, input_boxes=[[[x, y, x + w, y + h]]],
                return_tensors="pt")
            with torch.no_grad():
                out = self._sam(**inputs, multimask_output=False)
            mask = self._sam_processor.post_process_masks(
                out.pred_masks, inputs["original_sizes"])[0][0, 0]
            mask = mask.numpy().astype(bool)
            if not mask.any():
                return None
            ys, xs = np.nonzero(mask)
            return (int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        except Exception as exc:  # noqa: BLE001 - refinement is optional
            log.debug("sam2 refinement unavailable: %s", exc)
            return None

    # -- public API -------------------------------------------------------
    def verify(self, frame_bgr: np.ndarray,
               bbox: tuple[int, int, int, int]) -> dict[str, Any]:
        """Caption the (margined) crop and decide whether it's a delivery.

        With refine=sam2 the crop is fitted to the segmented object boundary
        first — measured on real footage this yields cleaner captions (the
        real background stays in frame; removing it made Florence read
        objects as '3D renderings')."""
        refined = None
        if self.cfg.refine == "sam2":
            refined = self._mask_bbox(frame_bgr, bbox)
        crop = (crop_with_margin(frame_bgr, refined, 0.2)
                if refined else
                crop_with_margin(frame_bgr, bbox, self.cfg.crop_margin))
        if crop.size == 0:
            return {"accepted": False, "label": "other", "caption": ""}
        caption = self._caption(crop)
        verdict = decide(caption, self.cfg.accept)
        verdict["backend"] = "florence"
        verdict["model"] = self.cfg.model
        verdict["refined"] = refined is not None
        return verdict


def build_verifier(cfg: VerifierConfig) -> Optional[FlorenceVerifier]:
    """Instantiate the configured verifier, or None when off."""
    if cfg.backend == "off":
        return None
    if cfg.backend == "florence":
        return FlorenceVerifier(cfg)
    raise ValueError(
        f"unknown verifier backend {cfg.backend!r}; choices: 'off', 'florence'")
