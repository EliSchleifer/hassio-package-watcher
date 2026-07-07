"""package_watcher — find things in the frame that weren't there before.

THE GOAL (the north star — read this before changing the pipeline)
===================================================================

This is a **cron-cadence detector**: every so often (or when something
interesting happens), compare the scene NOW against a **defined "before"**
and report what appeared. "Before" is a choice, not a constant:

  - a TRIGGER defines it best: a person appears and leaves — that visit
    brackets a (before, after) pair of clean frames, and whatever differs
    between them is what the visit left behind;
  - absent a trigger, "before" is simply the previous sample (30 minutes
    ago, 60 seconds ago — whatever the cadence is), with lighting tracked
    so time-of-day never masquerades as an object.

Nothing here is latency-sensitive. Quality beats speed at every stage:
native-resolution frames, model verification measured in seconds — a
delivery sits on the porch for hours.

THE PIPELINE (division of labor — each stage does the one thing it's best at)
=============================================================================

  trigger            Protect person events (live websocket, or event history
                     for backtests) say WHEN to compare and what to skip.

  localize (OpenCV)  A pixel diff between before/after finds WHERE changed —
                     O(pixels), lighting-tracked reference with masked
                     absorption, stillness + ghost + shape + zone gates.

  delineate (SAM 2)  Prompted with the changed region, the segmenter returns
                     the object's true boundary -> tight crops.

  name (Florence-2)  The captioner says WHAT the crop is; caption keywords
                     decide package vs noise.

WHY NOT LET ONE MODEL DO EVERYTHING (asked, measured, answered)
===============================================================

  - SAM 2 cannot replace the diff: unprompted "segment everything and find
    new segments" needs ~1000 grid prompts per frame (minutes on CPU vs
    milliseconds for the diff) and segment sets are not stable across
    lighting — matching them between frames is brittle. SAM is superb at
    "what is the extent of the thing HERE"; the diff supplies the HERE.
  - SAM's objectness scores cannot gate candidates: a door panel segments
    as confidently (0.93) as a package (0.95); segmentability is not
    deliveredness.
  - Florence's grounding cannot confirm candidates: asked to find "a
    delivered package", it obliges on doors, ladders, and bare walls.
  - Florence on background-removed crops hallucinates ("3D rendering");
    tight crops with the real background kept measured best.

All of the above were validated against real labeled clips in fixtures/ —
extend that set before trusting any new idea.
"""

__version__ = "0.1.0"
