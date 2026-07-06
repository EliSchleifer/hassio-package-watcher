"""package_watcher — CPU-only new-object detection for fixed cameras.

Watches RTSP streams from fixed (Unifi Protect) cameras and reports when
something new shows up in frame and stays there — the signature of a package
being dropped off. Every report includes the coordinates of the region and an
evidence bundle (annotated frame, crop, diff mask, baseline) so a downstream
LLM can verify the finding.
"""

__version__ = "0.1.0"
