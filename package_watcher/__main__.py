"""CLI: `package-watcher run --config ...` or `package-watcher analyze clip.mp4`."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import AppConfig, CameraConfig, SinkConfig, load_config
from .service import WatcherService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="package-watcher",
        description="Detect new stationary objects (packages) on fixed cameras.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="watch cameras from a config file")
    run.add_argument("--config", "-c", required=True, help="path to YAML config")

    analyze = sub.add_parser(
        "analyze", help="one-shot analysis of a recorded clip (for testing)")
    analyze.add_argument("video", help="path to a video file")
    analyze.add_argument("--fps", type=float, default=2.0,
                         help="sampling rate in frames/sec of video time")
    analyze.add_argument("--out", default="./events",
                         help="directory for evidence bundles")
    analyze.add_argument("--persist", type=int, default=6,
                         help="samples an object must persist before reporting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr)

    if args.command == "run":
        config = load_config(args.config)
    else:  # analyze
        from .detector import DetectorConfig
        config = AppConfig(
            cameras=[CameraConfig(name="clip", source=args.video,
                                  sample_fps=args.fps)],
            detector=DetectorConfig(persist_samples=args.persist,
                                    persist_samples_triggered=args.persist),
            sinks=SinkConfig(stdout=True),
            events_dir=args.out)

    WatcherService(config).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
