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

    test = sub.add_parser(
        "test", help="run fixture cases and report detect/no-detect grades")
    test.add_argument("--fixtures", default="fixtures",
                      help="fixtures directory containing cases.yaml")
    test.add_argument("--name", help="run only the case with this name")

    ui = sub.add_parser(
        "ui", help="launch the fixture-authoring web UI")
    ui.add_argument("--config", "-c", help="config with a unifi block "
                    "(enables pulling clips from Protect cameras)")
    ui.add_argument("--fixtures", default="fixtures",
                    help="fixtures directory to read/write")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr)

    if args.command == "test":
        return _run_tests(args)
    if args.command == "ui":
        from .ui.app import serve
        unifi = load_config(args.config).unifi if args.config else None
        serve(fixtures_dir=args.fixtures, unifi=unifi,
              host=args.host, port=args.port)
        return 0

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


def _run_tests(args) -> int:
    import os
    from .harness import load_cases, run_and_evaluate

    manifest = os.path.join(args.fixtures, "cases.yaml")
    cases = load_cases(manifest)
    if args.name:
        cases = [c for c in cases if c.name == args.name]
        if not cases:
            print(f"no case named {args.name!r}", file=sys.stderr)
            return 2

    passed = failed = skipped = 0
    for case in cases:
        if case.clip:
            path = (case.clip if os.path.isabs(case.clip)
                    else os.path.join(args.fixtures, case.clip))
            if not os.path.isfile(path):
                print(f"SKIP {case.name}: clip not present ({case.clip})")
                skipped += 1
                continue
        outcome = run_and_evaluate(case, args.fixtures)
        mark = "PASS" if outcome.passed else "FAIL"
        print(f"{mark} {case.name}: {outcome.reason}")
        passed += outcome.passed
        failed += not outcome.passed
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
