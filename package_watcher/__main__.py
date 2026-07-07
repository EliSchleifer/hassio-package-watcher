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

    bt = sub.add_parser(
        "backtest",
        help="scan a camera's recorded history every X minutes for "
             "candidate packages (people skipped via Protect events)")
    bt.add_argument("--config", "-c", required=True,
                    help="config with a unifi block")
    bt.add_argument("--camera", required=True, help="Protect camera id or name")
    bt.add_argument("--date", required=True, help="day to scan, YYYY-MM-DD (local)")
    bt.add_argument("--interval", type=float, default=10.0,
                    help="minutes between samples (default 10)")
    bt.add_argument("--verify", action="store_true",
                    help="run the vision-model second stage on candidates")
    bt.add_argument("--out", help="directory to save annotated hit frames")

    ui = sub.add_parser(
        "ui", help="launch the fixture-authoring web UI")
    ui.add_argument("--config", "-c", help="config with a unifi block "
                    "(enables pulling clips from Protect cameras)")
    ui.add_argument("--fixtures", default="fixtures",
                    help="fixtures directory to read/write")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8080)
    ui.add_argument("--reload", action="store_true",
                    help="auto-restart on code edits (dev; needs an editable "
                         "install) so you can iterate without rebuilding")
    ui.add_argument("--setup", action="store_true",
                    help="prompt for UniFi Protect credentials and write them "
                         "to --config (default config.yaml), then serve")
    return parser


def _setup_unifi(path: str) -> None:
    """Interactively write a UniFi Protect `unifi:` config for the UI.

    Credentials are typed into the local terminal and written straight to the
    file — nothing is echoed back. Leave the host blank to skip (you can
    still upload clips).
    """
    import getpass

    import yaml

    print(f"Set up UniFi Protect credentials (written to {path}).\n"
          f"Pulling recorded footage uses Protect's private API, which needs a "
          f"local Protect account (username + password). An API key is optional "
          f"— it only unlocks the newer public API, not clip export.\n"
          f"Leave the host blank to skip.", file=sys.stderr)
    host = input("  NVR host / IP: ").strip()
    if not host:
        print("  skipped.", file=sys.stderr)
        return
    username = input("  username: ").strip()
    password = getpass.getpass("  password: ").strip()
    api_key = getpass.getpass("  API key (optional, Enter to skip): ").strip()
    if not (username and password):
        print("  warning: without a username + password, recorded-clip pull "
               "and scrubbing will not work (API key alone is public-only).",
               file=sys.stderr)
    block: dict = {"host": host, "verify_ssl": False}
    if username:
        block["username"] = username
    if password:
        block["password"] = password
    if api_key:
        block["api_key"] = api_key
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"unifi": block}, f, sort_keys=False)
    print(f"  wrote {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr)

    if args.command == "test":
        return _run_tests(args)
    if args.command == "backtest":
        return _run_backtest(args)
    if args.command == "ui":
        import os
        from .ui.app import serve
        cfg_path = args.config or "config.yaml"
        if args.setup:
            _setup_unifi(cfg_path)
        unifi = verifier = zones_path = None
        if os.path.isfile(cfg_path):
            cfg = load_config(cfg_path, require_cameras=False)
            unifi, verifier = cfg.unifi, cfg.verifier
            # Zones live next to the config so the live watcher sees them.
            zones_path = os.path.join(
                os.path.dirname(os.path.abspath(cfg_path)), "zones.yaml")
        serve(fixtures_dir=args.fixtures, unifi=unifi,
              host=args.host, port=args.port, reload=args.reload,
              verifier_cfg=verifier, zones_path=zones_path)
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


def _run_backtest(args) -> int:
    import os
    from datetime import datetime, timedelta

    from .backtest import protect_snapshot_fn, run_backtest
    from .ui import protect
    from .verify import build_verifier

    cfg = load_config(args.config, require_cameras=False)
    if cfg.unifi is None:
        print("backtest needs a unifi block in the config", file=sys.stderr)
        return 2

    # Resolve the camera by id or display name.
    cameras = protect.list_cameras(cfg.unifi)
    cam = next((c for c in cameras
                if c["id"] == args.camera or c["name"] == args.camera), None)
    if cam is None:
        print(f"no Protect camera {args.camera!r}; have: "
              f"{', '.join(c['name'] for c in cameras)}", file=sys.stderr)
        return 2

    start = datetime.fromisoformat(args.date).astimezone()
    end = start + timedelta(days=1)
    try:
        windows = protect.person_windows(cfg.unifi, cam["id"], start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: person events unavailable ({exc})", file=sys.stderr)
        windows = []
    verifier = None
    if args.verify:
        vcfg = cfg.verifier
        if vcfg.backend == "off":
            vcfg.backend = "florence"
        verifier = build_verifier(vcfg)

    def progress(i: int, n: int) -> None:
        print(f"\r  sample {i}/{n}", end="", file=sys.stderr, flush=True)

    result = run_backtest(
        protect_snapshot_fn(cfg.unifi, cam["id"]), cam["id"], start, end,
        interval_s=args.interval * 60.0, person_windows=windows,
        verifier=verifier, progress=progress)
    print("", file=sys.stderr)

    print(f"{result.samples_total} samples, "
          f"{result.samples_skipped_person} skipped (person present), "
          f"{result.samples_missing} missing, "
          f"{result.scene_flips} lighting flips, "
          f"{len(result.hits)} candidate(s)\n")
    for i, hit in enumerate(result.hits, 1):
        line = (f"{hit.at.astimezone():%H:%M}  "
                f"norm=({', '.join(f'{v:.3f}' for v in hit.bbox_norm)})  "
                f"conf={hit.confidence:.2f}")
        if hit.verification and "caption" in hit.verification:
            mark = "ACCEPT" if hit.verification["accepted"] else "reject"
            line += f'  [{mark}] "{hit.verification["caption"]}"'
        print(line)
        if args.out:
            import cv2
            os.makedirs(args.out, exist_ok=True)
            img = hit.frame.copy()
            x, y, w, h = hit.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (60, 220, 60), 2)
            cv2.imwrite(os.path.join(
                args.out, f"{hit.at:%H%M}-{i}.jpg"), img)
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
