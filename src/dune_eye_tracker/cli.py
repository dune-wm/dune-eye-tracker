"""dune-eye-tracker COMMAND: `calibrate` (28 points, once) then `run` (the pointer goes where
you look; your clicks keep it calibrated), `toggle` (start / stop `run`: a key), `game`
(how far off it is, and how well it plays), `share` (the camera for another program
while `run` goes: a key). The lab: `record` (known points looked at,
numbers kept), `eval` (every model on the recordings), `check` (whether the face's pose
holds), `live` (the gaze as you look)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dune-eye-tracker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def camera(p):
        p.add_argument("--device", default=0, type=lambda d: int(d) if d.isdigit() else d, help="camera (default 0: /dev/video0)")

    def window(p):
        camera(p)
        p.add_argument("--no-fullscreen", dest="fullscreen", action="store_false", help="do not ask dune for fullscreen")

    size = lambda s: tuple(int(v) for v in s.split("x"))  # noqa: E731

    calibrate = sub.add_parser("calibrate", help="28 points looked at, on the monitor the pointer is on: the calibration `run` uses")
    window(calibrate)
    calibrate.add_argument("--from", dest="from_rec", metavar="REC.npz", help="no window: a recording's calibration points (record, live C; dune-gaze's too)")
    calibrate.add_argument("--output", help="with --from: the monitor it was on (dune.toml's [output.NAME])")

    from .snap import SNAP
    from .track import GLIDE, HEAD_GAIN

    run = sub.add_parser("run", help="move the pointer where you look, until stopped")
    camera(run)
    run.add_argument("--head-gain", type=float, default=HEAD_GAIN, help=f"px per unit of head turn between landings (default {HEAD_GAIN:g}; 0: the head does not move it)")
    run.add_argument("--glide", type=float, default=GLIDE, help=f"s the pointer takes to glide most of the way (default {GLIDE:g}; 0: no glide)")
    run.add_argument("--snap", type=float, default=SNAP, help=f"px within which a jump lands on a button, link or tab (dune-a11y; default {SNAP:g}; 0: on the gaze)")
    run.add_argument("--share", action="store_true", help="start sharing the camera on the virtual one (v4l2loopback, /dev/video10); `share` turns it on and off")
    run.add_argument("--no-trace", dest="trace", action="store_false", help="do not write why the pointer moved (trace.jsonl: jumps, bounces)")
    run.add_argument("-v", "--verbose", action="store_true", help="fps and moves every second, each click")

    sub.add_parser("toggle", help="start `run` in the background, or stop it (`calibrate` when there is no calibration)")
    sub.add_parser("share", help="the running `run` shares the camera on the virtual one (/dev/video10), or stops: a key")

    game = sub.add_parser("game", help="points to look at (the tracker), then circles to take with the gaze alone (`run` off)")
    window(game)
    game.add_argument("--rounds", type=int, default=4, help="circles of each radius (default 4: 28, 7 a quadrant)")
    game.add_argument("--head-gain", type=float, default=HEAD_GAIN, help=f"as in `run` (default {HEAD_GAIN:g})")
    game.add_argument("--glide", type=float, default=GLIDE, help=f"as in `run` (default {GLIDE:g})")
    game.add_argument("--only", choices=("measure", "play"), help="one round: the points (the tracker) or the circles")
    game.add_argument("--report", action="store_true", help="no window: the games of the current gaze, against the click log")

    record = sub.add_parser("record", help="calibration, head-moving and grid points looked at; numbers saved, for eval (`run` off)")
    window(record)
    record.add_argument("--size", default=(1920, 1080), type=size, help="camera size (default 1920x1080)")
    record.add_argument("--keep-eyes", action="store_true", help="also keep the eye crops (grey images of your eyes), to tune the iris finder on")

    live = sub.add_parser("live", help="the gaze as you look (`run` off): C calibrates here, R adjusts, T tests a point")
    window(live)
    live.add_argument("--from", dest="from_file", metavar="RECORDING.npz", help="fit on this recording's calibration (default: the last at 1080p)")
    live.add_argument("--size", default=(1920, 1080), type=size, help="camera size (default 1920x1080)")
    live.add_argument("--keep-eyes", action="store_true", help="also save the eye crops of the session (grey images of your eyes)")

    ev = sub.add_parser("eval", help="every model fitted and tested on the recordings (default: all of them; games/*.npz too)")
    ev.add_argument("files", nargs="*", help="recordings (.npz)")
    ev.add_argument("--no-cross", dest="cross", action="store_false", help="not across recordings")

    sub.add_parser("trace", help="the pointer's jumps and bounces (there and back) from trace.jsonl, and what came before each")

    ck = sub.add_parser("check", help="whether the face's pose holds: the canonical face put by it against the landmarks")
    ck.add_argument("files", nargs="*", help="recordings (.npz; default: the last)")

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        if args.from_rec:
            from . import gaze

            return gaze.import_session(args.from_rec, args.output)
        from . import record as r

        return r.run(args)
    if args.command == "run":
        from . import track

        return track.run(args)
    if args.command == "toggle":
        from . import track

        return track.toggle(args)
    if args.command == "share":
        from . import track

        return track.share(args)
    if args.command == "game":
        from . import game as g

        return g.run(args)
    if args.command == "record":
        from . import record as r

        return r.run(args)
    if args.command == "live":
        from . import live as lv

        return lv.run(args)
    if args.command == "trace":
        import json

        from .recording import state_dir
        from .trace import LOG, summary

        path = state_dir() / LOG
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
        print("\n".join(summary(lines)))
        return 0
    return lab(args)


def lab(args) -> int:
    """`eval` and `check` on recordings (dune-gaze's too)."""
    from .evaluate import check, report
    from .recording import Recording, all_recordings

    paths = [Path(f) for f in args.files] or all_recordings()
    if not paths:
        print("no recordings yet: dune-eye-tracker record", file=sys.stderr)
        return 1
    if args.command == "check":
        for p in paths if args.files else paths[-1:]:
            print("\n".join(check(Recording.load(p))))
        return 0
    print("\n".join(report([Recording.load(p) for p in paths], args.cross)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
