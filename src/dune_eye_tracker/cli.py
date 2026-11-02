"""dune-eye-tracker COMMAND: `calibrate` (and `calibrate-head`) then `run` (the pointer goes where you look),
`toggle` (start / stop `run`: a key), `measure` (calibrate, test, report), `analyze RUN.json`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dune-eye-tracker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def angles(p):
        p.add_argument("--distance-cm", type=float, help="from your eyes to the screen, for errors in degrees")
        p.add_argument("--width-cm", type=float, help="width of the monitor's picture, for errors in degrees")

    def camera(p):
        p.add_argument("--device", default=0, type=lambda d: int(d) if d.isdigit() else d, help="camera (default 0: /dev/video0)")

    def window(p):
        camera(p)
        p.add_argument("--no-fullscreen", dest="fullscreen", action="store_false", help="do not ask dune for fullscreen")

    measure = sub.add_parser("measure", help="calibrate on 15 points, test on 10 others, report the error")
    window(measure)
    angles(measure)

    calibrate = sub.add_parser("calibrate", help="the 15 points, saved for `run` (on the monitor the pointer is on)")
    window(calibrate)
    calibrate.set_defaults(distance_cm=None, width_cm=None)
    calibrate.add_argument("--from", dest="from_run", metavar="RUN.json", help="no window: the calibration points of a `measure` run")
    calibrate.add_argument("--output", help="with --from: the monitor it was on (dune.toml's [output.NAME])")

    head = sub.add_parser("calibrate-head", help="after `calibrate`: 5 points looked at while the head turns, for `run`")
    window(head)

    from .snap import SNAP
    from .track import GLIDE, HEAD_GAIN

    run = sub.add_parser("run", help="move the pointer where you look, until stopped")
    camera(run)
    run.add_argument("--head-gain", type=float, default=HEAD_GAIN, help=f"px per unit of head turn (default {HEAD_GAIN:g})")
    run.add_argument("--glide", type=float, default=GLIDE, help=f"s the pointer takes to glide most of the way (default {GLIDE:g}; 0: no glide)")
    run.add_argument("--snap", type=float, default=SNAP, help=f"px within which a jump lands on a button, link or tab (dune-a11y; default {SNAP:g}; 0: on the gaze)")
    run.add_argument("-v", "--verbose", action="store_true", help="fps and moves every second")

    sub.add_parser("toggle", help="start `run` in the background, or stop it (`calibrate` when there is no calibration)")

    analyze = sub.add_parser("analyze", help="the report of a saved run")
    analyze.add_argument("file")
    angles(analyze)

    args = parser.parse_args(argv)
    from . import measure as m
    from . import track

    if args.command == "calibrate" and args.from_run:
        return track.calibration_from_run(args.from_run, args.output)
    if args.command in ("measure", "calibrate"):
        return m.run(args)
    if args.command == "calibrate-head":
        from . import headcal

        return headcal.run(args)
    if args.command == "run":
        return track.run(args)
    if args.command == "toggle":
        return track.toggle(args)
    return m.analyze_file(args)


if __name__ == "__main__":
    sys.exit(main())
