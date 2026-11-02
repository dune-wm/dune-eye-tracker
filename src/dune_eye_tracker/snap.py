"""Landing on what can be clicked: where the buttons, links and tabs of the monitor are,
from `dune-a11y watch OUTPUT` (the apps' accessibility trees, see ../dune-a11y), and the
one a gaze is nearest. A jump lands on its middle instead of the gaze's pixel: the webcam
is ~120 px off, a button is ~30 px tall."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue

SNAP = 90.0  # px: a target this near the gaze takes the jump
BIG = 160  # px: a target this wide or tall (a field, a list) keeps the gaze inside it


def nearest(targets: list[tuple[int, int, int, int]], x: float, y: float, radius: float):
    """Where a jump to the gaze (x, y) lands: the middle of the nearest target (x, y, w, h)
    within `radius` (the smaller when the gaze is in more than one); in a big one the gaze
    kept inside it. None: none that near."""
    best = None
    for tx, ty, tw, th in targets:
        dx = max(tx - x, 0.0, x - (tx + tw - 1))
        dy = max(ty - y, 0.0, y - (ty + th - 1))
        d = (dx * dx + dy * dy) ** 0.5
        if d <= radius and (best is None or (d, tw * th) < best[:2]):
            best = (d, tw * th, (tx, ty, tw, th))
    if best is None:
        return None
    tx, ty, tw, th = best[2]
    if tw > BIG or th > BIG:
        inset = 4
        return (min(max(x, tx + inset), tx + tw - 1 - inset), min(max(y, ty + inset), ty + th - 1 - inset))
    return (tx + (tw - 1) / 2, ty + (th - 1) / 2)


def parse(line: str) -> list[tuple[int, int, int, int]] | None:
    """A `dune-a11y watch` line's targets; None when it is not one."""
    try:
        found = json.loads(line)["targets"]
        return [(int(t["x"]), int(t["y"]), int(t["w"]), int(t["h"])) for t in found]
    except (ValueError, KeyError, TypeError):
        return None


class Targets(threading.Thread):
    """`dune-a11y watch OUTPUT` in the background: the latest targets of the monitor.
    Without dune-a11y (or when it stops) there are none, and the gaze lands as it is."""

    def __init__(self, output: str, radius: float = SNAP):
        super().__init__(daemon=True)
        self.output, self.radius = output, radius
        self.targets: list[tuple[int, int, int, int]] = []
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None

    def run(self):
        exe = shutil.which("dune-a11y")
        if exe is None:
            self.error = "dune-a11y is not installed: the pointer lands where you look"
            return
        try:
            self.proc = subprocess.Popen(
                [exe, "watch", self.output], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError as e:
            self.error = f"dune-a11y: {e}"
            return
        for line in self.proc.stdout:
            found = parse(line)
            if found is not None:
                self.targets = found  # one assignment: the reader sees the old or the new
        self.targets = []

    def __call__(self, gaze):
        """`Pointer`'s snap: where a jump to `gaze` lands, None to land on it."""
        return nearest(self.targets, float(gaze[0]), float(gaze[1]), self.radius)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def socket_path() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "dune-eye-tracker.sock"


def parse_put(line: str, output: str) -> tuple[int, int] | None:
    """`put X Y OUTPUT` (dune-a11y: the pointer onto what it clicked) on this monitor."""
    parts = line.split()
    if len(parts) != 4 or parts[0] != "put" or parts[3] != output:
        return None
    try:
        return int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None


class Puts(threading.Thread):
    """`$XDG_RUNTIME_DIR/dune-eye-tracker.sock`: another program (dune-a11y's `click`, by
    voice) puts the pointer somewhere on this monitor. `run` moves it there and leaves it,
    as after the mouse, until you look elsewhere. Answers `ok`, or `no` (another monitor:
    it moves the pointer itself)."""

    def __init__(self, output: str):
        super().__init__(daemon=True)
        self.output = output
        self.queue: Queue = Queue()
        self.path = socket_path()
        self.server: socket.socket | None = None

    def run(self):
        self.path.unlink(missing_ok=True)
        try:
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(str(self.path))
            self.server.listen(4)
        except OSError:
            return
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:  # closed: `run` stops
                return
            with conn:
                conn.settimeout(1.0)
                try:
                    at = parse_put(conn.recv(256).decode(errors="replace"), self.output)
                    if at is not None:
                        self.queue.put(("put", time.monotonic(), *at))
                    conn.sendall(b"ok\n" if at is not None else b"no\n")
                except OSError:
                    continue

    def pending(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

    def stop(self):
        if self.server is not None:
            self.server.close()
        self.path.unlink(missing_ok=True)
