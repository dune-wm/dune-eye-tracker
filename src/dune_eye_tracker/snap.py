"""Landing on what can be clicked: at each jump, `dune-a11y serve OUTPUT` is asked what is
near where you look (the apps' accessibility trees, see ../dune-a11y; only the branches
near it are read, nothing between jumps) and the pointer glides onto it when it answers:
the middle of a button, in a field or a list where you look inside it. The webcam is ~120
px off, a button ~30 px tall. And `Puts`: another program putting the pointer somewhere."""

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

SNAP = 90.0  # px: a thing this near the gaze takes the jump
# px: what can be clicked this near a jump is asked for too, for the log of clicks only
# (the candidates a model of where clicks go chooses among: the gaze is ~180 px off)
AROUND = 250.0


def parse_answer(line: str) -> tuple[int, tuple[float, float] | None, list | None] | None:
    """`SEQ X Y` (land there) or `SEQ -` (nothing near: stay on the gaze), then, when
    asked, a JSON object with what is `around` (else None)."""
    head, brace, rest = line.partition("{")
    tail = None
    if brace:
        try:
            tail = json.loads(brace + rest).get("around")
        except (ValueError, AttributeError):
            return None
    parts = head.split()
    try:
        if len(parts) == 2 and parts[1] == "-":
            return int(parts[0]), None, tail
        if len(parts) == 3:
            return int(parts[0]), (float(parts[1]), float(parts[2])), tail
    except ValueError:
        pass
    return None


class Near(threading.Thread):
    """`dune-a11y serve OUTPUT` in the background: `ask` where a jump landed, `answers`
    where it should have (the jump's number with it, so a late answer for an older jump is
    told apart). Without dune-a11y (or when it stops) nothing answers: the gaze lands as it
    is."""

    def __init__(self, output: str, radius: float = SNAP, around: float = AROUND):
        super().__init__(daemon=True)
        self.output, self.radius, self.around = output, radius, around
        self.queue: Queue = Queue()
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None
        exe = shutil.which("dune-a11y")
        if exe is None:
            self.error = "dune-a11y is not installed: the pointer lands where you look"
            return
        try:
            self.proc = subprocess.Popen(
                [exe, "serve", output], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except OSError as e:
            self.error = f"dune-a11y: {e}"

    def run(self):
        if self.proc is None:
            return
        for line in self.proc.stdout:
            if (answer := parse_answer(line)) is not None:
                self.queue.put(answer)

    def ask(self, seq: int, x: float, y: float):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            wide = f" {self.around:.0f}" if self.around > 0 else ""
            self.proc.stdin.write(f"near {seq} {x:.0f} {y:.0f} {self.radius:.0f}{wide}\n")
            self.proc.stdin.flush()
        except OSError:  # gone: no more answers
            self.proc = None

    def answers(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

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
