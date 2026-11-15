#!/usr/bin/env python3
"""
Two flight boards (two "screens"), each a P300 grid with its own SSVEP border.

  Screen A  main window     36 original flights   markers -> UDP port 5005   SSVEP 15 Hz
  Screen B  second window   36 extra flights      markers -> UDP port 5006   SSVEP 18 Hz

Each board works exactly like p300_ssvep_speller.py: rows/columns flash one at a time
(the P300 stimulus) and send ["row"|"col", index] over UDP, while a checkerboard border
shimmers at a fixed rate (the SSVEP stimulus). The only differences are the flight list,
the UDP port and the SSVEP frequency, so the EEG can tell which board you are looking at.
B's P300 flashes are offset by half a flash interval so the two boards never flash at
the same instant.

Photosensitivity-minded design of the SSVEP border:
  * smooth sine modulation, not hard on/off flashing
  * low brightness and moderate contrast, neutral grey (no red)
  * "reversal" mode: the two checker colours swap while the AVERAGE brightness stays
    constant, so there is far less overall flicker than an on/off light
  * only a border strip flickers, not the whole screen
This reduces the risk and discomfort, it does NOT remove it. Two flickering windows
side by side are more than one. Warn anyone nearby and stop immediately if anyone
feels unwell.

Timing: waveforms are computed from the clock at every redraw, so the frequencies are
correct whatever the monitor's refresh rate is. Both windows share one Tk loop (about
8-16 ms per redraw) which is fine for 15 and 18 Hz, but if the achieved update rate
printed in the console looks low, close other apps or lower the frequencies.

Controls (either window):
    SPACE   start / pause BOTH boards
    F11     toggle fullscreen for that window
    ESC     quit both

Examples:
    python p300_ssvep_speller_dual.py --host 192.168.1.42
    python p300_ssvep_speller_dual.py --host 192.168.1.42 --ssvep-hz 15 --ssvep-hz-b 20
    python p300_ssvep_speller_dual.py --no-second     # screen A only
"""
import argparse
import json
import math
import random
import socket
import time
import tkinter as tk
import tkinter.font as tkfont

# ---------------- Settings ----------------
# Screen A: 36 flights, row-major (row 0 = top, column 0 = leftmost).
CHARS_A = [
    "AA101", "DL202", "UA303", "WN404", "B6505", "AS606",
    "DL707", "AA808", "UA909", "WN110", "DL111", "B6112",
    "AS113", "AA114", "UA115", "WN116", "DL117", "B6118",
    "AS119", "AA120", "UA121", "WN122", "DL123", "B6124",
    "AS125", "AA126", "UA127", "WN128", "DL129", "B6130",
    "AS131", "AA132", "UA133", "WN134", "DL135", "B6136",
]

# Screen B: the 36 extra flights. Listed by airline (as in flights.csv); the grid is
# built so each COLUMN is one airline and each row mixes the six airlines.
_B_BY_AIRLINE = [
    ["DL312", "DL448", "DL581", "DL626", "DL739", "DL852"],
    ["AA215", "AA337", "AA459", "AA562", "AA684", "AA791"],
    ["UA218", "UA342", "UA467", "UA573", "UA698", "UA824"],
    ["WN225", "WN351", "WN476", "WN582", "WN607", "WN733"],
    ["B6241", "B6368", "B6494", "B6517", "B6659", "B6782"],
    ["AS253", "AS379", "AS405", "AS538", "AS664", "AS771"],
]
CHARS_B = [_B_BY_AIRLINE[c][r] for r in range(6) for c in range(6)]   # row-major

ROWS, COLS = 6, 6
FLASHES_PER_SECOND = 2
FLASH_DURATION_MS = 150          # how long a row/column stays bright
SHUFFLE_ORDER = True             # random order within the row pass / column pass
BG = "#000000"
DARK_GREEN = "#005a00"
BRIGHT_GREEN = "#4dff4d"
FONT_FAMILY = "Arial"
FONT_SIZE = 56
FONT_WEIGHT = "bold"

# --- SSVEP border ---
SSVEP_HZ_A = 15.0        # keep out of 8-13 Hz (alpha)
SSVEP_HZ_B = 18.0
MIN_HZ_GAP = 2.0         # the two boards must differ by at least this much
SSVEP_MODE = "reversal"  # "reversal" (gentler) or "luminance" (stronger, harsher)
SSVEP_MEAN = 0.25
SSVEP_DEPTH = 0.6
SSVEP_BORDER_PX = 70
# ------------------------------------------

INTERVAL_MS = int(1000 / FLASHES_PER_SECOND)
SSVEP_TICK_MS = 8
DEFAULT_HOST = "127.0.0.1"   # IP of the computer running p300_live.py / p300_read.py
DEFAULT_PORT_A = 5005
DEFAULT_PORT_B = 5006


def ssvep_levels(t, f_cycle, mean, depth, mode):
    """Brightness (0..1) of the two checker colours A and B at time t (seconds)."""
    s = math.sin(2 * math.pi * f_cycle * t)
    if mode == "reversal":
        return mean * (1 + depth * s), mean * (1 - depth * s)
    level = mean * (1 + depth * s)
    return level, level


def grey(v):
    g = max(0, min(255, int(round(255 * v))))
    return f"#{g:02x}{g:02x}{g:02x}"


class Board:
    """One flight board: a P300 grid plus an SSVEP border, in a Tk or Toplevel window."""

    def __init__(self, win, name, chars, host, port, ssvep, ssvep_hz, mode, mean, depth,
                 border, start_delay_ms=0, geometry="1900x1200", fullscreen=False,
                 on_toggle=None, on_quit=None):
        assert len(chars) == ROWS * COLS
        self.win, self.name, self.chars = win, name, chars
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ssvep_on = ssvep
        self.mode, self.mean, self.depth, self.border = mode, mean, depth, border
        # Reversal mode: the brain responds at the reversal rate = 2 x the swap cycle rate.
        self.f_cycle = ssvep_hz / 2 if mode == "reversal" else ssvep_hz
        self.ssvep_hz = ssvep_hz
        self.start_delay_ms = start_delay_ms
        self.fullscreen = fullscreen

        win.title(f"P300 + SSVEP flights - screen {name} ({ssvep_hz:g} Hz, port {port})")
        win.configure(bg=BG)
        win.geometry(geometry)
        if fullscreen:
            win.attributes("-fullscreen", True)
        win.bind("<Escape>", lambda e: on_quit())
        win.bind("<space>", lambda e: on_toggle())
        win.bind("<F11>", self.toggle_fullscreen)
        win.protocol("WM_DELETE_WINDOW", on_quit)

        font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE, weight=FONT_WEIGHT)
        self.hint = tk.Label(win, text=f"Screen {name}   SPACE start / pause both   F11 fullscreen   ESC quit",
                             fg="#003300", bg=BG, font=("Arial", 10))
        self.hint.pack(side="bottom", pady=4)

        self.canvas = tk.Canvas(win, bg=BG, highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<Configure>", self.draw_border)

        grid = tk.Frame(self.canvas, bg=BG)
        pad = border if ssvep else 20
        grid.place(x=pad, y=pad, relwidth=1, relheight=1, width=-2 * pad, height=-2 * pad)
        for i in range(ROWS):
            grid.rowconfigure(i, weight=1, uniform="r")
        for j in range(COLS):
            grid.columnconfigure(j, weight=1, uniform="c")

        self.labels = [[None] * COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                lbl = tk.Label(grid, text=chars[r * COLS + c], font=font, fg=DARK_GREEN, bg=BG)
                lbl.grid(row=r, column=c, sticky="nsew")
                self.labels[r][c] = lbl

        self.running = False
        self.sequence = []
        self.job = None            # next P300 tick (or the delayed first tick)
        self.ssvep_job = None
        self.t0 = time.perf_counter()
        self.last_tick = None
        self.intervals = []
        self.next_report = None

    def toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        self.win.attributes("-fullscreen", self.fullscreen)

    # ----- SSVEP border -----
    def draw_border(self, _event=None):
        if not self.ssvep_on:
            return
        c, b = self.canvas, self.border
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        ncols, nrows = max(1, math.ceil(w / b)), max(1, math.ceil(h / b))
        for ix in range(ncols):
            for iy in range(nrows):
                if iy == 0 or iy == nrows - 1 or ix == 0 or ix == ncols - 1:
                    tag = "chkA" if (ix + iy) % 2 == 0 else "chkB"
                    c.create_rectangle(ix * b, iy * b, (ix + 1) * b, (iy + 1) * b,
                                       fill=grey(self.mean), outline="", tags=(tag,))

    def set_border(self, a, b):
        self.canvas.itemconfig("chkA", fill=grey(a))
        self.canvas.itemconfig("chkB", fill=grey(b))

    def ssvep_tick(self):
        if not (self.running and self.ssvep_on):
            return
        now = time.perf_counter()
        a, b = ssvep_levels(now - self.t0, self.f_cycle, self.mean, self.depth, self.mode)
        self.set_border(a, b)
        if self.last_tick is not None:
            self.intervals.append(now - self.last_tick)
        self.last_tick = now
        if self.next_report is None:
            self.next_report = now + 10
        elif now >= self.next_report and self.intervals:
            mean_dt = sum(self.intervals) / len(self.intervals)
            worst = max(self.intervals)
            per_cycle = (1 / mean_dt) / self.f_cycle
            print(f"[SSVEP {self.name}] {self.mode} @ {self.f_cycle:.2f} Hz cycle -> expect EEG response "
                  f"at {self.ssvep_hz:.1f} Hz | redraws {1 / mean_dt:.0f}/s "
                  f"(mean {mean_dt * 1000:.1f} ms, worst {worst * 1000:.0f} ms), "
                  f"{per_cycle:.1f} per cycle", flush=True)
            if per_cycle < 5:
                print(f"[SSVEP {self.name}] WARNING: too few redraws per cycle; lower the frequency "
                      "or close other apps.")
            self.intervals, self.next_report = [], now + 30
        self.ssvep_job = self.win.after(SSVEP_TICK_MS, self.ssvep_tick)

    # ----- P300 sequence -----
    def build_sequence(self):
        rows, cols = list(range(ROWS)), list(range(COLS))
        if SHUFFLE_ORDER:
            random.shuffle(rows)
            random.shuffle(cols)
        return [("row", i) for i in rows] + [("col", j) for j in cols]

    def send_marker(self, kind, idx):
        """Send ["row"|"col", index] as JSON. Row 0 = top, column 0 = leftmost."""
        msg = json.dumps(["row" if kind == "row" else "col", idx]).encode()
        try:
            self.sock.sendto(msg, self.addr)
        except OSError as e:
            print(f"[{self.name}] marker send failed: {e}")

    def set_color(self, kind, idx, color):
        cells = (self.labels[idx] if kind == "row"
                 else [self.labels[r][idx] for r in range(ROWS)])
        for lbl in cells:
            lbl.config(fg=color)

    # ----- control -----
    def start(self):
        self.running = True
        self.hint.config(text="")
        self.last_tick, self.next_report, self.intervals = None, None, []
        self.job = self.win.after(self.start_delay_ms, self.tick) if self.start_delay_ms else None
        if not self.start_delay_ms:
            self.tick()
        self.ssvep_tick()

    def stop(self):
        self.running = False
        for job in (self.job, self.ssvep_job):
            if job:
                self.win.after_cancel(job)
        self.job = self.ssvep_job = None
        if self.ssvep_on:
            self.set_border(self.mean, self.mean)
        self.hint.config(text=f"Screen {self.name} paused - SPACE to resume   ESC to quit")

    def tick(self):
        if not self.running:
            return
        if not self.sequence:
            self.sequence = self.build_sequence()
        kind, idx = self.sequence.pop(0)
        self.set_color(kind, idx, BRIGHT_GREEN)
        self.win.update_idletasks()        # draw the flash now, then send the marker
        self.send_marker(kind, idx)
        self.win.after(FLASH_DURATION_MS, lambda: self.set_color(kind, idx, DARK_GREEN))
        self.job = self.win.after(INTERVAL_MS, self.tick)


def main():
    ap = argparse.ArgumentParser(description="Two P300 flight boards with different SSVEP frequencies")
    ap.add_argument("--host", default=DEFAULT_HOST, help="IP of the computer running p300_live.py")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT_A, help="UDP port for screen A")
    ap.add_argument("--port-b", type=int, default=DEFAULT_PORT_B, help="UDP port for screen B")
    ap.add_argument("--ssvep-hz", type=float, default=SSVEP_HZ_A, help="screen A frequency (default 15)")
    ap.add_argument("--ssvep-hz-b", type=float, default=SSVEP_HZ_B, help="screen B frequency (default 18)")
    ap.add_argument("--stagger-ms", type=int, default=INTERVAL_MS // 2,
                    help="delay before B's first flash so A and B never flash together (default 250)")
    ap.add_argument("--no-second", action="store_true", help="open screen A only")
    ap.add_argument("--no-ssvep", action="store_true", help="P300 grids only, no flickering borders")
    ap.add_argument("--mode", choices=["reversal", "luminance"], default=SSVEP_MODE)
    ap.add_argument("--mean", type=float, default=SSVEP_MEAN, help="average brightness 0..1")
    ap.add_argument("--depth", type=float, default=SSVEP_DEPTH, help="modulation depth 0..1")
    ap.add_argument("--border", type=int, default=SSVEP_BORDER_PX, help="border thickness in pixels")
    ap.add_argument("--fullscreen", action="store_true", help="screen A fullscreen")
    ap.add_argument("--geometry-b", default="1900x1200+80+60", help="screen B window size/position")
    args = ap.parse_args()

    if not (0 < args.mean <= 0.5 and 0 < args.depth <= 1):
        ap.error("--mean must be in (0, 0.5] and --depth in (0, 1]")
    two = not args.no_second
    if two and args.port == args.port_b:
        ap.error("--port and --port-b must differ")
    if two and not args.no_ssvep and abs(args.ssvep_hz - args.ssvep_hz_b) < MIN_HZ_GAP:
        ap.error(f"the two SSVEP frequencies must differ by at least {MIN_HZ_GAP:g} Hz")
    if not args.no_ssvep:
        print("NOTE: these screens contain flickering borders. Warn anyone nearby; stop at once "
              "if anyone feels unwell.")
        for hz in ([args.ssvep_hz] + ([args.ssvep_hz_b] if two else [])):
            if 8 <= hz <= 13:
                print(f"NOTE: {hz:g} Hz overlaps your alpha rhythm and can cause false detections.")

    root = tk.Tk()
    boards = []

    def toggle_all(_event=None):
        run = not any(b.running for b in boards)
        for b in boards:
            (b.start if run else b.stop)()

    def quit_all(_event=None):
        for b in boards:
            b.stop()
        root.destroy()

    common = dict(host=args.host, ssvep=not args.no_ssvep, mode=args.mode, mean=args.mean,
                  depth=args.depth, border=args.border, on_toggle=toggle_all, on_quit=quit_all)
    boards.append(Board(root, "A", CHARS_A, port=args.port, ssvep_hz=args.ssvep_hz,
                        fullscreen=args.fullscreen, **common))
    if two:
        win_b = tk.Toplevel(root)
        boards.append(Board(win_b, "B", CHARS_B, port=args.port_b, ssvep_hz=args.ssvep_hz_b,
                            start_delay_ms=args.stagger_ms, geometry=args.geometry_b, **common))
    root.mainloop()


if __name__ == "__main__":
    main()
