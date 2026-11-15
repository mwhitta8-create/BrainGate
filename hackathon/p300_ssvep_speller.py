#!/usr/bin/env python3
"""
P300 flight grid + SSVEP "link" border (one stage, always on).

A 6x6 grid of flight codes. Rows/columns flash one at a time (the P300 stimulus)
and send ["row"|"col", index] markers over UDP to p300_live.py / p300_read.py.
At the same time a checkerboard border around the grid shimmers at a fixed rate
(the SSVEP stimulus) so the EEG can show whether someone is looking at the board.

Photosensitivity-minded design of the SSVEP border:
  * smooth sine modulation, not hard on/off flashing
  * low brightness and moderate contrast, neutral grey (no red)
  * "reversal" mode: the two checker colours swap while the AVERAGE brightness stays
    constant, so there is far less overall flicker than an on/off light
  * only a border strip flickers, not the whole screen
This reduces the risk and discomfort, it does NOT remove it. Warn anyone nearby and
stop immediately if anyone feels unwell.

Timing: the waveform is computed from the clock at every redraw, so the frequency
is correct whatever the monitor's refresh rate is (120, 144, ...). Tkinter redraws
about every 8-16 ms, which is plenty for the default 7.5 Hz modulation. The
script prints the achieved update rate so you can check.

Controls:
    SPACE   start / pause
    ESC     quit

Examples:
    python p300_ssvep_speller.py --host 192.168.1.42
    python p300_ssvep_speller.py --ssvep-hz 12 --depth 0.4
    python p300_ssvep_speller.py --no-ssvep          # P300 grid only
    --ssvep-hz 15 sets the response frequency, and --depth 0.4 makes it gentler (and the signal weaker).
    --mean sets the brightness, and --border sets the thickness.
    --mode luminance brightens and dims the whole border instead. It gives a stronger signal but is harsher, so I wouldn’t use it for the airport version.
    --no-ssvep runs the P300 grid alone.
    --fullscreen fills the screen.
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
CHARS = [
    "AA101", "DL202", "UA303", "WN404", "B6505", "AS606",
    "DL707", "AA808", "UA909", "WN110", "DL111", "B6112",
    "AS113", "AA114", "UA115", "WN116", "DL117", "B6118",
    "AS119", "AA120", "UA121", "WN122", "DL123", "B6124",
    "AS125", "AA126", "UA127", "WN128", "DL129", "B6130",
    "AS131", "AA132", "UA133", "WN134", "DL135", "B6136"
]

ROWS, COLS = 6, 6
FLASHES_PER_SECOND = 2
FLASH_DURATION_MS = 150          # how long a row/column stays bright
SHUFFLE_ORDER = True             # random order within the row pass / column pass
                                 # (standard for P300); set False for 1..6 order
BG = "#000000"
DARK_GREEN = "#005a00"
BRIGHT_GREEN = "#4dff4d"
FONT_FAMILY = "Arial"
FONT_SIZE = 56
FONT_WEIGHT = "bold"

# --- SSVEP border ---
SSVEP_HZ = 15.0          # frequency the EEG response should appear at (keep out of 8-13 Hz alpha)
SSVEP_MODE = "reversal"  # "reversal" = checkerboard colours swap (default, gentler)
                         # "luminance" = whole border brightens/dims (stronger, harsher)
SSVEP_MEAN = 0.25        # average brightness, 0..1 (0.25 = dim grey)
SSVEP_DEPTH = 0.6        # modulation depth, 0..1 (lower = gentler, weaker signal)
SSVEP_BORDER_PX = 70     # border thickness = size of one checker square
# ------------------------------------------

INTERVAL_MS = int(1000 / FLASHES_PER_SECOND)
SSVEP_TICK_MS = 8        # requested redraw interval (Windows rounds to ~15.6 ms)

# Where flash markers are sent (UDP). Override on the command line:
#   python p300_ssvep_speller.py --host 192.168.1.42 --port 5005
DEFAULT_HOST = "127.0.0.1"   # IP of the computer running p300_live.py / p300_read.py
DEFAULT_PORT = 5005


def ssvep_levels(t, f_cycle, mean, depth, mode):
    """Brightness (0..1) of the two checker colours A and B at time t (seconds).

    reversal : A and B swap smoothly; (A + B) / 2 stays constant.
    luminance: A and B are equal and brighten/dim together.
    """
    s = math.sin(2 * math.pi * f_cycle * t)
    if mode == "reversal":
        return mean * (1 + depth * s), mean * (1 - depth * s)
    level = mean * (1 + depth * s)
    return level, level


def grey(v):
    g = max(0, min(255, int(round(255 * v))))
    return f"#{g:02x}{g:02x}{g:02x}"


class P300SSVEPApp:
    def __init__(self, root, host=DEFAULT_HOST, port=DEFAULT_PORT, ssvep=True,
                 ssvep_hz=SSVEP_HZ, mode=SSVEP_MODE, mean=SSVEP_MEAN,
                 depth=SSVEP_DEPTH, border=SSVEP_BORDER_PX, fullscreen=False):
        self.root = root
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ssvep_on = ssvep
        self.mode, self.mean, self.depth, self.border = mode, mean, depth, border
        # In reversal mode the brain's response lands at the REVERSAL rate, which is
        # twice the colour-swap cycle rate. So to get a response at ssvep_hz, cycle at half.
        self.f_cycle = ssvep_hz / 2 if mode == "reversal" else ssvep_hz
        self.ssvep_hz = ssvep_hz

        root.title("P300 + SSVEP flights")
        root.configure(bg=BG)
        root.geometry("1900x1200")
        if fullscreen:
            root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda e: root.destroy())
        root.bind("<space>", self.toggle)

        font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE, weight=FONT_WEIGHT)

        self.hint = tk.Label(root, text="SPACE to start / pause   ESC to quit",
                             fg="#003300", bg=BG, font=("Arial", 10))
        self.hint.pack(side="bottom", pady=4)

        # Canvas holds the checkerboard border; the flight grid sits inside it.
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
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
                lbl = tk.Label(grid, text=CHARS[r * COLS + c], font=font,
                               fg=DARK_GREEN, bg=BG)
                lbl.grid(row=r, column=c, sticky="nsew")
                self.labels[r][c] = lbl

        self.running = False
        self.sequence = []      # queue of ("row"|"col", index)
        self.job = None
        self.ssvep_job = None
        self.t0 = time.perf_counter()
        self.last_tick = None
        self.intervals = []
        self.next_report = None

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
                on_ring = iy == 0 or iy == nrows - 1 or ix == 0 or ix == ncols - 1
                if on_ring:
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
        # timing diagnostics
        if self.last_tick is not None:
            self.intervals.append(now - self.last_tick)
        self.last_tick = now
        if self.next_report is None:
            self.next_report = now + 10
        elif now >= self.next_report and self.intervals:
            mean_dt = sum(self.intervals) / len(self.intervals)
            worst = max(self.intervals)
            per_cycle = (1 / mean_dt) / self.f_cycle
            print(f"[SSVEP] {self.mode} @ {self.f_cycle:.2f} Hz cycle -> expect EEG response at "
                  f"{self.ssvep_hz:.1f} Hz | redraws {1 / mean_dt:.0f}/s "
                  f"(mean {mean_dt * 1000:.1f} ms, worst {worst * 1000:.0f} ms), "
                  f"{per_cycle:.1f} per cycle", flush=True)
            if per_cycle < 5:
                print("[SSVEP] WARNING: too few redraws per cycle; lower --ssvep-hz or close other apps.")
            self.intervals, self.next_report = [], now + 30
        self.ssvep_job = self.root.after(SSVEP_TICK_MS, self.ssvep_tick)

    # ----- sequence handling -----
    def build_sequence(self):
        rows = list(range(ROWS))
        cols = list(range(COLS))
        if SHUFFLE_ORDER:
            random.shuffle(rows)
            random.shuffle(cols)
        return [("row", i) for i in rows] + [("col", j) for j in cols]

    def send_marker(self, kind, idx):
        """Send ["row"|"col", index] as JSON. Row 0 = top, column 0 = leftmost."""
        msg = json.dumps(["row" if kind == "row" else "col", idx]).encode()
        try:
            self.sock.sendto(msg, self.addr)
        except OSError as e:               # network down etc.: never stall the flashing
            print(f"marker send failed: {e}")

    def set_color(self, kind, idx, color):
        cells = (self.labels[idx] if kind == "row"
                 else [self.labels[r][idx] for r in range(ROWS)])
        for lbl in cells:
            lbl.config(fg=color)

    # ----- control -----
    def toggle(self, _event=None):
        self.running = not self.running
        if self.running:
            self.hint.config(text="")
            self.last_tick, self.next_report, self.intervals = None, None, []
            self.tick()
            self.ssvep_tick()
        else:
            for job in (self.job, self.ssvep_job):
                if job:
                    self.root.after_cancel(job)
            self.job = self.ssvep_job = None
            if self.ssvep_on:
                self.set_border(self.mean, self.mean)      # static, no flicker while paused
            self.hint.config(text="Paused - SPACE to resume   ESC to quit")

    def tick(self):
        if not self.running:
            return
        if not self.sequence:
            self.sequence = self.build_sequence()
        kind, idx = self.sequence.pop(0)
        self.set_color(kind, idx, BRIGHT_GREEN)
        self.root.update_idletasks()       # draw the flash now, then send the marker
        self.send_marker(kind, idx)
        self.root.after(FLASH_DURATION_MS,
                        lambda: self.set_color(kind, idx, DARK_GREEN))
        self.job = self.root.after(INTERVAL_MS, self.tick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P300 flight grid with an SSVEP checkerboard border")
    ap.add_argument("--host", default=DEFAULT_HOST, help="IP of the computer running p300_live.py")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-ssvep", action="store_true", help="P300 grid only, no flickering border")
    ap.add_argument("--ssvep-hz", type=float, default=SSVEP_HZ,
                    help="frequency the EEG response should appear at (default 15)")
    ap.add_argument("--mode", choices=["reversal", "luminance"], default=SSVEP_MODE)
    ap.add_argument("--mean", type=float, default=SSVEP_MEAN, help="average brightness 0..1")
    ap.add_argument("--depth", type=float, default=SSVEP_DEPTH, help="modulation depth 0..1")
    ap.add_argument("--border", type=int, default=SSVEP_BORDER_PX, help="border thickness in pixels")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    if not (0 < args.mean <= 0.5 and 0 < args.depth <= 1):
        ap.error("--mean must be in (0, 0.5] and --depth in (0, 1]")
    if not args.no_ssvep:
        print("NOTE: this screen contains a flickering border. Warn anyone nearby; stop at once "
              "if anyone feels unwell.")
        if 8 <= args.ssvep_hz <= 13:
            print("NOTE: 8-13 Hz overlaps your alpha rhythm and can cause false detections; "
                  "15 Hz or higher is safer for detection.")
    root = tk.Tk()
    P300SSVEPApp(root, args.host, args.port, ssvep=not args.no_ssvep, ssvep_hz=args.ssvep_hz,
                 mode=args.mode, mean=args.mean, depth=args.depth, border=args.border,
                 fullscreen=args.fullscreen)
    root.mainloop()
