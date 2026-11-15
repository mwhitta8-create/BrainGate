#!/usr/bin/env python3
"""
P300 speller-style stimulus display.

6x6 matrix of A-Z and 0-9 on a black background. Characters are dark green and
flash to bright green. Rows flash one at a time, then columns one at a time,
at 2 flashes per second.

Controls:
    SPACE   start / pause
    ESC     quit
"""
import argparse
import json
import random
import socket
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

#CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

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
# ------------------------------------------

INTERVAL_MS = int(1000 / FLASHES_PER_SECOND)

# Where flash markers are sent (UDP). Override on the command line:
#   python p300_speller.py --host 192.168.1.42 --port 5005
DEFAULT_HOST = "127.0.0.1"   # IP of the computer running p300_read.py
DEFAULT_PORT = 5005


class P300App:
    def __init__(self, root, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.root = root
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        root.title("P300 Test")
        root.configure(bg=BG)
        root.geometry("1900x1200")
        root.bind("<Escape>", lambda e: root.destroy())
        root.bind("<space>", self.toggle)

        font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE, weight=FONT_WEIGHT)

        grid = tk.Frame(root, bg=BG)
        grid.pack(expand=True, fill="both", padx=20, pady=20)
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

        self.hint = tk.Label(root, text="SPACE to start / pause   ESC to quit",
                             fg="#003300", bg=BG, font=("Arial", 10))
        self.hint.pack(side="bottom", pady=4)

        self.running = False
        self.sequence = []      # queue of ("row"|"col", index)
        self.job = None

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
            self.tick()
        elif self.job:
            self.root.after_cancel(self.job)
            self.job = None
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST, help="IP of the computer running p300_read.py")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    root = tk.Tk()
    P300App(root, args.host, args.port)
    root.mainloop()
