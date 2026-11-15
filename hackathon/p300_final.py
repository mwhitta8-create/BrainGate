"""
FINAL live system for the Octopus 16 - single board version.
Use with:  p300_ssvep_speller.py   (one 6x6 flight board, UDP port 5005, SSVEP 15 Hz)

What it does
  * reads the 16-channel Bluetooth stream (python_quickstart.py) or a simulated board (--sim)
  * receives the speller's flash markers over Wi-Fi (p300_read.py)
  * classifies each flash as P300-like or not (shrinkage LDA) and keeps a running score per
    row and column -> the flight you are looking at
  * measures the SSVEP response to the board's flickering border (spectral SNR, no classifier)
    -> "SSVEP Hz: 15.0 Hz ... LINKED" means someone is looking at the board
  * NEW: when the estimated flight has stayed the same for --hold-s seconds (default 3) while
    the SSVEP link is present, that flight's details are pushed to your phone with ntfy
    (same message as ntfy_test.py, using flights.csv)

Keep these in one folder (or put ntfy_test.py and flights.csv in a subfolder called ntfy):
    p300_final.py  python_quickstart.py  p300_read.py  ntfy_test.py  flights.csv
Needs: numpy scipy matplotlib scikit-learn bleak (bleak only for the real board).

THREE MODES (the SSVEP readout is on in all of them; notifications only in mode 3)
  1. Look at the signal (no P300 classification). Check contact quality first:
         python p300_final.py
  2. Calibrate: you stare at ONE known flight while the speller flashes.
     Close the window (or Ctrl+C / --seconds) and it trains + saves a model:
         python p300_final.py --calibrate --target DL111
  3. Live detection + notifications with the saved model:
         python p300_final.py --model p300_model.pkl --topic my-long-random-topic

Simulated run (no hardware, no phone needed - prints what it WOULD send):
    python p300_final.py --sim --calibrate --target DL111 --seconds 60     (speller running)
    python p300_final.py --sim --sim-target DL117 --model p300_model.pkl
  Add --topic NAME to really send. In the plot press L to toggle "subject looking at the
  board" (no link = nothing is sent), R to clear the scores.

How a flight gets sent (all tunable):
  * the estimate needs at least --min-rounds scores for every row and column (default 2,
    i.e. two full row/column passes, about 12 s at 2 flashes per second)
  * it must be the SAME flight, continuously, for --hold-s seconds (default 3)
  * the SSVEP link must be present the whole time (turn off with --no-require-link)
  * after sending, the scores are cleared; that flight is not sent again until the link drops
    or a different flight is selected
Typical bad-pin handling:  --exclude 4,7   (1-based channel numbers)
"""
import argparse
import asyncio
import math
import os
import pickle
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
from scipy import signal

import python_quickstart as oq
from p300_read import MarkerListener

FS = oq.NOMINAL_SPS
N_CH = oq.NUM_CHANNELS
CHARS = [
    "AA101", "DL202", "UA303", "WN404", "B6505", "AS606",
    "DL707", "AA808", "UA909", "WN110", "DL111", "B6112",
    "AS113", "AA114", "UA115", "WN116", "DL117", "B6118",
    "AS119", "AA120", "UA121", "WN122", "DL123", "B6124",
    "AS125", "AA126", "UA127", "WN128", "DL129", "B6130",
    "AS131", "AA132", "UA133", "WN134", "DL135", "B6136"
]   # same layout as p300_speller.py
BAND = (0.5, 10.0)      # Hz P300 band-pass (causal). Stops below the SSVEP border frequency
                        # so the flicker does not leak into P300 epochs.
SS_BAND = (5.0, 40.0)   # Hz band-pass used for the SSVEP analysis
PRE_S, POST_S = 0.1, 0.8
STEP = 5                # keep every 5th sample after the low-pass (50 Hz)


# ----------------------------------------------------------------------------
# Stream tap: sits on the quickstart's StreamMonitor
# ----------------------------------------------------------------------------
class Tap(oq.StreamMonitor):
    """Quickstart StreamMonitor (CSV, packet-loss counting) plus a queue of
    (arrival_time, 16-ch sample). Lost packets are filled by repeating the last
    sample so the sample index stays true to real time."""

    def __init__(self, q, show_hud=False, **kw):
        super().__init__(**kw)
        self.q, self.show_hud = q, show_hud
        self._prev, self._last, self.filled = None, None, 0

    def ingest(self, counter, channels):
        super().ingest(counter, channels)
        now = time.time()
        if self._prev is not None:
            step = (counter - self._prev) & 0xFF
            if step == 0 or step >= 128:          # duplicate / out of order
                return
            for _ in range(step - 1):             # lost packets
                self.q.put((float("nan"), self._last))
                self.filled += 1
        self._prev, self._last = counter, channels
        self.q.put((now, channels))

    def print_hud(self):
        if self.show_hud:
            super().print_hud()


# ----------------------------------------------------------------------------
# Simulated board (for testing without hardware)
# ----------------------------------------------------------------------------
class Sim:
    """Fake board. Adds a P300-like bump after each flash marker that matches
    the simulated subject's target character."""

    def __init__(self, target_char, ssvep_hz=15.0, looking=True):
        self.trow, self.tcol = divmod(CHARS.index(target_char), 6)
        self.ssvep_hz, self.looking = ssvep_hz, looking     # looking: subject watches the board
        self.L = FS * 2
        self.ring = np.zeros((N_CH, self.L))
        self.p = 0
        self.lock = threading.Lock()
        t = np.arange(int(0.8 * FS)) / FS
        self.template = 8.0 * np.exp(-((t - 0.35) ** 2) / (2 * 0.07 ** 2))
        self.w = np.random.default_rng(0).uniform(0.3, 1.0, N_CH)

    def on_marker(self, kind, idx):
        if (kind == "row" and idx == self.trow) or (kind == "col" and idx == self.tcol):
            with self.lock:
                pos = (self.p + np.arange(len(self.template))) % self.L
                self.ring[:, pos] += self.w[:, None] * self.template[None, :]

    def ssvep(self, t):
        """Small steady-state response at the border frequency while 'looking'."""
        return self.w * 4.0 * math.sin(2 * math.pi * self.ssvep_hz * t) if self.looking else 0.0

    def pop(self):
        with self.lock:
            v = self.ring[:, self.p].copy()
            self.ring[:, self.p] = 0
            self.p = (self.p + 1) % self.L
        return v


async def run_sim(monitor, sim):
    print("[Sim] simulated 16-channel board at 250 Hz (no hardware).")
    rng = np.random.default_rng()
    phase = rng.uniform(0, 2 * np.pi, N_CH)
    dt, n, counter = 1.0 / FS, 0, 0
    next_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        while now >= next_t:
            t = n * dt
            ch = (300.0 + 20.0 * math.sin(2 * math.pi * 0.2 * t)       # DC + drift
                  + 8.0 * np.sin(2 * np.pi * 10 * t + phase)           # alpha
                  + rng.normal(0, 5.0, N_CH) + sim.pop() + sim.ssvep(t))
            monitor.ingest(counter, [round(float(x), 2) for x in ch])
            counter, n, next_t = (counter + 1) & 0xFF, n + 1, next_t + dt
        await asyncio.sleep(0.002)


# ----------------------------------------------------------------------------
# SSVEP frequency estimate (no classifier: spectrum + signal-to-noise)
# ----------------------------------------------------------------------------
def estimate_ssvep(x, fs, lo, hi, step=0.1):
    """x: (channels, samples) band-passed EEG. Averages the power spectrum over channels and,
    for every candidate frequency in [lo, hi], compares the power right at it (+-0.25 Hz) with
    the power in the surrounding bands (0.8-2.0 Hz either side). Returns (best_freq, its_snr)."""
    n = x.shape[1]
    x = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))                      # zero-pad for a finer frequency grid
    power = (np.abs(np.fft.rfft(x * np.hanning(n), nfft, axis=1)) ** 2).mean(axis=0)
    df = fs / nfft
    cs = np.concatenate([[0.0], np.cumsum(power)])

    def band_mean(f1, f2):
        a = np.clip(np.round(f1 / df).astype(int), 0, len(power) - 1)
        b = np.clip(np.round(f2 / df).astype(int) + 1, 1, len(power))
        return (cs[b] - cs[a]) / np.maximum(b - a, 1)

    cand = np.arange(lo, hi + 1e-9, step)
    sig = band_mean(cand - 0.25, cand + 0.25)
    noise = 0.5 * (band_mean(cand - 2.0, cand - 0.8) + band_mean(cand + 0.8, cand + 2.0))
    snr = sig / np.maximum(noise, 1e-12)
    k = int(np.argmax(snr))
    return float(cand[k]), float(snr[k])


# ----------------------------------------------------------------------------
# ntfy: flight details (flights.csv) pushed to the phone, as in ntfy_test.py
# ----------------------------------------------------------------------------
def import_ntfy_test():
    """Find ntfy_test.py next to this file or in a subfolder called ntfy."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here, "ntfy")):
        if os.path.isfile(os.path.join(d, "ntfy_test.py")):
            sys.path.insert(0, d)
            import ntfy_test
            return ntfy_test
    return None


class Notifier:
    """Sends a flight's notification in a background thread (so BLE/processing never waits)."""

    def __init__(self, ntfy_test, flights_csv, topic, server, click, dry_run):
        self.nt, self.topic, self.server, self.click, self.dry = ntfy_test, topic, server, click, dry_run
        self.flights = {f["code"]: f for f in ntfy_test.load_flights(flights_csv or ntfy_test.CSV_PATH)}

    def send(self, code, note=""):
        f = self.flights.get(code)
        if f is None:
            print(f"[NTFY] {code} is not in flights.csv - nothing sent")
            return
        threading.Thread(target=self._send, args=(f, note), daemon=True).start()

    def _send(self, f, note):
        msg = self.nt.build_notification(f)
        print(f"[NTFY] {'DRY RUN - would send' if self.dry else 'sending'}: {msg['title']}"
              f"  (priority {msg['priority']}){note}")
        for line in msg["message"].split("\n"):
            print(f"        {line}")
        if self.dry:
            return
        ok = self.nt.send(msg, self.topic, self.server, self.click)
        print(f"[NTFY] {'sent' if ok else 'NOT sent'}: {msg['title']}", flush=True)


class Selector:
    """Decides when a candidate (a flight) has been looked at long enough to send.

    update(cand, linked, now) is called ~10x per second:
      cand    the current best-estimate flight (any hashable) or None if there is no estimate yet
      linked  False when the SSVEP link is missing (this resets everything)
    Returns (action, cand) where action is None, "new" (a different candidate started its
    hold), "drop" (the candidate disappeared), "unlink" (link lost) or "fire" (held long
    enough: send it). After "fire" the same candidate is not fired again until the link drops
    or a different candidate fires (self.locked)."""

    def __init__(self, hold_s):
        self.hold_s = hold_s
        self.cand = None        # candidate currently being held
        self.since = 0.0        # when it became the candidate
        self.locked = None      # candidate that was already sent

    def update(self, cand, linked, now):
        if not linked:
            had = self.cand is not None or self.locked is not None
            self.cand, self.locked = None, None
            return ("unlink", None) if had else (None, None)
        if cand is None:
            had, self.cand = self.cand, None
            return ("drop", had) if had is not None else (None, None)
        if cand == self.locked:
            self.cand = None
            return None, None
        if cand != self.cand:
            self.cand, self.since = cand, now
            return "new", cand
        if now - self.since >= self.hold_s:
            self.locked, self.cand = cand, None
            return "fire", cand
        return None, None


# ----------------------------------------------------------------------------
# Processor: filtering, time alignment, epoching, classification
# ----------------------------------------------------------------------------
class Processor:
    def __init__(self, args, q, listener, sim=None, model=None, notifier=None):
        self.args, self.q, self.listener, self.sim, self.model = args, q, listener, sim, model
        self.notifier = notifier
        self.mode = "calibrate" if args.calibrate else ("live" if model else "feed")
        excl = {int(x) - 1 for x in args.exclude.split(",") if x.strip()}
        self.chans = model["chans"] if model else [c for c in range(N_CH) if c not in excl]
        self.n_pre, self.n_post = int(PRE_S * FS), int(POST_S * FS)
        self.latency = args.latency_ms / 1000.0
        self.ss_on = not args.no_ssvep
        self.sos = signal.butter(4, BAND, btype="bandpass", fs=FS, output="sos")
        # Notch out the border flicker (and the speller's slower 7.5 Hz colour-swap cycle) so
        # it cannot leak into the P300 epochs. Same filter in calibration and live.
        self.notch = (args.ssvep_hz, args.ssvep_hz / 2) if self.ss_on else None
        for f0 in (self.notch or ()):
            b, a = signal.iirnotch(f0, Q=5.0, fs=FS)
            self.sos = np.vstack([self.sos, signal.tf2sos(b, a)])
        self.zi = None
        self.sos_ss = signal.butter(4, SS_BAND, btype="bandpass", fs=FS, output="sos")
        self.zi_ss = None
        self.ss_win = int(args.ssvep_window * FS)
        self.ss = np.zeros((N_CH, self.ss_win + FS), dtype=np.float32)   # recent SSVEP-band data
        self.ss_count = 0
        self.ss_hist = deque(maxlen=5)          # last few estimates, median-smoothed for display
        self.ssvep = dict(freq=None, snr=None, linked=False)
        self.last_ss = self.last_ss_print = 0.0
        self.N = int(FS * 60 * args.max_minutes)
        self.filt = np.zeros((N_CH, self.N), dtype=np.float32)
        self.n = 0
        self.anch_idx = np.zeros(self.N)
        self.anch_t = np.zeros(self.N)
        self.n_anch = 0
        self.fit, self.last_fit = None, 0.0
        self.pending = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.marks = deque(maxlen=300)          # (sample_index, label)
        self.flashes = deque(maxlen=60)         # scores of recent flashes
        self.rows = [deque(maxlen=args.rounds) for _ in range(6)]
        self.cols = [deque(maxlen=args.rounds) for _ in range(6)]
        self.nflash = 0
        self.X, self.y, self.meta, self.eps = [], [], [], []
        self.stats = dict(markers=0, epochs=0, rejected=0)
        self.sel = Selector(args.hold_s)
        self.sel_status = ""
        self.last_decide = self.last_progress = 0.0
        self.sent = []                          # (time, flight) of every notification fired
        if args.target:
            self.trow, self.tcol = divmod(CHARS.index(args.target), 6)

    # ---- time mapping ----
    def time_to_idx(self, t):
        i0, t0, slope, icpt = self.fit
        return i0 + (t - t0 - icpt) / slope

    def _refit(self):
        k = min(self.n_anch, 3000)
        idx, t = self.anch_idx[self.n_anch - k:self.n_anch], self.anch_t[self.n_anch - k:self.n_anch]
        slope, icpt = np.polyfit(idx - idx[0], t - t[0], 1)
        self.fit = (idx[0], t[0], slope, icpt)

    # ---- main loop ----
    def run(self):
        while not self.stop.is_set():
            self._ingest_samples()
            self._ingest_markers()
            self._process_pending()
            self._update_ssvep()
            self._decide()

    def _update_ssvep(self):
        """Every 0.5 s: find the strongest peak near the border frequency in the last few seconds."""
        a = self.args
        if not self.ss_on or time.time() - self.last_ss < 0.5 or self.ss_count < self.ss_win:
            return
        self.last_ss = time.time()
        with self.lock:
            x = self.ss[self.chans, -self.ss_win:].astype(float)
        fs_eff = FS                                    # use the measured rate if it is close to nominal
        if self.fit:
            measured = 1.0 / self.fit[2]
            if 0.95 * FS < measured < 1.05 * FS:
                fs_eff = measured
        freq, snr = estimate_ssvep(x, fs_eff, a.ssvep_hz - a.ssvep_width, a.ssvep_hz + a.ssvep_width)
        self.ss_hist.append((freq, snr))
        f_med = float(np.median([h[0] for h in self.ss_hist]))
        s_med = float(np.median([h[1] for h in self.ss_hist]))
        linked = s_med >= a.ssvep_snr and abs(f_med - a.ssvep_hz) <= 1.0
        was = self.ssvep["linked"]
        with self.lock:
            self.ssvep = dict(freq=f_med, snr=s_med, linked=linked)
        if linked != was:
            print(f"[SSVEP] {'LINKED' if linked else 'link lost'}  ({f_med:.1f} Hz, SNR {s_med:.1f})", flush=True)
        if a.no_plot and time.time() - self.last_ss_print > 2:
            self.last_ss_print = time.time()
            print(f"[SSVEP] {f_med:.1f} Hz  SNR {s_med:.1f}  {'linked' if linked else 'no link'}", flush=True)

    def _ingest_samples(self):
        try:
            items = [self.q.get(timeout=0.01)]
        except queue.Empty:
            return
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        ts = np.array([i[0] for i in items])
        data = np.array([i[1] for i in items], dtype=float).T          # (16, n)
        m = data.shape[1]
        if self.n + m > self.N:
            print("Recording buffer full; stopping.")
            self.stop.set()
            return
        if self.zi is None:                                            # start with no filter transient
            self.zi = signal.sosfilt_zi(self.sos)[:, None, :] * data[:, 0][None, :, None]
        out, self.zi = signal.sosfilt(self.sos, data, axis=1, zi=self.zi)
        if self.zi_ss is None:
            self.zi_ss = signal.sosfilt_zi(self.sos_ss)[:, None, :] * data[:, 0][None, :, None]
        out_ss, self.zi_ss = signal.sosfilt(self.sos_ss, data, axis=1, zi=self.zi_ss)
        real = np.flatnonzero(~np.isnan(ts))
        with self.lock:
            L = self.ss.shape[1]
            if m >= L:
                self.ss[:] = out_ss[:, -L:]
            else:
                self.ss[:, :-m] = self.ss[:, m:]
                self.ss[:, -m:] = out_ss
            self.ss_count += m
            self.filt[:, self.n:self.n + m] = out
            k = len(real)
            self.anch_idx[self.n_anch:self.n_anch + k] = self.n + real
            self.anch_t[self.n_anch:self.n_anch + k] = ts[real]
            self.n_anch += k
            self.n += m
        if self.n_anch >= 100 and time.time() - self.last_fit > 1.0:
            self._refit()
            self.last_fit = time.time()

    def _ingest_markers(self):
        for t, kind, idx in self.listener.drain():
            self.pending.append((t, kind, idx))
            self.stats["markers"] += 1
            if self.sim:
                self.sim.on_marker(kind, idx)

    def _process_pending(self):
        if self.fit is None:
            self.pending = [p for p in self.pending if time.time() - p[0] < 10]
            return
        keep = []
        for t, kind, idx in self.pending:
            s = int(round(self.time_to_idx(t - self.latency)))
            if self.mode == "feed":
                with self.lock:
                    self.marks.append((s, None))
                continue
            if s + self.n_post > self.n:
                if time.time() - t < 5:
                    keep.append((t, kind, idx))
                continue
            if s - self.n_pre >= 0:
                self._handle_epoch(s, kind, idx, t)
        self.pending = keep

    def _handle_epoch(self, s, kind, idx, t):
        seg = self.filt[:, s - self.n_pre:s + self.n_post].astype(float)
        ep = seg - seg[:, :self.n_pre].mean(axis=1, keepdims=True)
        stamp = f"{time.strftime('%H:%M:%S', time.localtime(t))}.{int((t % 1) * 1000):03d}"
        if np.ptp(ep[self.chans], axis=1).max() > self.args.reject:
            self.stats["rejected"] += 1
            with self.lock:
                self.marks.append((s, None))
            print(f"[{stamp}] {kind} {idx}: rejected (artifact)")
            return
        feat = ep[self.chans][:, self.n_pre::STEP].reshape(-1)
        self.stats["epochs"] += 1
        if self.mode == "calibrate":
            is_t = (kind == "row" and idx == self.trow) or (kind == "col" and idx == self.tcol)
            self.X.append(feat)
            self.y.append(is_t)
            self.meta.append((t, kind, idx))
            self.eps.append(ep.astype(np.float32))
            with self.lock:
                self.marks.append((s, bool(is_t)))
            return
        score = float(self.model["clf"].decision_function(feat[None])[0])
        hit = score > self.args.thresh
        with self.lock:
            self.marks.append((s, hit))
            self.flashes.append(score)
            (self.rows if kind == "row" else self.cols)[idx].append(score)
            self.nflash += 1
        print(f"[{stamp}] {kind} {idx}: score {score:+.2f}  {'** P300-like **' if hit else ''}")
        if self.nflash % 12 == 0:
            r, c, letter = self.estimate()
            if letter:
                print(f"    estimate after {self.nflash} flashes: {letter} (row {r}, col {c})")

    # ---- flight selection + notification ----
    def ready_estimate(self, min_rounds):
        """Like estimate(), but only once every row and column has at least min_rounds scores."""
        if min(len(d) for d in self.rows + self.cols) < min_rounds:
            return None
        r, c, code = self.estimate()
        return (r, c, code) if code else None

    def _decide(self):
        """Live mode only: send the flight once it has been the estimate, with the SSVEP link,
        for --hold-s seconds."""
        a = self.args
        if self.mode != "live":
            return
        now = time.time()
        if now - self.last_decide < 0.1:
            return
        self.last_decide = now
        est = self.ready_estimate(a.min_rounds)
        with self.lock:
            linked = self.ssvep["linked"] if (self.ss_on and a.require_link) else True
        cand = est[2] if est else None
        act, c = self.sel.update(cand, linked, now)
        if act == "new":
            print(f"[SELECT] candidate {c} (row {est[0]}, col {est[1]}) - hold {a.hold_s:g} s to send", flush=True)
        elif act == "drop":
            print(f"[SELECT] candidate {c} dropped (no estimate)", flush=True)
        elif act == "unlink":
            print("[SELECT] SSVEP link lost - selection reset", flush=True)
        elif act == "fire":
            self.sent.append((now, c))
            print(f"[SELECT] *** {c} held for {a.hold_s:g} s -> SENDING ***", flush=True)
            if self.notifier:
                self.notifier.send(c)
            self.reset_scores()                 # next selection starts from fresh evidence
            cand = None
        elif self.sel.cand is not None and now - self.last_progress >= 1.0:
            self.last_progress = now
            print(f"[SELECT] holding {self.sel.cand}: {now - self.sel.since:.1f}/{a.hold_s:g} s", flush=True)
        if self.sel.locked and not (cand and cand != self.sel.locked) and linked:
            status = f"SENT {self.sel.locked}  (look away or at another flight to re-arm)"
        elif not linked:
            status = "Waiting for SSVEP link (look at the board)"
        elif cand is None:
            status = f"Collecting P300 evidence (need {a.min_rounds}+ per row/col)"
        else:
            status = f"Holding {cand}: {now - self.sel.since:.1f}/{a.hold_s:g} s"
        self.sel_status = status

    def estimate(self):
        rm = np.array([np.mean(d) if d else -np.inf for d in self.rows])
        cm = np.array([np.mean(d) if d else -np.inf for d in self.cols])
        if np.isinf(rm).all() or np.isinf(cm).all():
            return None, None, None
        r, c = int(np.argmax(rm)), int(np.argmax(cm))
        return r, c, CHARS[r * 6 + c]

    def reset_scores(self):
        with self.lock:
            for d in self.rows + self.cols:
                d.clear()
            self.flashes.clear()

    def snapshot(self, show):
        with self.lock:
            n = self.n
            data = np.zeros((N_CH, show))
            lo = max(0, n - show)
            data[:, show - (n - lo):] = self.filt[:, lo:n]
            seg = self.filt[:, max(0, n - 2 * FS):n].astype(float)
            rms = np.sqrt((seg ** 2).mean(axis=1)) if seg.shape[1] else np.zeros(N_CH)
            marks = [((s - n) / FS, lab) for s, lab in self.marks if n - show <= s <= n]
            flashes = list(self.flashes)
            rm = np.array([np.mean(d) if d else np.nan for d in self.rows])
            cm = np.array([np.mean(d) if d else np.nan for d in self.cols])
            ss = dict(self.ssvep)
        slope = self.fit[2] if self.fit else None
        return dict(data=data, rms=rms, marks=marks, flashes=flashes, rm=rm, cm=cm,
                    sps=(1 / slope if slope else 0.0), n=n, est=self.estimate(), ssvep=ss,
                    sel=self.sel_status, sent=list(self.sent))


# ----------------------------------------------------------------------------
# Calibration: train + save
# ----------------------------------------------------------------------------
def train_and_save(proc, args):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    if not proc.X:
        print("No epochs collected. Was the speller running and sending markers?")
        return
    X, y = np.array(proc.X), np.array(proc.y, dtype=bool)
    print(f"\nCalibration epochs: {len(y)} (target {y.sum()}, non-target {(~y).sum()}), "
          f"rejected {proc.stats['rejected']}")
    if y.sum() < 10 or (~y).sum() < 30:
        print("Not enough epochs to train. Record longer (aim for 15+ full row/column cycles).")
        return
    aucs = []
    for tr, te in KFold(5, shuffle=False).split(X):
        if y[tr].sum() == 0 or y[te].sum() == 0 or y[te].all():
            continue
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    if aucs:
        print(f"Cross-validated AUC per fold: {np.round(aucs, 3)}  mean {np.mean(aucs):.3f}  "
              f"(0.5 = chance, 1.0 = perfect)")
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    with open(args.out, "wb") as f:
        pickle.dump(dict(clf=clf, chans=proc.chans, fs=FS, band=BAND, notch=proc.notch, step=STEP,
                         pre=PRE_S, post=POST_S, target=args.target), f)
    npz = args.out.rsplit(".", 1)[0] + "_epochs.npz"
    np.savez(npz, epochs=np.array(proc.eps), labels=y,
             kinds=np.array([m[1] for m in proc.meta]),
             index=np.array([m[2] for m in proc.meta]),
             times=np.array([m[0] for m in proc.meta]))
    print(f"Saved model to {args.out} and the epochs to {npz}")


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def run_plot(proc, args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    show = 8 * FS
    sp = args.spacing
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax_s = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1])
    x = np.linspace(-8, 0, show)
    lines = [ax.plot(x, np.zeros(show), lw=0.6, color="tab:cyan")[0] for _ in range(N_CH)]
    for c, ln in enumerate(lines):
        if c not in proc.chans:
            ln.set_color("0.5")
    ax.set_xlim(-8, 0)
    ax.set_ylim(-sp, N_CH * sp)
    ax.set_yticks([(N_CH - 1 - c) * sp for c in range(N_CH)])
    ax.set_xlabel("seconds (0 = now)")
    lc = LineCollection([], linewidths=1.5)
    ax.add_collection(lc)
    ss_text = fig.text(0.985, 0.985, "", ha="right", va="top", fontsize=15, fontweight="bold",
                       color="white", bbox=dict(facecolor="black", alpha=0.85, edgecolor="none", pad=6))
    sel_text = fig.text(0.01, 0.985, "", ha="left", va="top", fontsize=12, fontweight="bold",
                        color="white", bbox=dict(facecolor="black", alpha=0.85, edgecolor="none", pad=6))
    t_start = time.time()

    def update(_):
        if args.seconds and time.time() - t_start > args.seconds:
            plt.close(fig)
            return
        s = proc.snapshot(show)
        ss = s["ssvep"]
        if not proc.ss_on:
            ss_text.set_text("")
        elif ss["freq"] is None:
            ss_text.set_text("SSVEP Hz: --\n(collecting %.0f s of data)" % args.ssvep_window)
            ss_text.set_color("0.7")
        else:
            ss_text.set_text(f"SSVEP Hz: {ss['freq']:.1f} Hz\nSNR {ss['snr']:.1f}  "
                             f"{'LINKED' if ss['linked'] else 'no link'}")
            ss_text.set_color("#4dff4d" if ss["linked"] else "white")
        if proc.mode == "live":
            sent = s["sent"]
            last = f"\nlast sent: {sent[-1][1]} at {time.strftime('%H:%M:%S', time.localtime(sent[-1][0]))}" if sent else ""
            sel_text.set_text(f"{s['sel']}{last}")
            sel_text.set_color("#4dff4d" if s["sel"].startswith("SENT") else "white")
        else:
            sel_text.set_text("")
        for c, ln in enumerate(lines):
            ln.set_ydata(s["data"][c] + (N_CH - 1 - c) * sp)
        labels = []
        for c in range(N_CH):
            r = s["rms"][c]
            flag = "" if 2 <= r <= 150 else "  !"
            labels.append(f"CH{c + 1}  {r:4.0f}uV{flag}" + ("  (off)" if c not in proc.chans else ""))
        ax.set_yticklabels(labels, fontsize=7)
        segs = [[(t, -sp), (t, N_CH * sp)] for t, _ in s["marks"]]
        cols = ["0.75" if lab is None else ("tab:red" if lab else "0.55") for _, lab in s["marks"]]
        lc.set_segments(segs)
        lc.set_color(cols)
        title = {"feed": "LIVE FEED  (grey lines = flash markers received)",
                 "calibrate": f"CALIBRATING on '{args.target}'  (red lines = target flashes)",
                 "live": "LIVE P300 DETECTION  (red lines = P300-like flashes)"}[proc.mode]
        st = proc.stats
        ax.set_title(f"{title}   |   {s['sps']:.1f} SPS   markers {st['markers']}   "
                     f"epochs {st['epochs']}   rejected {st['rejected']}", fontsize=10)

        ax_s.clear(); ax_r.clear(); ax_c.clear()
        if proc.mode == "live":
            sc = s["flashes"]
            ax_s.bar(range(len(sc)), sc, color=["tab:red" if v > args.thresh else "0.6" for v in sc])
            ax_s.axhline(args.thresh, color="k", lw=0.6)
            ax_s.set_title("score of each recent flash (red = P300-like)", fontsize=9)
            for a, v, nm, rng_ in ((ax_r, s["rm"], "row", s["est"][0]), (ax_c, s["cm"], "column", s["est"][1])):
                vals = np.nan_to_num(v)
                a.bar(range(6), vals, color=["tab:red" if i == rng_ else "0.6" for i in range(6)])
                a.set_xticks(range(6))
                a.set_title(f"mean score per {nm} (0 = top/left)", fontsize=9)
            r, c, letter = s["est"]
            if letter:
                fig.suptitle(f"Estimated flight: {letter}   (row {r}, column {c})",
                             fontsize=16, color="tab:red", y=0.925)
        else:
            for a in (ax_s, ax_r, ax_c):
                a.axis("off")
            msg = ("Calibration: stare at the target character and silently count its flashes.\n"
                   "Close this window when done to train and save the model."
                   if proc.mode == "calibrate" else
                   "Feed only. Check every channel is steady and shows ~2-150 uV (marked ! if not).")
            ax_s.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, transform=ax_s.transAxes)

    def on_key(e):
        if e.key == "r":
            proc.reset_scores()
        elif e.key == "l" and proc.sim:
            proc.sim.looking = not proc.sim.looking
            print(f"[Sim] subject is now {'LOOKING at' if proc.sim.looking else 'looking away from'} the board")
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.06, hspace=0.5)
    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()
    return anim


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="record labeled epochs, then train + save a model")
    ap.add_argument("--target", type=str.upper, help="flight you are staring at during calibration, e.g. DL111")
    ap.add_argument("--model", help="model file to use for live detection")
    ap.add_argument("--out", default="p300_model.pkl", help="where calibration saves the model")
    ap.add_argument("--sim", action="store_true", help="simulated board instead of Bluetooth")
    ap.add_argument("--sim-target", type=str.upper, help="(sim) flight the simulated subject attends")
    ap.add_argument("--no-plot", action="store_true", help="terminal only")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--port", type=int, default=5005, help="UDP port for flash markers")
    ap.add_argument("--exclude", default="", help="bad channels to ignore, 1-based, e.g. 4,7")
    ap.add_argument("--reject", type=float, default=150.0, help="drop epochs with peak-to-peak above this (uV)")
    ap.add_argument("--thresh", type=float, default=0.0, help="score above this counts as P300-like")
    ap.add_argument("--rounds", type=int, default=5, help="how many recent scores per row/col to average")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="shift marker times (see notes)")
    ap.add_argument("--spacing", type=float, default=60.0, help="plot spacing between channels (uV)")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--csv", help="also record the raw stream to this CSV (quickstart format)")
    ap.add_argument("--hud", action="store_true", help="also print the quickstart's status line")
    ap.add_argument("--no-ssvep", action="store_true", help="turn the SSVEP readout off")
    ap.add_argument("--ssvep-hz", type=float, default=15.0,
                    help="frequency of the speller's border response (must match the speller)")
    ap.add_argument("--ssvep-window", type=float, default=3.0, help="seconds of EEG used per SSVEP estimate")
    ap.add_argument("--ssvep-width", type=float, default=3.0, help="search +-this many Hz around --ssvep-hz")
    ap.add_argument("--ssvep-snr", type=float, default=3.0, help="SNR needed to call it LINKED")
    ap.add_argument("--sim-away", action="store_true", help="(sim) subject starts out not looking at the board")
    ng = ap.add_argument_group("sending flights to the phone (ntfy)")
    ng.add_argument("--topic", default=os.environ.get("NTFY_TOPIC"),
                    help="your ntfy topic (or set NTFY_TOPIC). Without one, notifications are only printed")
    ng.add_argument("--ntfy-server", default="https://ntfy.sh")
    ng.add_argument("--ntfy-click", help="URL to open when the notification is tapped (optional)")
    ng.add_argument("--flights-csv", help="path to flights.csv (default: next to ntfy_test.py)")
    ng.add_argument("--ntfy-dry-run", action="store_true", help="print the notification, do not send it")
    ng.add_argument("--no-ntfy", action="store_true", help="never send or print notifications")
    ng.add_argument("--hold-s", type=float, default=3.0,
                    help="the estimated flight must stay the same this many seconds before it is sent")
    ng.add_argument("--min-rounds", type=int, default=2,
                    help="scores needed for EVERY row and column before an estimate counts (2-3 = safer, slower)")
    ng.add_argument("--no-require-link", dest="require_link", action="store_false",
                    help="send even when the SSVEP link is missing (P300 alone decides)")
    args = ap.parse_args()

    if args.calibrate and (not args.target or args.target not in CHARS):
        ap.error("--calibrate needs --target with a flight from the grid, e.g. --target DL111")
    if args.calibrate and args.model:
        ap.error("use either --calibrate or --model, not both")
    model = None
    if args.model:
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded model {args.model} (trained on '{model.get('target')}', {len(model['chans'])} channels)")
        if tuple(model.get("band", ())) != BAND:
            print(f"WARNING: this model used a {model.get('band')} Hz filter but this version uses {BAND}. "
                  "Recalibrate for reliable results.")

    notifier = None
    if not args.no_ntfy:
        nt = import_ntfy_test()
        if nt is None:
            print("WARNING: ntfy_test.py not found next to this file (or in ./ntfy) - notifications disabled.")
        else:
            dry = args.ntfy_dry_run or not args.topic
            if not args.topic and not args.ntfy_dry_run:
                print("NOTE: no --topic (or NTFY_TOPIC), so notifications will only be PRINTED, not sent.")
            notifier = Notifier(nt, args.flights_csv, args.topic, args.ntfy_server, args.ntfy_click, dry)
    listener = MarkerListener(port=args.port)
    q = queue.Queue()
    monitor = Tap(q, args.hud, csv_path=args.csv)
    sim = None
    if args.sim:
        sim_char = args.sim_target or args.target or CHARS[0]
        if sim_char not in CHARS:
            ap.error("--sim-target must be a flight from the grid, e.g. DL117")
        sim = Sim(sim_char, args.ssvep_hz, looking=not args.sim_away)
    proc = Processor(args, q, listener, sim, model, notifier)
    if model and model.get("notch") != proc.notch:
        print(f"WARNING: model was trained with notch {model.get('notch')} but this run uses {proc.notch} "
              "(check --ssvep-hz / --no-ssvep). Recalibrate for reliable results.")

    def producer():
        try:
            if args.sim:
                asyncio.run(run_sim(monitor, sim))
            else:
                print("[BLE] Board must be powered from the POWER BANK while on your head, and "
                      "disconnected from the PiEEG website.")
                asyncio.run(oq.run_ble(monitor))
                print("[BLE] Board disconnected.")
        except SystemExit:
            print("[BLE] No board found. Use --sim to test without hardware.")
            proc.stop.set()
        except Exception as e:                                       # noqa: BLE001
            print(f"[source error] {e!r}")
            proc.stop.set()

    threading.Thread(target=producer, daemon=True).start()
    worker = threading.Thread(target=proc.run, daemon=True)
    worker.start()
    print(f"Mode: {proc.mode}. Listening for flash markers on UDP port {args.port}.", flush=True)
    if proc.mode == "live":
        print(f"[SELECT] will send a flight after it stays the estimate for {args.hold_s:g} s"
              f"{' with the SSVEP link' if proc.ss_on and args.require_link else ' (no link required)'}; "
              f"needs {args.min_rounds}+ score(s) per row/column. "
              f"ntfy: {'off' if not notifier else ('dry run' if notifier.dry else 'topic ' + args.topic)}", flush=True)
    elif notifier:
        print("(notifications only happen in live mode: add --model)")
    t0 = time.time()
    try:
        if args.no_plot:
            while worker.is_alive() and not (args.seconds and time.time() - t0 > args.seconds):
                time.sleep(0.2)
        else:
            run_plot(proc, args)
    except KeyboardInterrupt:
        pass
    finally:
        proc.stop.set()
        time.sleep(0.2)
        if args.calibrate:
            train_and_save(proc, args)
        monitor.close()
        listener.close()
        if proc.sent:
            print("Flights sent this session: " + ", ".join(
                f"{c} @ {time.strftime('%H:%M:%S', time.localtime(t))}" for t, c in proc.sent))
        print(f"Stopped. (stream samples filled for lost packets: {monitor.filled})")


if __name__ == "__main__":
    main()
