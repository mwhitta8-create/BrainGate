#!/usr/bin/env python3
"""
Live blink detector for a 16-channel, 250 Hz EEG stream.

Runs a built-in simulator that streams data in real time (like the board would),
detects blinks as the data arrives, and shows a live plot.

    python live_blink_detector.py              # live plot, press SPACE to inject a blink
    python live_blink_detector.py --auto       # also inject a random blink every 3-6 s
    python live_blink_detector.py --no-plot --seconds 30   # terminal only (auto blinks)

Needs: numpy, scipy, matplotlib (matplotlib only for the plot).

To use real data later, write a class with the same read() method as SimSource
(returns an array of shape (16, n_new_samples) in microvolts) and pass it to run().
"""
import argparse
import threading
import time
import random

import numpy as np
from scipy import signal

FS = 250            # samples per second
N_CH = 16
CHUNK = 25          # samples per update (0.1 s)

# ---- detector tuning ----
BAND = (1.0, 10.0)      # Hz; blinks are slow, big deflections
MIN_THRESH_UV = 40.0    # never trigger below this envelope level
K_MAD = 8.0             # adaptive threshold = median + K * MAD of recent envelope
BASELINE_SEC = 10.0     # window used for the adaptive baseline
REFRACTORY_SEC = 0.4    # ignore new blinks this soon after one
WARMUP_SEC = 2.0        # let filters settle before detecting


# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------
class SimSource:
    """Simulated board. read() blocks so data arrives in real time (250 Hz)."""

    def __init__(self, auto=False, seed=None):
        self.rng = np.random.default_rng(seed)
        self.auto = auto
        self.lock = threading.Lock()
        self.pending = np.zeros((N_CH, FS * 2))      # future blink signal to add
        self.sample_index = 0
        self.alpha_phase = self.rng.uniform(0, 2 * np.pi, N_CH)
        self.next_t = None
        self.next_auto = time.time() + random.uniform(3, 6)
        # blinks are biggest on the first channels, weaker on the rest
        self.weights = np.linspace(1.0, 0.3, N_CH)[:, None]
        t = np.arange(int(0.5 * FS)) / FS
        self.template = 150.0 * np.exp(-((t - 0.2) ** 2) / (2 * 0.06 ** 2))  # ~150 uV, ~0.15 s wide
        self.injected = []                            # wall-clock times of injected blinks

    def trigger_blink(self):
        with self.lock:
            self.pending[:, : len(self.template)] += self.weights * self.template
            self.injected.append(time.time())

    def read(self):
        # pace to real time
        if self.next_t is None:
            self.next_t = time.time()
        self.next_t += CHUNK / FS
        delay = self.next_t - time.time()
        if delay > 0:
            time.sleep(delay)

        if self.auto and time.time() >= self.next_auto:
            self.trigger_blink()
            self.next_auto = time.time() + random.uniform(3, 6)

        n = np.arange(self.sample_index, self.sample_index + CHUNK) / FS
        self.sample_index += CHUNK
        alpha = 8.0 * np.sin(2 * np.pi * 10 * n[None, :] + self.alpha_phase[:, None])
        drift = 20.0 * np.sin(2 * np.pi * 0.3 * n)[None, :]
        noise = self.rng.normal(0, 6.0, (N_CH, CHUNK))
        data = alpha + drift + noise
        with self.lock:
            data += self.pending[:, :CHUNK]
            self.pending = np.roll(self.pending, -CHUNK, axis=1)
            self.pending[:, -CHUNK:] = 0
        return data


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class BlinkDetector:
    def __init__(self):
        self.sos = signal.butter(4, BAND, btype="bandpass", fs=FS, output="sos")
        self.zi = np.zeros((self.sos.shape[0], N_CH, 2))
        self.env_hist = np.zeros(int(BASELINE_SEC * FS))
        self.hist_len = 0
        self.in_blink = False
        self.last_blink_t = -1e9
        self.samples_seen = 0
        self.threshold = MIN_THRESH_UV

    def process(self, chunk):
        """chunk: (N_CH, n) in uV. Returns (envelope, threshold_array, detection_sample_offsets)."""
        filt, self.zi = signal.sosfilt(self.sos, chunk, axis=1, zi=self.zi)
        # median across channels of |signal|: robust to one bad/noisy pin
        env = np.median(np.abs(filt), axis=0)
        thr_arr = np.empty_like(env)
        detections = []
        for i, e in enumerate(env):
            self.env_hist = np.roll(self.env_hist, -1)
            self.env_hist[-1] = e
            self.hist_len = min(self.hist_len + 1, len(self.env_hist))
            hist = self.env_hist[-self.hist_len:]
            med = np.median(hist)
            mad = np.median(np.abs(hist - med))
            self.threshold = max(MIN_THRESH_UV, med + K_MAD * 1.4826 * mad)
            thr_arr[i] = self.threshold

            t = (self.samples_seen + i) / FS
            if t < WARMUP_SEC:
                continue
            if not self.in_blink and e > self.threshold and (t - self.last_blink_t) > REFRACTORY_SEC:
                self.in_blink = True
                self.last_blink_t = t
                detections.append(i)
            elif self.in_blink and e < 0.5 * self.threshold:
                self.in_blink = False
        self.samples_seen += chunk.shape[1]
        return env, thr_arr, detections


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
class Stream:
    """Reads the source in a background thread, runs the detector, keeps display buffers."""

    def __init__(self, source, show_sec=6):
        self.source = source
        self.det = BlinkDetector()
        self.n_show = show_sec * FS
        self.raw = np.zeros((N_CH, self.n_show))
        self.env = np.zeros(self.n_show)
        self.thr = np.full(self.n_show, MIN_THRESH_UV)
        self.marks = np.zeros(self.n_show, dtype=bool)
        self.blink_count = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.start_time = time.time()

    def loop(self, max_seconds=None):
        while not self.stop.is_set():
            chunk = self.source.read()
            env, thr, dets = self.det.process(chunk)
            n = chunk.shape[1]
            with self.lock:
                self.raw = np.roll(self.raw, -n, axis=1); self.raw[:, -n:] = chunk
                self.env = np.roll(self.env, -n); self.env[-n:] = env
                self.thr = np.roll(self.thr, -n); self.thr[-n:] = thr
                self.marks = np.roll(self.marks, -n); self.marks[-n:] = False
                for d in dets:
                    self.marks[-n + d] = True
                    self.blink_count += 1
            for d in dets:
                print(f"[{time.strftime('%H:%M:%S')}] BLINK #{self.blink_count}  "
                      f"(env {env[d]:.0f} uV, threshold {thr[d]:.0f} uV)", flush=True)
            if max_seconds and time.time() - self.start_time > max_seconds:
                self.stop.set()


def run_plot(stream, source):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    t = np.linspace(-stream.n_show / FS, 0, stream.n_show)
    offset = 150.0
    lines = [ax1.plot(t, np.zeros_like(t), lw=0.7)[0] for _ in range(N_CH)]
    ax1.set_ylim(-offset, offset * N_CH)
    ax1.set_yticks([i * offset for i in range(N_CH)])
    ax1.set_yticklabels([f"CH{i + 1}" for i in range(N_CH)], fontsize=7)
    ax1.set_title("Live EEG (raw, uV)  -  SPACE = inject simulated blink  -  close window to quit")
    (env_line,) = ax2.plot(t, stream.env, color="tab:blue", label="blink envelope")
    (thr_line,) = ax2.plot(t, stream.thr, color="tab:orange", ls="--", label="threshold")
    (mark_pts,) = ax2.plot([], [], "rv", ms=10, label="blink detected")
    ax2.set_ylim(0, 150)
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("uV")
    ax2.legend(loc="upper left", fontsize=8)
    status = ax2.text(0.99, 0.9, "", transform=ax2.transAxes, ha="right", fontsize=12)

    def on_key(event):
        if event.key == " " and hasattr(source, "trigger_blink"):
            source.trigger_blink()
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        with stream.lock:
            raw = stream.raw.copy(); env = stream.env.copy()
            thr = stream.thr.copy(); marks = stream.marks.copy()
            count = stream.blink_count
        for i, ln in enumerate(lines):
            ln.set_ydata(raw[i] - raw[i].mean() + i * offset)
        env_line.set_ydata(env)
        thr_line.set_ydata(thr)
        idx = np.where(marks)[0]
        mark_pts.set_data(t[idx], env[idx])
        status.set_text(f"blinks detected: {count}")
        return lines + [env_line, thr_line, mark_pts, status]

    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="inject a random simulated blink every 3-6 s")
    ap.add_argument("--no-plot", action="store_true", help="terminal only (implies --auto)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    args = ap.parse_args()

    source = SimSource(auto=args.auto or args.no_plot)
    stream = Stream(source)
    th = threading.Thread(target=stream.loop, args=(args.seconds,), daemon=True)
    th.start()
    print("Streaming simulated 16-ch EEG at 250 Hz. Waiting for blinks...", flush=True)
    try:
        if args.no_plot:
            while th.is_alive():
                time.sleep(0.2)
        else:
            run_plot(stream, source)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop.set()
        print(f"Done. Blinks detected: {stream.blink_count}, "
              f"injected: {len(source.injected)}")


if __name__ == "__main__":
    main()
