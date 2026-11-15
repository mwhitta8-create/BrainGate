#!/usr/bin/env python3
"""
Live blink detection on top of the PiEEG python_quickstart.py stream.

Put this file in the same folder as python_quickstart.py and live_blink_detector.py.

    python live_blink_octopus.py --sim                 # quickstart-style simulator + injectable blinks
    python live_blink_octopus.py --sim --auto          # plus a random blink every 3-6 s
    python live_blink_octopus.py --sim --no-plot --seconds 30
    python live_blink_octopus.py                       # real Octopus 16 over Bluetooth (needs bleak)
    python live_blink_octopus.py --sim --csv out.csv   # also record the stream

Press SPACE in the plot window to inject a blink (simulator only).
"""
import argparse
import asyncio
import math
import queue
import random
import threading
import time

import numpy as np

import python_quickstart as oq
from live_blink_detector import FS, N_CH, Stream, run_plot


class TapMonitor(oq.StreamMonitor):
    """The quickstart's StreamMonitor (CSV, packet loss, ...), plus a copy of every
    sample put on a queue for the blink detector."""

    def __init__(self, q, show_hud=False, **kw):
        super().__init__(**kw)
        self.q = q
        self.show_hud = show_hud

    def ingest(self, counter, channels):
        super().ingest(counter, channels)
        self.q.put(channels)

    def print_hud(self):
        if self.show_hud:
            super().print_hud()


class BlinkInjector:
    """Lets the simulator add blink-shaped bumps on request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.injected = []

    def trigger(self):
        with self.lock:
            self.requests += 1
            self.injected.append(time.time())

    def take(self):
        with self.lock:
            n, self.requests = self.requests, 0
        return n


class QueueSource:
    """Adapts the monitor's sample queue to the read() interface the detector uses."""

    def __init__(self, q, injector=None):
        self.q = q
        self.injector = injector

    @property
    def injected(self):
        return self.injector.injected if self.injector else []

    def trigger_blink(self):
        if self.injector:
            self.injector.trigger()

    def read(self):
        try:
            rows = [self.q.get(timeout=0.5)]
        except queue.Empty:
            return None
        time.sleep(0.04)                      # batch ~10 samples per chunk
        while True:
            try:
                rows.append(self.q.get_nowait())
            except queue.Empty:
                break
        return np.array(rows, dtype=float).T  # (16, n) in uV


async def sim_with_blinks(monitor, injector, auto):
    """Same synthetic signal as python_quickstart.py --sim, plus blinks."""
    print("[Simulator] 250 Hz 16-channel stream. SPACE in the plot (or --auto) injects blinks.")
    weights = np.linspace(1.0, 0.3, N_CH)     # blinks strongest on the first channels
    active = []                               # sample index where each blink started
    counter = 0
    sample_idx = 0
    dt = 1.0 / oq.NOMINAL_SPS
    next_t = time.perf_counter()
    next_auto = time.time() + random.uniform(3, 6)

    while True:
        now = time.perf_counter()
        while now >= next_t:
            t = sample_idx * dt
            if auto and time.time() >= next_auto:
                injector.trigger()
                next_auto = time.time() + random.uniform(3, 6)
            for _ in range(injector.take()):
                active.append(sample_idx)

            alpha_amp = 32.0 if (int(t) // 6) % 2 == 1 else 6.0
            base = (
                6.0 * math.sin(2 * math.pi * 2.5 * t)
                + 5.0 * math.sin(2 * math.pi * 6.0 * t)
                + alpha_amp * math.sin(2 * math.pi * 10.2 * t)
                + 6.0 * math.sin(2 * math.pi * 20.0 * t)
            )
            blink = 0.0
            for s in active:
                d = (sample_idx - s) * dt
                blink += 150.0 * math.exp(-((d - 0.2) ** 2) / (2 * 0.06 ** 2))
            active = [s for s in active if (sample_idx - s) * dt < 0.6]

            channels = [
                round(base * (0.9 + 0.015 * ch) + blink * weights[ch] + random.uniform(-4.0, 4.0), 2)
                for ch in range(N_CH)
            ]
            monitor.ingest(counter, channels)
            counter = (counter + 1) & 0xFF
            sample_idx += 1
            next_t += dt
        await asyncio.sleep(0.002)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of a real board")
    ap.add_argument("--auto", action="store_true", help="(sim) inject a random blink every 3-6 s")
    ap.add_argument("--no-plot", action="store_true", help="terminal only (sim implies --auto)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--csv", type=str, default=None, help="also record the stream to CSV")
    ap.add_argument("--hud", action="store_true", help="also print the quickstart's once-a-second status line")
    args = ap.parse_args()

    q = queue.Queue()
    monitor = TapMonitor(q, show_hud=args.hud, csv_path=args.csv)
    injector = BlinkInjector() if args.sim else None
    source = QueueSource(q, injector)
    stream = Stream(source)

    def producer():
        try:
            if args.sim:
                asyncio.run(sim_with_blinks(monitor, injector, args.auto or args.no_plot))
            else:
                asyncio.run(oq.run_ble(monitor))
        except SystemExit:
            print("[BLE] No board found. Run with --sim to use the simulator.")
        except Exception as e:                                    # noqa: BLE001
            print(f"[source error] {e!r}")
        finally:
            stream.stop.set()

    threading.Thread(target=producer, daemon=True).start()
    worker = threading.Thread(target=stream.loop, args=(args.seconds,), daemon=True)
    worker.start()
    print("Waiting for blinks...", flush=True)
    try:
        if args.no_plot:
            while worker.is_alive():
                time.sleep(0.2)
        else:
            run_plot(stream, source)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop.set()
        monitor.close()
        print(f"Done. Blinks detected: {stream.blink_count}"
              + (f", injected: {len(source.injected)}" if args.sim else ""))


if __name__ == "__main__":
    main()
