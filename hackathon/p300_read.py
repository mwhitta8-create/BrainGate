#!/usr/bin/env python3
"""
Receives P300 flash markers sent by p300_speller.py over the network (UDP).

Each marker is a JSON list:  ["row", 2]  or  ["col", 4]
    row 0 = top row,  column 0 = leftmost column.

Run on its own to watch the markers arrive:
    python p300_read.py              # listens on port 5005
    python p300_read.py --port 6000

Use it from another script (e.g. live_blink_octopus.py later):
    from p300_read import MarkerListener
    listener = MarkerListener(port=5005)       # starts a background thread
    ...
    for t, kind, idx in listener.drain():      # call this as often as you like
        print(t, kind, idx)

t is time.time() on THIS computer when the marker arrived, the same clock the
quickstart's CSV timestamps use. Network delay (a few ms on Wi-Fi, sometimes
more) is not removed, so expect small timing jitter.
"""
import argparse
import json
import queue
import socket
import threading
import time

DEFAULT_PORT = 5005


class MarkerListener:
    def __init__(self, port=DEFAULT_PORT, host="0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                data, _addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            t = time.time()                      # stamp immediately on arrival
            try:
                kind, idx = json.loads(data.decode())
                if kind not in ("row", "col") or not isinstance(idx, int):
                    raise ValueError
            except (ValueError, TypeError):
                print(f"ignored malformed marker: {data!r}")
                continue
            self.q.put((t, kind, idx))

    def drain(self):
        """Return all markers received since the last call, as [(time, kind, index), ...]."""
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.stop_event.set()
        self.sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    listener = MarkerListener(port=args.port)
    print(f"Listening for P300 markers on UDP port {args.port} (Ctrl+C to stop)...")
    try:
        while True:
            for t, kind, idx in listener.drain():
                print(f"{time.strftime('%H:%M:%S', time.localtime(t))}.{int((t % 1) * 1000):03d}  "
                      f"[{kind!r}, {idx}]")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
