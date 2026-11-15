#!/usr/bin/env python3
"""
ntfy_test.py - pick a flight, send its details to your phone through ntfy.

    python ntfy_test.py --topic my-long-random-topic-name
    python ntfy_test.py --flight DL111            # skip the prompt
    python ntfy_test.py --dry-run                 # print the notification, send nothing

Setup (once): install the ntfy app on your phone and subscribe to your topic. There is no
sign-up on the public server (ntfy.sh), so the topic name acts like a password: make it
long and random. You can also set it once with the NTFY_TOPIC environment variable.

All flight data is FAKE, read from flights.csv (same folder). The first 36 rows are the P300
grid in order (row 0 = top, column 0 = left); the other 36 are extras you can type by code.

------------------------------------------------------------------------------
VARIABLES
Stored in flights.csv (one row per flight):
  code                   flight code as shown on the grid, e.g. DL111
  on_grid                yes/no - is it one of the 36 flights on the P300 grid
  grid_row, grid_col     grid position (0-based; row 0 = top, column 0 = leftmost)
  airline_code           DL, AA, UA, WN, B6, AS
  airline                airline name
  flight_number          digits after the airline code
  origin_iata/city       always DTW / Detroit (the "home" airport of the board)
  dest_iata, dest_city   destination
  dest_tz                destination time zone (used to show the arrival in local time)
  dep_terminal           departure terminal (McNamara or North)
  dep_gate               departure gate
  status                 On time / Delayed / Boarding / Final call
  delay_min              minutes late (0 if not delayed)
  dep_in_min             minutes from "now" (when you send) to the scheduled departure
  boarding_min_before    boarding opens this many minutes before departure
  gate_close_min_before  gate closes this many minutes before departure
  distance_mi            approximate distance
  duration_min           scheduled flight time in minutes
  aircraft               aircraft type
  meal_service           what is served on board
  wifi                   Yes / No
  arr_terminal           arrival terminal
  baggage_claim          baggage carousel number

Worked out when you send (so the times always look current):
  scheduled departure    now + dep_in_min, rounded to 5 minutes (departure airport's local time)
  estimated departure    scheduled departure + delay_min
  boarding time          estimated departure - boarding_min_before
  gate close time        estimated departure - gate_close_min_before
  arrival time           estimated departure + duration_min, shown in the destination's local time
  flight time            duration_min as "1h 53m"

The notification:
  title      "DL111 to ATL"
  body       "Gate B12 · Boards 2:35 PM · Departs 3:05 PM" followed by airline/aircraft,
             route, terminal and gate-close time, arrival time/duration/distance,
             arrival terminal and baggage claim, and status/meal/Wi-Fi
  tags       an airplane emoji plus one for the status (check mark / warning / alarm)
  priority   3 normally, 4 (high) when Boarding or Final call
------------------------------------------------------------------------------
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flights.csv")

# Fixed UTC offsets (standard time, no daylight saving) used only if the system has no
# time zone database (common on Windows: `pip install tzdata` fixes it).
STD_OFFSETS = {"America/New_York": -5, "America/Chicago": -6, "America/Denver": -7,
               "America/Phoenix": -7, "America/Los_Angeles": -8}
_warned_tz = False


def load_flights(path):
    with open(path, newline="", encoding="utf-8") as f:
        flights = list(csv.DictReader(f))
    for r in flights:
        for k in ("delay_min", "dep_in_min", "boarding_min_before", "gate_close_min_before",
                  "distance_mi", "duration_min"):
            r[k] = int(r[k])
    return flights


def fmt_time(dt):
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_duration(minutes):
    h, m = divmod(minutes, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def to_dest_local(utc_dt, tz_name):
    """Convert a UTC datetime to the destination's local time."""
    global _warned_tz
    try:
        from zoneinfo import ZoneInfo
        return utc_dt.astimezone(ZoneInfo(tz_name))
    except Exception:                                   # no tz database on this machine
        if not _warned_tz:
            print("(note: no time zone database found, arrival times ignore daylight saving; "
                  "run `pip install tzdata` for exact local times)")
            _warned_tz = True
        return utc_dt.astimezone(timezone(timedelta(hours=STD_OFFSETS.get(tz_name, 0))))


def build_notification(f, now=None):
    """Turn one flight row into {title, message, tags, priority}."""
    now = (now or datetime.now()).astimezone()
    sched = now + timedelta(minutes=f["dep_in_min"])
    sched = sched.replace(second=0, microsecond=0)
    sched -= timedelta(minutes=sched.minute % 5)                       # round down to 5 min
    est = sched + timedelta(minutes=f["delay_min"])
    boards = est - timedelta(minutes=f["boarding_min_before"])
    closes = est - timedelta(minutes=f["gate_close_min_before"])
    arrive = to_dest_local(est.astimezone(timezone.utc) + timedelta(minutes=f["duration_min"]),
                           f["dest_tz"])

    status = f["status"]
    if status == "Delayed":
        status_line = f"Delayed {f['delay_min']} min"
        dep_txt = f"Departs {fmt_time(est)} (was {fmt_time(sched)})"
    else:
        status_line = status
        dep_txt = f"Departs {fmt_time(est)}"
    boards_txt = "Boarding now" if status == "Boarding" else (
        "Final call" if status == "Final call" else f"Boards {fmt_time(boards)}")

    lines = [
        f"Gate {f['dep_gate']} · {boards_txt} · {dep_txt}",
        f"{f['airline']} {f['flight_number']} · {f['aircraft']}",
        f"{f['origin_city']} ({f['origin_iata']}) to {f['dest_city']} ({f['dest_iata']})",
        f"Terminal {f['dep_terminal']} · Gate closes {fmt_time(closes)}",
        f"Arrives {fmt_time(arrive)} {f['dest_iata']} time · {fmt_duration(f['duration_min'])} · "
        f"{f['distance_mi']} mi",
        f"Arrival Terminal {f['arr_terminal']} · Baggage claim {f['baggage_claim']}",
        f"{status_line} · {f['meal_service']} · Wi-Fi: {f['wifi']}",
    ]
    status_tag = {"On time": "white_check_mark", "Delayed": "warning",
                  "Boarding": "bell", "Final call": "rotating_light"}.get(status, "information_source")
    return {
        "title": f"{f['code']} to {f['dest_iata']}",
        "message": "\n".join(lines),
        "tags": ["airplane", status_tag],
        "priority": 4 if status in ("Boarding", "Final call") else 3,
    }


def send(note, topic, server="https://ntfy.sh", click=None, timeout=10):
    """Publish as JSON (so emoji and symbols are fine). Returns True on success."""
    payload = dict(note, topic=topic)
    if click:
        payload["click"] = click
    req = urllib.request.Request(server.rstrip("/"), data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        print(f"ntfy refused the message: HTTP {e.code} {e.reason}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"could not reach {server}: {e}")
    return False


def show_grid(flights):
    print("\nP300 grid (type a code, or a number 1-36 counting left to right, top to bottom):")
    for r in range(6):
        print("  " + "   ".join(f"{r * 6 + c + 1:>2} {flights[r * 6 + c]['code']:<5}" for c in range(6)))
    extras = [f["code"] for f in flights[36:]]
    print("\nExtra flights (not on the grid, type the code): " + ", ".join(extras))


def find_flight(text, flights):
    text = text.strip().upper()
    if text.isdigit() and 1 <= int(text) <= 36:
        return flights[int(text) - 1]
    for f in flights:
        if f["code"] == text:
            return f
    return None


def main():
    ap = argparse.ArgumentParser(description="Send a flight's details to your phone via ntfy")
    ap.add_argument("--topic", default=os.environ.get("NTFY_TOPIC"), help="your ntfy topic (or set NTFY_TOPIC)")
    ap.add_argument("--server", default="https://ntfy.sh")
    ap.add_argument("--flight", help="flight code to send without prompting, e.g. DL111")
    ap.add_argument("--csv", default=CSV_PATH, help="path to flights.csv")
    ap.add_argument("--click", help="URL to open when the notification is tapped (optional)")
    ap.add_argument("--dry-run", action="store_true", help="print the notification instead of sending")
    args = ap.parse_args()

    if not args.topic and not args.dry_run:
        sys.exit("No topic. Use --topic NAME or set NTFY_TOPIC (or try --dry-run).")
    flights = load_flights(args.csv)

    def handle(f):
        note = build_notification(f)
        print(f"\n--- {note['title']}  (priority {note['priority']}, tags {','.join(note['tags'])}) ---")
        print(note["message"])
        if args.dry_run:
            return
        ok = send(note, args.topic, args.server, args.click)
        print("sent." if ok else "NOT sent.")

    if args.flight:
        f = find_flight(args.flight, flights)
        if not f:
            sys.exit(f"Unknown flight {args.flight!r}.")
        handle(f)
        return

    show_grid(flights)
    while True:
        try:
            text = input("\nFlight (code or 1-36, q to quit): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in ("q", "quit", "exit"):
            break
        f = find_flight(text, flights)
        if not f:
            print("Not in the list. Type a code like DL111 or a number from 1 to 36.")
            continue
        handle(f)


if __name__ == "__main__":
    main()
