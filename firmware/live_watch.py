"""Start the policies live on the duck and keep them running, watched from the Mac. Stops the run (joints limp) only
if the battery sags too low, a servo or the board gets too hot, or the runner itself reports an error.

    python3 firmware/live_watch.py [--volts 6.6] [--servo 65] [--board 100]

Every 2 s it appends one line to docs/live_watch.log: time, tilt, state, falls, loop rate, battery volts, hottest
servo, board temperature. Drive it meanwhile from the web GUI's Drive panel. Ctrl+C (or Stop in the GUI) ends it.
"""
import argparse, json, time, urllib.request
from pathlib import Path

BUS, WEB = "http://127.0.0.1:8772", "http://127.0.0.1:8770"
get = lambda u: json.load(urllib.request.urlopen(u, timeout=8))
post = lambda p: urllib.request.urlopen(urllib.request.Request(BUS + p, data=b"{}"), timeout=10).read()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--volts", type=float, default=6.6, help="stop below this for 3 readings (2 cells: 3.3 V each under load)")
    ap.add_argument("--servo", type=float, default=65.0)
    ap.add_argument("--board", type=float, default=100.0)
    a = ap.parse_args()
    log = Path(__file__).resolve().parent.parent / "docs/live_watch.log"
    r = json.loads(post("/pilot/start/live"))
    if "error" in r:
        raise SystemExit("not started: " + r["error"])
    low, why, t0 = 0, None, time.time()
    try:
        while why is None:
            time.sleep(2)
            try:
                s = get(BUS + "/status")
                m = get(WEB + "/api/state")["motors"]
            except Exception as e:
                print("no reading:", e, flush=True)
                continue
            p, hot = s.get("pilot"), max((x["temp"] for x in m.values()), default=0)
            volts = min((x["volt"] for x in m.values()), default=s.get("volts") or 0)
            line = dict(t=round(time.time() - t0), tilt=p and p.get("tilt"), state=p and p.get("state"), falls=p and p.get("falls"),
                        hz=p and p.get("hz"), missed=p and p.get("missed"), volts=volts, servo_c=hot, board_c=s.get("temp"))
            with open(log, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(line, flush=True)
            low = low + 1 if 0 < volts < a.volts else 0
            why = ("the runner ended: " + str(p and p.get("error")) if not p or p.get("ended") else
                   f"battery below {a.volts} V" if low >= 3 else f"a servo at {hot} C" if hot >= a.servo else
                   f"board at {s.get('temp')} C" if (s.get("temp") or 0) >= a.board else None)
    except KeyboardInterrupt:
        why = "stopped by hand"
    finally:
        post("/pilot/stop")
    print("ended:", why, flush=True)
