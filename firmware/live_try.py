"""A first, guarded live run of the stand (walk policy, zero command) on the real duck. Run on the Mac, studio up:

    python3 firmware/live_try.py [--torque 500] [--seconds 8] [--tilt 45]

Limits every joint's torque (register 48, of 1000; back to what it was afterwards), starts the policies live, and
watches about 8 times a second: the pilot's tilt and state and every joint's angle. Stops the run (all joints limp)
if the tilt passes the limit, the runner reports an error, or the time is up. Writes docs/live_<time>.json.
"""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import duck_pilot

BUS = "http://127.0.0.1:8772"
IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14)


def post(path, body=b"{}"):
    return urllib.request.urlopen(urllib.request.Request(BUS + path, data=body), timeout=10).read()


def get(path):
    return json.load(urllib.request.urlopen(BUS + path, timeout=6))


def tx(sid, instr, params, reply=0):
    b = [sid, len(params) + 2, instr, *params]
    return post(f"/bus/{reply}/30", bytes([0xFF, 0xFF, *b, ~sum(b) & 0xFF]))


def angles(cal):
    b = [0xFE, len(IDS) + 4, 0x82, 56, 2, *IDS]
    r = post(f"/bus/{8 * len(IDS)}/30", bytes([0xFF, 0xFF, *b, ~sum(b) & 0xFF]))
    sm = lambda v: -(v & 0x7FFF) if v & 0x8000 else v
    got = {r[i + 2]: sm(r[i + 5] | r[i + 6] << 8) for i in range(0, len(r) - 7, 8) if r[i:i + 2] == b"\xff\xff"}
    return [round((got[i] - cal[i]["zero_raw"]) * 360 / 4096 * cal[i]["direction"], 1) if i in got else None for i in IDS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--torque", type=int, default=500)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--tilt", type=float, default=45.0)
    ap.add_argument("--level", action="store_true", help="once it has stood steady for 5 s, take that as level (the studio's Level), at most twice")
    ap.add_argument("--down", type=float, default=0.0, help="instead of the tilt limit: stop only after this many s past 60 deg (the walk policy gets up by itself)")
    a = ap.parse_args()
    cal = {j["id"]: j for j in get("/calibration")["joints"]}
    was = {i: int.from_bytes(tx(i, 2, [48, 2], 8)[5:7], "little") for i in IDS}
    before = angles(cal)
    log, why, levels = [], "time up", []
    try:
        for i in IDS:
            tx(i, 3, [48, a.torque & 0xFF, a.torque >> 8])
        r = json.loads(post("/pilot/start/live"))
        if "error" in r:
            raise SystemExit("not started: " + r["error"])
        t0, over, down_t, levelled, level_t, levels = time.time(), 0, None, 0, 0.0, []
        while time.time() - t0 < a.seconds:
            p = get("/status")["pilot"] or {}
            imu = get("/imu") or {}
            log.append(dict(t=round(time.time() - t0, 2), roll=imu.get("roll"), pitch=imu.get("pitch"), rate=imu.get("rate"), tilt=p.get("tilt"), state=p.get("state"), off=p.get("off_deg"), hz=p.get("hz"),
                            missed=p.get("missed"), deg=angles(cal), ask=p.get("ask_deg"), at=p.get("at_deg")))
            if a.level and levelled < 2 and len(log) > 30 and time.time() - level_t > 12:
                last = [r for r in log if r["t"] > log[-1]["t"] - 5 and r["roll"] is not None]
                if len(last) > 15 and all(r["tilt"] is not None and r["tilt"] < 25 for r in last) and \
                        max(r["roll"] for r in last) - min(r["roll"] for r in last) < 1.5 and max(r["pitch"] for r in last) - min(r["pitch"] for r in last) < 1.5:
                    was_rp = (last[-1]["roll"], last[-1]["pitch"])
                    req = urllib.request.Request("http://127.0.0.1:8770/api/cal_level", data=b"{}")
                    try:
                        urllib.request.urlopen(req, timeout=10).read()
                        levelled, level_t = levelled + 1, time.time()
                        levels.append(dict(t=log[-1]["t"], roll_was=was_rp[0], pitch_was=was_rp[1]))
                    except Exception as e:
                        levels.append(dict(t=log[-1]["t"], error=str(e)))
                        level_t = time.time()
            over = over + 1 if (p.get("tilt") or 0) > (60 if a.down else a.tilt) else 0
            down_t = (down_t or time.time()) if over else None
            if a.down:
                over = 2 if down_t and time.time() - down_t > a.down else 0
            if p.get("error") or p.get("ended"):
                why = "runner stopped: " + str(p.get("error"))
                break
            if over >= 2:
                why = f"down for {a.down} s" if a.down else f"tilt past {a.tilt} deg"
                break
    finally:
        falls = (get("/status")["pilot"] or {}).get("falls")
        post("/pilot/stop")
        for i in IDS:
            tx(i, 3, [48, was[i] & 0xFF, was[i] >> 8])
    out = HERE.parent / f"docs/live_{time.strftime('%H%M%S')}.json"
    out.write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), torque=a.torque, ended=why, joints=duck_pilot.JOINTS,
                                   default_deg=[round(math.degrees(v), 1) for v in duck_pilot.DEFAULT_POSE], before=before, log=log), indent=1))
    print("ended:", why, "| samples", len(log), "| torque limit", a.torque, "(was", sorted(set(was.values())), ")")
    print("tilt over time:", [(r["t"], r["tilt"], r["state"]) for r in log][::4])
    up = [r for r in log if r["tilt"] is not None and r["tilt"] < 20]
    print(f"upright (tilt < 20 deg) in {len(up)} of {len(log)} samples; falls counted by the pilot:", falls)
    print("roll, pitch over time:", [(r["t"], r["roll"], r["pitch"]) for r in log][::4])
    print("levelled while standing:", levels if a.level else "off")
    print("loop:", log[-1]["hz"] if log else None, "Hz, missed", log[-1]["missed"] if log else None)
    print(f"{'joint':<16}{'before':>8}{'at 2 s':>8}{'last':>8}{'default':>9}")
    mid = next((r for r in log if r["t"] >= 2.2), log[-1] if log else None)
    for k, n in enumerate(duck_pilot.JOINTS):
        print(f"{n:<16}{before[k]!s:>8}{(mid['deg'][k] if mid else None)!s:>8}{(log[-1]['deg'][k] if log else None)!s:>8}{math.degrees(duck_pilot.DEFAULT_POSE[k]):>9.1f}")
    print(out)
