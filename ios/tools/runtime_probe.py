#!/usr/bin/env python3
"""Read the duck's own runtime (robotd, through its web console's socket on port 8090) without moving anything.

It puts the duck in Run through the studio (the same call as the app's Setup / Run switch), says hello, subscribes
to the state stream for a few seconds, asks for health, and puts the duck back the way it was. It never sends
`robot.init`, `robot.enable`, `robot.move` or a skill, so no joint is powered by this script.

    ios/tools/runtime_probe.py [--host radxa-zero3.local] [--seconds 5] [--keep] [--out logs/ios/runtime_probe.json]
"""
import argparse
import json
import time
import urllib.error
import urllib.request

from websockets.sync.client import connect

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="radxa-zero3.local")
ap.add_argument("--seconds", type=float, default=5)
ap.add_argument("--hz", type=int, default=10)
ap.add_argument("--keep", action="store_true", help="leave the duck in Run")
ap.add_argument("--out", default="")
a = ap.parse_args()


def studio(name, body=None):
    r = urllib.request.Request(f"http://{a.host}:8770/api/{name}", json.dumps(body).encode() if body is not None else None)
    try:
        return json.loads(urllib.request.urlopen(r, timeout=45).read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"studio {name}: {e.read().decode()[:300]}")


was = studio("state")["brain"]["sensors"].get("mode")
print("mode before:", was)
if was != "run":
    t = time.time()
    studio("mode", dict(to="run"))
    print(f"to run: {time.time() - t:.2f} s")
out = dict(mode_before=was)
try:
    meta = json.loads(urllib.request.urlopen(f"http://{a.host}:8090/api/meta", timeout=5).read())
    out["meta"] = meta
    ws, n = None, 0
    for _ in range(20):                                  # the socket appears a moment after the runtime starts
        try:
            ws = connect(f"ws://{a.host}:8090/ws", open_timeout=3)
            break
        except OSError as e:
            print("socket:", e)
            time.sleep(0.5)

    def call(method, params=None):
        global n
        n += 1
        ws.send(json.dumps(dict(jsonrpc="2.0", id=n, method=method, params=params or {})))
        while True:
            m = json.loads(ws.recv(timeout=8))
            if m.get("id") == n:
                return m.get("result", m.get("error"))

    out["hello"] = call("hello", dict(api_version=meta["api_version"]))
    out["health"] = call("robot.health")
    out["subscribe"] = call("robot.subscribe", dict(hz=a.hz))
    states, t0 = [], time.time()
    while time.time() - t0 < a.seconds:
        m = json.loads(ws.recv(timeout=3))
        if m.get("method") == "robot.state":
            states.append((time.time() - t0, m["params"]))
    out["states"] = len(states)
    out["state_hz"] = (len(states) - 1) / (states[-1][0] - states[0][0]) if len(states) > 1 else 0
    out["first"], out["last"] = states[0][1], states[-1][1]
    ws.close()
except Exception as e:                                   # report it, and still put the duck back
    print("probe failed:", type(e).__name__, e)
finally:
    now = studio("state")["brain"]["sensors"].get("mode")
    print("mode after the read:", now)
    if was != "run" and now == "run" and not a.keep:
        studio("mode", dict(to=was or "setup"))
        print("back to", was)
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
if a.out:
    open(a.out, "w").write(json.dumps(out, indent=1, ensure_ascii=False))
