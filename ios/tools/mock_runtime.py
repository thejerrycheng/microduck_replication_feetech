#!/usr/bin/env python3
"""A stand-in for the duck's own runtime, for the simulator tests: no servo is ever involved.

The real one is Pollen's control daemon `robotd` behind its web console `webd` (port 8090 on the duck): JSON-RPC 2.0
over a WebSocket at /ws, and the joint list at /api/meta. This speaks the same messages, the ones the phone app uses
(ios/Duck/Link.swift, "the duck's runtime"), and models just enough of the robot to check the app's commands:

  * unpowered until `robot.init` (2 s ramp to the stance); `robot.relax` lets go again. As on the real daemon
    (read 2026-11-18), the state has no flag for that: unpowered it reads policy "held" with `limp` false;
  * `robot.enable {on}` hands the legs to the walk policy; only then do `robot.move` and `robot.do` act;
  * the applied velocity follows the requested one through the daemon's smoothing (cmd += 0.2 (target - cmd) at
    50 Hz), clamped to 0.3 m/s, 0.3 m/s and 1.5 rad/s, and is integrated into a planar pose (`odom`, extra);
  * a velocity not repeated within DEADMAN s is dropped to zero, as the daemon's deadman does.

Every message received is appended to --log as a JSON line `{t, method, params}`, with a `state` line at 50 Hz
`{t, req, applied, odom, policy, limp}`: ios/tools/plot_runtime.py draws them.

    ios/tools/mock_runtime.py [--port 8791] [--log logs/ios/runtime_mock.jsonl]
"""
import argparse
import http
import json
import math
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8791)
ap.add_argument("--log", default="")
a = ap.parse_args()

HZ, ALPHA, DEADMAN, LIMIT = 50, 0.2, 0.5, (0.3, 0.3, 1.5)
JOINTS = [(20, "left_hip_yaw"), (21, "left_hip_roll"), (22, "left_hip_pitch"), (23, "left_knee"), (24, "left_ankle"),
          (30, "neck_pitch"), (31, "head_pitch"), (32, "head_yaw"), (33, "head_roll"), (34, "mouth"),
          (10, "right_hip_yaw"), (11, "right_hip_roll"), (12, "right_hip_pitch"), (13, "right_knee"), (14, "right_ankle")]
META = dict(api_version=17, servo_port="mock",
            joints=[dict(index=i, id=sid, name=n, label=n, home=2048) for i, (sid, n) in enumerate(JOINTS)])
SKILLS = dict(roulade=2.0, kick_left=1.2, kick_right=1.2, ground_pick=3.0)      # how long each takes [s]


class Robot:
    def __init__(self):
        self.lock = threading.Lock()
        self.limp, self.walk, self.seated, self.rise = True, False, False, 0.0
        self.req, self.at, self.cmd, self.odom = [0.0] * 3, 0.0, [0.0] * 3, [0.0] * 3
        self.skill, self.until, self.t0 = "", 0.0, time.time()
        self.out = open(a.log, "w") if a.log else None

    def note(self, **row):
        if self.out:
            self.out.write(json.dumps(dict(t=round(time.time() - self.t0, 4), **row)) + "\n")
            self.out.flush()

    def policy(self):
        now = time.time()
        return ("held" if self.limp else "init" if now < self.rise else self.skill if now < self.until else
                "sitstand" if self.seated else "walk" if self.walk else "stand")

    def step(self):
        with self.lock:
            now = time.time()
            driving = self.policy() == "walk"
            want = self.req if driving and now - self.at < DEADMAN else [0.0] * 3
            want = [max(-m, min(m, v)) for v, m in zip(want, LIMIT)]
            self.cmd = [c + ALPHA * (w - c) for c, w in zip(self.cmd, want)]
            self.cmd = [0.0 if abs(c) < 1e-4 else c for c in self.cmd]
            x, y, yaw = self.odom
            vx, vy, w = self.cmd
            self.odom = [x + (vx * math.cos(yaw) - vy * math.sin(yaw)) / HZ, y + (vx * math.sin(yaw) + vy * math.cos(yaw)) / HZ,
                         yaw + w / HZ]
            self.note(method="state", req=self.req, applied=[round(c, 4) for c in self.cmd],
                      odom=[round(v, 4) for v in self.odom], policy=self.policy(), limp=self.limp)

    def state(self):
        with self.lock:
            limited = [n for n, v, m in zip(("vx", "vy", "vyaw"), self.req, LIMIT) if abs(v) > m]
            return dict(policy=self.policy(), safety=dict(fallen=False, limp=False, gravity=[0.0, 0.0, -1.0]),
                        loop=dict(hz=float(HZ), missed=0), move=dict(requested=self.req, applied=self.cmd, limited_by=limited),
                        joints=[0.0] * 15, targets=[0.0] * 15, odom=self.odom)

    def call(self, method, p):
        """The reply to a request, or None for a notification."""
        with self.lock:
            now = time.time()
            self.note(method=method, params=p)
            if method == "hello":
                return dict(api_version=17, daemon_version="mock", revision=None)
            if method == "robot.subscribe":
                return dict(accepted=True, walk="mock.onnx")
            if method == "robot.health":
                return dict(healthy=True, degraded=False, battery=dict(percent=77.0, volts=7.9),
                            motors=dict(hottest="left_knee", max_c=31.0))
            if method == "robot.init":
                if self.limp:
                    self.limp, self.rise = False, now + 2.0
                return dict(accepted=True)
            if method == "robot.relax":
                self.limp, self.walk, self.seated, self.req = True, False, False, [0.0] * 3
                return dict(accepted=True)
            if method == "robot.enable":
                if self.limp:
                    return dict(accepted=False, reason="the robot is limp")
                self.walk = bool(p.get("on")) if not p.get("toggle") else not self.walk
                return dict(accepted=True)
            if method == "robot.stop":
                self.req = [0.0] * 3
                return dict(accepted=True)
            if method == "robot.move":
                self.req, self.at = [float(p.get(k, 0)) for k in ("vx", "vy", "vyaw")], now
                return None
            if method == "robot.do":
                if self.limp or not self.walk or now < self.until:
                    return dict(accepted=False, reason="the policy is not driving")
                skill = p.get("skill")
                if skill == "sit_toggle":
                    self.seated = not self.seated
                elif skill in SKILLS:
                    self.skill, self.until = skill, now + SKILLS[skill]
                else:
                    return dict(accepted=False, reason="no such skill")
                return dict(accepted=True)
            return dict(accepted=False, reason="unknown method")


ROBOT = Robot()


def ticker():
    nxt = time.time()
    while True:
        ROBOT.step()
        nxt += 1 / HZ
        time.sleep(max(0, nxt - time.time()))


def page(conn, request):
    """Plain HTTP on the same port: the joint list the console serves."""
    if request.path.startswith("/api/meta"):
        r = conn.respond(http.HTTPStatus.OK, json.dumps(META))
        r.headers["Content-Type"] = "application/json"
        return r
    return None if request.path.startswith("/ws") else conn.respond(http.HTTPStatus.NOT_FOUND, "")


def client(ws):
    hz, stop = [0], threading.Event()

    def stream():
        while not stop.is_set():
            if hz[0]:
                try:
                    ws.send(json.dumps(dict(jsonrpc="2.0", method="robot.state", params=ROBOT.state())))
                except Exception:
                    return
            time.sleep(1 / (hz[0] or 10))

    threading.Thread(target=stream, daemon=True).start()
    try:                                             # a client that just goes away ends the loop quietly
        for raw in ws:
            m = json.loads(raw)
            out = ROBOT.call(m.get("method", ""), m.get("params") or {})
            if m.get("method") == "robot.subscribe":
                hz[0] = int((m.get("params") or {}).get("hz", 10))
            if "id" in m:
                ws.send(json.dumps(dict(jsonrpc="2.0", id=m["id"], result=out or {})))
    except ConnectionClosed:
        pass
    finally:
        stop.set()


threading.Thread(target=ticker, daemon=True).start()
with serve(client, "0.0.0.0", a.port, process_request=page) as server:
    print(f"mock runtime on :{a.port}", flush=True)
    server.serve_forever()
