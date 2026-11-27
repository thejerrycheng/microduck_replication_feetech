#!/usr/bin/env python3
"""Duck Live local server: serves the page (app/web) and runs the duck (engine.py) in real time on this Mac.

  ~/.venvs/microduck_live/bin/python app/server.py [--port 8811] [--walk FILE --run FILE --sitstand FILE]
  -> prints "DUCK_LIVE_READY <port> <token>" once listening (the macOS shell waits for this line)

HTTP, 127.0.0.1 only (`--lan`: every interface, announced over Bonjour for the phone app, see ios/lan.py). Static files need no token; /api/* needs ?token= (a random token per launch, so no other
local page can drive the duck).
  GET  /api/health                     {ok, ws_port, token, policies}
  GET  /api/scene                      bodies + visual geoms (engine.Duck.scene)
  GET  /api/mesh/<i>                   binary mesh i
  POST /api/policies {role: file}      hot-swap a policy (the duck is reset)
  POST /api/reset {seated}
  POST /api/config {lag, vin, sag_gain, auto_reset, walk_self_recovery, level, lively, react}   lag/vin/sag rebuild the plant (reset)
  GET  /animations/sounds/<f>.wav      the emotion sounds (static, like the page)
WebSocket ws://127.0.0.1:<ws_port>/?token=...
  -> {"t":"keys","held":[...],"meta":bool,"alt":bool}   the movement keys currently held (state, not events)
  -> {"t":"stick","x":r,"y":f,"w":cw,"run":bool}         proportional command, each -1..1 (the phone's two sticks; engine.Duck.set_stick)
  -> {"t":"press","key":"s"|"space"|"r"|"a"|"d"|"enter"}
  -> {"t":"perturb","body":i,"local":[x,y,z],"target":[x,y,z]}   ⌘-drag spring (body -1 ends it)
  -> {"t":"emote","name":"happy"}                       play an emotion clip (app/animations/manifest.json)
  -> {"t":"cam","pos":[x,y,z]}                          where the viewer is (the duck looks at you now and then)
  <- snapshots at 50 Hz (engine.Duck.snapshot)

The physics thread paces the 50 Hz control loop against the wall clock; the page only draws what it is sent.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WEB = os.path.join(HERE, "web")
ANIM = os.path.join(HERE, "animations")
LOGD = os.path.join(os.path.dirname(HERE), "logs", "app")
os.makedirs(LOGD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=int(os.environ.get("DUCK_APP_PORT", "8811")))
for _r in ("walk", "run", "sprint", "sitstand", "jump", "roulade", "recovery", "kick"):
    ap.add_argument("--" + _r, help=f"ONNX export for the {_r} role (default: the contract's pick, else the newest export)")
ap.add_argument("--lag", type=int, default=4); ap.add_argument("--vin", type=float, default=7.7); ap.add_argument("--sag", type=float, default=0.1)
ap.add_argument("--auto-reset", action="store_true", help="reset 2 s after a fall when no policy can get the duck up (default: stay down)")
ap.add_argument("--no-walk-self-recovery", action="store_true", help="hand every fall to the stand-up policy, even under VelStand")
ap.add_argument("--still", action="store_true", help="no idle life (glances, breathing, the head leading turns): the bare policies")
ap.add_argument("--lan", action="store_true", help="also serve the phone app (ios/): every interface + Bonjour (the token rides in it)")
ap.add_argument("--no-react", action="store_true", help="no automatic reactions (shake-off after a get-up, yawns, the greeting ...)")
args = ap.parse_args()

import engine  # noqa: E402

TOKEN = secrets.token_urlsafe(18)


def log(*a):
    with open(os.path.join(LOGD, "server.log"), "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


# ---- the duck, in its own real-time thread ------------------------------------------------------------------------------
class Sim:
    def __init__(self):
        self.lock = threading.Lock()
        self.chosen = dict((r, getattr(args, r)) for r in engine.ROLES if getattr(args, r))
        self.cfg = dict(lag=args.lag, vin=args.vin, sag_gain=args.sag, auto_reset=args.auto_reset, walk_self_recovery=not args.no_walk_self_recovery,
                        lively=not args.still, react=not args.no_react, anim_seed=int(time.time()) & 0xFFFF)
        self.greeted = False
        self.duck = engine.Duck(self.chosen, **self.cfg)
        self.snap = json.dumps(self.duck.snapshot())
        self.rate = 0.0
        self.overruns = 0
        self.pending = []           # (fn, args) applied inside the loop
        self.stop = threading.Event()
        self.clients = 0
        threading.Thread(target=self.loop, daemon=True).start()
        # the page asks for all 38 meshes at load; the crease-smoothed normals take ~3.5 s to build, so start now
        threading.Thread(target=lambda: [self.duck.mesh(i) for i in range(self.duck.model.nmesh)], daemon=True).start()

    def post(self, fn, *a):
        with self.lock:
            self.pending.append((fn, a))

    def loop(self):
        dt = self.duck.control_dt
        nxt = time.perf_counter()
        n, t_rate = 0, time.perf_counter()
        while not self.stop.is_set():
            with self.lock:
                todo, self.pending = self.pending, []
            for fn, a in todo:
                try:
                    fn(*a)
                except Exception as e:  # noqa: BLE001
                    log("pending failed:", repr(e))
            self.duck.step()
            s = self.duck.snapshot()
            s["rate"] = round(self.rate, 1)
            s["overruns"] = self.overruns
            self.snap = json.dumps(s)
            n += 1
            now = time.perf_counter()
            if now - t_rate >= 1.0:
                self.rate = n / (now - t_rate)
                n, t_rate = 0, now
            nxt += dt
            lag = now - nxt
            if lag > 0.25:          # far behind: drop the backlog instead of fast-forwarding
                self.overruns += 1
                nxt = now
            elif lag < 0:
                time.sleep(-lag)

    def rebuild(self, **cfg):
        self.cfg.update(cfg)
        viewer = self.duck.anim.viewer_pos
        d = engine.Duck(self.chosen, **self.cfg)
        d.anim.viewer_pos = viewer
        self.duck = d

    def set_anim(self, key, value):
        self.cfg[key] = bool(value)
        setattr(self.duck.anim, key, bool(value))

    def greet(self):
        """The first page to connect is greeted (once per launch)."""
        if self.greeted or not self.cfg.get("react"):
            return
        self.greeted = True
        threading.Timer(1.6, lambda: self.post(lambda: self.duck.emote("hello"))).start()


SIM = Sim()


# ---- HTTP -------------------------------------------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authed(self, u):
        tok = parse_qs(u.query).get("token", [None])[0] or self.headers.get("X-Live-Token")
        if tok != TOKEN:
            self.send(403, {"error": "bad token"})
            return False
        return True

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/"):
            if not self.authed(u):
                return
            if u.path == "/api/health":
                return self.send(200, dict(ok=True, ws_port=WS_PORT, token=TOKEN, policies=engine.list_exports(),
                                           chosen=dict((r, p.name) for r, p in SIM.duck.policies.items()), cfg=SIM.cfg))
            if u.path == "/api/scene":
                return self.send(200, SIM.duck.scene())
            if u.path.startswith("/api/mesh/"):
                return self.send(200, SIM.duck.mesh(int(u.path.rsplit("/", 1)[1])), "application/octet-stream")
            return self.send(404, {"error": "no such api"})
        p = u.path if u.path != "/" else "/index.html"
        root = ANIM if p.startswith("/animations/") else WEB
        f = os.path.normpath(os.path.join(root, p[len("/animations/"):] if root is ANIM else p.lstrip("/")))
        if not f.startswith(root + os.sep) or not os.path.isfile(f):
            return self.send(404, "not found", "text/plain")
        with open(f, "rb") as fh:
            body = fh.read()
        if p == "/index.html":
            body = body.replace(b"__TOKEN__", TOKEN.encode()).replace(b"__WS_PORT__", str(WS_PORT).encode())
        self.send(200, body, mimetypes.guess_type(f)[0] or "application/octet-stream")

    def do_POST(self):
        u = urlparse(self.path)
        if not self.authed(u):
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/policies":
            chosen = dict((r, body[r]) for r in engine.ROLES if r in body and (body[r] or r not in engine.REQUIRED))   # "" = optional role off
            try:
                SIM.chosen.update(chosen)
                SIM.post(lambda: (SIM.duck.load_policies(SIM.chosen), SIM.duck.reset()))
                return self.send(200, {"ok": True})
            except Exception as e:  # noqa: BLE001
                return self.send(400, {"error": str(e)})
        if u.path == "/api/reset":
            SIM.post(SIM.duck.reset, bool(body.get("seated", False)))
            return self.send(200, {"ok": True})
        if u.path == "/api/config":
            cfg = {}
            if "lag" in body: cfg["lag"] = int(body["lag"])
            if "vin" in body: cfg["vin"] = float(body["vin"])
            if "sag_gain" in body: cfg["sag_gain"] = float(body["sag_gain"])
            if "auto_reset" in body: cfg["auto_reset"] = bool(body["auto_reset"])
            if "walk_self_recovery" in body: cfg["walk_self_recovery"] = bool(body["walk_self_recovery"])
            if "level" in body:
                SIM.post(setattr, SIM.duck, "level", int(body["level"]))
            for k in ("lively", "react"):                        # the animation layer's switches
                if k in body:
                    SIM.post(SIM.set_anim, k, bool(body[k]))
            live = ("auto_reset", "walk_self_recovery")          # flags that apply without rebuilding the plant
            if any(k not in live for k in cfg):
                SIM.post(SIM.rebuild, **cfg)
            else:
                for k, v in cfg.items():
                    SIM.cfg[k] = v
                    SIM.post(setattr, SIM.duck, k, v)
            return self.send(200, {"ok": True})
        self.send(404, {"error": "no such api"})


# ---- WebSocket ---------------------------------------------------------------------------------------------------------
def ws_main():
    from websockets.sync.server import serve

    def handler(conn):
        q = parse_qs(urlparse(conn.request.path).query)
        if q.get("token", [None])[0] != TOKEN:
            conn.close(1008, "bad token")
            return
        SIM.clients += 1
        SIM.greet()
        stop = threading.Event()

        def sender():
            last = None
            while not stop.is_set():
                s = SIM.snap
                if s is not last:
                    try:
                        conn.send(s)
                    except Exception:  # noqa: BLE001
                        break
                    last = s
                time.sleep(0.004)
        threading.Thread(target=sender, daemon=True).start()
        try:
            for msg in conn:
                try:
                    m = json.loads(msg)
                except Exception:  # noqa: BLE001
                    continue
                if m.get("t") == "keys":
                    SIM.post(SIM.duck.set_keys, set(m.get("held", [])), bool(m.get("meta")), bool(m.get("alt")))
                elif m.get("t") == "stick":
                    SIM.post(SIM.duck.set_stick, float(m.get("x", 0)), float(m.get("y", 0)), float(m.get("w", 0)), bool(m.get("run")))
                elif m.get("t") == "press":
                    SIM.post(SIM.duck.press, str(m.get("key", "")))
                elif m.get("t") == "perturb":
                    SIM.post(SIM.duck.set_perturb, int(m.get("body", -1)), m.get("local"), m.get("target"))
                elif m.get("t") == "emote":
                    SIM.post(lambda n=str(m.get("name", "")): SIM.duck.emote(n))
                elif m.get("t") == "cam":
                    pos = m.get("pos")
                    if isinstance(pos, list) and len(pos) == 3:
                        SIM.post(lambda p=[float(x) for x in pos]: setattr(SIM.duck.anim, "viewer_pos", p))
        finally:
            stop.set()
            SIM.clients -= 1
            SIM.post(SIM.duck.set_keys, set(), False, False)   # a dropped page must not leave the duck running
            SIM.post(SIM.duck.set_perturb, -1)

    with serve(handler, BIND, WS_PORT, max_size=1 << 16) as srv:
        srv.serve_forever()


def watch_parent():
    """Exit when the process that started us (the app window) is gone, so no duck keeps running unseen."""
    ppid = os.getppid()
    while True:
        time.sleep(1.0)
        if os.getppid() != ppid:
            log("parent gone, exiting")
            os._exit(0)


if os.getppid() != 1:
    threading.Thread(target=watch_parent, daemon=True).start()

BIND = "0.0.0.0" if args.lan else "127.0.0.1"
PORT = free_port(args.port)
WS_PORT = free_port(PORT + 1)
threading.Thread(target=ws_main, daemon=True).start()
httpd = ThreadingHTTPServer((BIND, PORT), H)
httpd.daemon_threads = True
if args.lan:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ios"))
    import lan
    log("lan", lan.announce("live", PORT, ws=WS_PORT, token=TOKEN), lan.address())
log("ready", PORT, WS_PORT, dict((r, p.name) for r, p in SIM.duck.policies.items()))
print(f"DUCK_LIVE_READY {PORT} {TOKEN}", flush=True)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
