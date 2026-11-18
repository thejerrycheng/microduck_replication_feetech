#!/usr/bin/env python3
"""The mock studio for the simulator tests, with a stand-in for what the real duck's bridge adds: its animations and
its pilot. No servo, no duck: nothing here can reach hardware.

`firmware/studio.py --mock` has simulated servos and sensors but no animations and no policies (those live in the
duck's bridge, duck_agent.Player and duck_agent.Drive, which need the real servo port). This runs that same studio,
unchanged, and gives its mock duck:

  animations  the real clip list, built from app/animations by firmware/anim_pack.py. `play` only keeps the clock
              (0.5 s ease-in, the clip, 0.4 s ease-out, as the duck's player); nothing moves and no sound is made.
  pilot       the duck's real pilot (firmware/duck_pilot.py) with the real policy pack (firmware/policies.npz), ticked
              50 times a second as on the duck, but fed a still, upright IMU and joints at the default pose. So the
              states, speed levels, command slew and timings are the duck's own code; balance and falls are not
              modelled. The command it applies is integrated into a planar pose (`odom`, x forward, y left, heading;
              not on the real duck) so a test can see which way a stick sends it.
  refusals    the bridge's own, in its words: Run mode, a calibration under way, joints or IMU not calibrated, an
              animation while the pilot drives.
  IMU poses   after step 1 of the two-pose IMU calibration the mock duck is "laid on its back, beak up" (its raw
              reading turns 90 deg about the pitch axis) and after step 2 it is upright again, standing in for the
              hands that would do it.

Every pilot message is appended to --log as a JSON line `{t, what, ...}`, with a `state` line at 50 Hz while the
pilot runs `{t, stick, cmd, state, level, odom}`: ios/tools/plot_pilot.py draws them.

    ios/tools/mock_studio.py [--port 8780] [--log logs/ios/pilot_mock.jsonl]
"""
import argparse
import hashlib
import json
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "firmware"), str(ROOT / "ios")]
import anim_pack                                             # noqa: E402
import duck_agent                                            # noqa: E402
import duck_pilot                                            # noqa: E402
import policy_net                                            # noqa: E402
import studio                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8780)
ap.add_argument("--log", default="")
a = ap.parse_args()


class StandIn(duck_agent.Mock, studio.Link):
    """The mock sensors, plus the animation and pilot calls of the duck's bridge (same names, replies and errors)."""
    saved, cal_on = None, False                  # real attributes: Link answers any unknown name with a web request

    def __init__(self):
        super().__init__()
        data, clips, _ = anim_pack.pack()
        self.pack, self.clips = hashlib.sha256(data).hexdigest(), [{k: v for k, v in c.items() if k != "k"} for c in clips]
        self.nets, self.lock = policy_net.load(str(ROOT / "firmware/policies.npz")), threading.Lock()
        self.playing, self.info, self.pilot_, self.halt = None, None, None, threading.Event()
        self.stick, self.stick_t, self.keys, self.odom, self.back = (0.0, 0.0, 0.0, False), 0.0, [], [0.0, 0.0, 0.0], False
        self.out = open(a.log, "a") if a.log else None

    def note(self, **row):
        if self.out:
            self.out.write(json.dumps(dict(t=round(time.time(), 4), **row)) + "\n")
            self.out.flush()

    def set_cal(self, on):                       # as the bridge: a calibration ends the clip
        self.cal_on = bool(on)
        if on:
            self.playing = None

    # ---- what the bridge reports ----
    def status(self):
        now = self.playing
        if now and time.time() - now["t0"] > now["T"] + 0.9:
            now = self.playing = None
        anim = now and dict(name=now["name"], t=round(min(now["T"], max(0.0, time.time() - now["t0"] - 0.5)), 2), T=now["T"])
        i, p = self.info, self.pilot_
        pilot = None if i is None else dict(i, **(dict(p.status(), odom=[round(v, 3) for v in self.odom]) if p else {}))
        return dict(super().status(), anim=anim, pilot=pilot)

    def imu(self):
        """Still, for a calibration: upright with the mock's tilted mount, or that turned onto its back, beak up."""
        r, p = math.radians(6) / 2, math.radians(-4) / 2
        raw = duck_agent.qmul((math.cos(p), 0, math.sin(p), 0), (math.cos(r), math.sin(r), 0, 0))
        if self.back:
            raw = duck_agent.qmul((math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0), raw)
        return duck_agent.pose(raw, ((self.calibration() or {}).get("imu") or {}).get("mount"), [0.0, 0.0, 0.0])

    # ---- animations ----
    def anims(self):
        return dict(pack=self.pack, clips=self.clips)

    def anim(self, what, data=b"{}"):
        if what == "stop":
            self.playing = None
            return dict(now=None)
        name = what[5:]
        clip = next((c for c in self.clips if c["name"] == name), None)
        if clip is None:
            raise ValueError(f"no animation called {name}")
        self._free()
        if not self._cal():
            raise ValueError("calibrate first: the head's zero is not known")
        if self.info and self.info["live"] and not self.info.get("ended"):
            raise ValueError("the policies are driving the duck: stop them first")
        self.playing = dict(name=name, T=clip["T"], t0=time.time())
        self.note(what="anim", name=name, T=clip["T"])
        return dict(now=dict(name=name, t=0.0, T=clip["T"]))

    # ---- the pilot ----
    def _free(self):
        if self.status()["mode"] != "setup":
            raise ValueError("the duck is in Run: its runtime has the servos")
        if getattr(self, "cal_on", False):
            raise ValueError("a calibration is under way")

    def _cal(self, imu=False):
        cal = self.calibration() or {}
        ids = {j["id"] for j in cal.get("joints", [])}
        return all(i in ids for i in duck_agent.Drive.IDS) and (not imu or bool((cal.get("imu") or {}).get("mount")))

    def pilot(self, what, data=b"{}"):
        if what.startswith("start/"):
            live = what == "start/live"
            self._free()
            if not self._cal(imu=True):
                raise ValueError("calibrate first: joints and IMU")
            self._stop()
            self.playing = None
            self.halt = threading.Event()
            self.info, self.odom = dict(live=live, ticks=0, late=0, missed=0, ms=0.0, hz=0.0), [0.0, 0.0, 0.0]
            self.note(what="start", live=live)
            threading.Thread(target=self._run, args=(self.info, self.halt), daemon=True).start()
            return dict(pilot=self.status()["pilot"])
        if what == "stop":
            self._stop()
            self.note(what="stop")
            return dict(pilot=None)
        if what == "drive":
            q = json.loads(data)
            self.stick, self.stick_t = (float(q.get("x", 0)), float(q.get("y", 0)), float(q.get("w", 0)), bool(q.get("run"))), time.monotonic()
            self.note(what="drive", stick=[*self.stick[:3], int(self.stick[3])])
            return dict(pilot=None)
        if what == "pack":
            return dict(pack="mock")
        key = what[6:]
        if key not in ("sit", "jump", "flip", "faster", "slower"):
            raise ValueError(f"no such key: {key}")
        self.keys.append(key)
        self.note(what="press", key=key)
        return dict(pilot=None)

    def _stop(self):
        self.halt.set()
        with self.lock:
            self.info = self.pilot_ = None

    def _run(self, info, halt):
        import numpy as np
        with self.lock:
            pilot = self.pilot_ = duck_pilot.Pilot(self.nets)
            t0 = nxt = time.monotonic()
            while not halt.is_set():
                t = time.monotonic()
                x, y, w, run = self.stick if t - self.stick_t < 0.5 else (0.0, 0.0, 0.0, False)
                pilot.drive(x, y, w, run)
                while self.keys:
                    pilot.press(self.keys.pop(0))
                pilot.tick(np.zeros(3), (1.0, 0.0, 0.0, 0.0), duck_pilot.DEFAULT_POSE, np.zeros(14))
                vx, vy, vw = pilot.v
                px, py, yaw = self.odom
                dt = duck_pilot.DT
                self.odom = [px + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt, py + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt, yaw + vw * dt]
                n = info["ticks"] = info["ticks"] + 1
                info["ms"] = round(info["ms"] + ((time.monotonic() - t) * 1e3 - info["ms"]) / min(n, 100), 2)
                info["hz"] = round(n / max(1e-3, time.monotonic() - t0), 1)
                self.note(what="state", stick=[x, y, w, int(run)], cmd=[round(v, 3) for v in pilot.v], state=pilot.state,
                          level=pilot.level, odom=[round(v, 4) for v in self.odom])
                nxt += duck_pilot.DT
                wait = nxt - time.monotonic()
                if wait < 0:
                    info["late"] += 1
                    nxt = time.monotonic()
                else:
                    time.sleep(wait)
            info["ended"] = True


class Studio(studio.Studio):
    def imu_cal(self, q):
        out = super().imu_cal(q)                             # raises if refused: the duck then stays as it lies
        self.brain.back = int(q.get("step", 1)) == 1
        return out


duck_agent.Mock = StandIn
srv = ThreadingHTTPServer(("0.0.0.0", a.port), studio.Handler)
srv.daemon_threads = True
studio.STUDIO = Studio(True, False)
import lan                                                   # noqa: E402
print(lan.announce("studio", a.port, mock=1, duck=0), f"http://{lan.address()}:{a.port}", flush=True)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
