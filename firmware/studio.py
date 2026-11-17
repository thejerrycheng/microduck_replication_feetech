"""Duck Servo Studio: find, rename and test the Microduck's servos from a browser.

    python firmware/studio.py            # opens http://127.0.0.1:8770
    python firmware/studio.py --mock     # no hardware

Needs pyserial. Stdlib HTTP + JSON; every route is in `Studio`. The duck's own computer and its sensors (camera,
ToF, IMU, microphone, speaker) are reached through `duck_agent.py`, which `Brain` copies over SSH and runs there.
"""
import argparse, collections, json, socket, subprocess, threading, time, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duck_agent
import duck_voice
import feetech
from feetech import DUCK

HERE = Path(__file__).resolve().parent
FILES = {"/": HERE / "web/index.html", "/duck.bin": HERE / "web/duck.bin",
         "/three.min.js": HERE.parent / "app/web/vendor/three.min.js",
         "/OrbitControls.js": HERE.parent / "app/web/vendor/OrbitControls.js"}
TYPES = {".html": "text/html", ".js": "text/javascript", ".bin": "application/octet-stream"}
RATE = 500          # ticks/s the Test wiggle ramps at (44 deg/s): slow enough to stop by hand
SPAN = 512          # a servo with no joint ID yet: the slider reaches +-45 deg from where its torque came on
EDGE = 400          # ticks from either end of the encoder inside which a joint cannot be trimmed both ways
TRIM = 256          # while calibrating, a held joint can be nudged +-22 deg from where it was caught
STILL = 3.0         # deg/s: the IMU must be quieter than this to be calibrated
BACK = (0.70711, 0.0, -0.70711, 0.0)         # body orientation of a duck lying flat on its back: nose (+x) up
SLACK = 57          # ticks (5 deg) a joint may sit outside its range and still be driven back in
WATCH, GONE = 0.5, 1.2                       # s between hot-plug checks; s of silence before a servo is dropped
HZ, KEEP = 40, 40 * 120                      # telemetry rate, and how many samples the server keeps (2 min)
SIGNALS = ("pos", "vel", "load", "temp", "volt", "cur", "goal")   # per servo, in this order, in every sample
TUNNEL = 8772       # local end of the SSH tunnel to the duck's agent
SENSORS = ("camera", "tof", "imu", "mic")


class Brain:
    """The agent on the duck's computer: copied there, started, and talked to through one SSH connection. `host` is
    anything `ssh` accepts without a password prompt (an alias in ~/.ssh/config). Closing the connection stops it."""

    def __init__(self, host):
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        subprocess.run(["pkill", "-f", f"{TUNNEL}:127.0.0.1:{duck_agent.PORT}"])       # a tunnel left by an earlier studio
        for name in ("duck_agent.py", "duck_voice.py"):
            subprocess.run([*ssh, host, f"cat > /tmp/{name}"], input=(HERE / name).read_bytes(),
                           check=True, timeout=30, capture_output=True)
        if subprocess.run([*ssh, host, "test -x ~/microduck/tof_vl53l5cx -a -x ~/microduck/tof_vl53l8cx"]).returncode:
            src = subprocess.run(["tar", "czf", "-", "-C", str(HERE), "tof"], capture_output=True, check=True,
                                 env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"}).stdout
            subprocess.run([*ssh, host, "rm -rf /tmp/tof && tar xzf - -C /tmp 2>/dev/null; sh /tmp/tof/build.sh"],
                           input=src, timeout=180, capture_output=True)      # the ToF readers, built once (~15 s)
        self.ssh = subprocess.Popen(
            [*ssh, "-tt", "-o", "ExitOnForwardFailure=yes", "-L", f"{TUNNEL}:127.0.0.1:{duck_agent.PORT}", host,
             "pkill -f '^python3 -u /tmp/duck_agent.py'; python3 -u /tmp/duck_agent.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        late = threading.Timer(20, self.ssh.terminate)     # never wait on a start that hangs
        late.start()
        for line in iter(self.ssh.stdout.readline, b""):
            if b"duck agent on" in line:
                late.cancel()
                return
        late.cancel()
        raise ValueError(f"the agent did not start on {host}")

    def _get(self, name):
        with urllib.request.urlopen(f"http://127.0.0.1:{TUNNEL}/{name}", timeout=4) as r:
            body = r.read()
            return body if r.headers["Content-Type"] == "application/octet-stream" else json.loads(body)

    def __getattr__(self, name):                 # status, camera, tof, mic, imu
        return lambda: self._get(name)

    def _post(self, name, data=b"{}"):
        return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{TUNNEL}/{name}", data=data), timeout=12).read()

    def say(self, tag="chirp"):
        return json.loads(self._post(f"say/{tag}"))

    def save_calibration(self, data):
        self._post("calibration", json.dumps(data).encode())

    def set_mode(self, to):
        out = json.loads(self._post(f"mode/{to}"))
        if "error" in out:
            raise ValueError(out["error"])
        return out["mode"]

    def bus(self, tx, n, timeout):
        url = f"http://127.0.0.1:{TUNNEL}/bus/{n}/{round(timeout * 1000)}"
        return urllib.request.urlopen(urllib.request.Request(url, data=tx), timeout=timeout + 4).read()

    def close(self):
        self.ssh.terminate()


class Wire:
    """Stands in for a serial port: the servo bus on the duck's own computer, one HTTP round trip per packet.
    `feetech.Bus` writes a packet and then reads its reply; the two travel together."""
    timeout = 0.02

    def __init__(self, brain):
        self.brain, self.out = brain, b""

    def reset_input_buffer(self):
        pass

    def write(self, data):
        if self.out:                             # a packet nobody read a reply to (a broadcast)
            self.brain.bus(self.out, 0, 0)
        self.out = bytes(data)

    def read(self, n):
        tx, self.out = self.out, b""
        return self.brain.bus(tx, n, self.timeout)

    def close(self):
        try:
            self.write(b"")                      # send a pending broadcast, if the link is still there
        except Exception:
            pass


class Studio:
    def __init__(self, mock=False):
        self.mock, self.bus, self.port, self.baud = mock, None, None, feetech.BAUDS[0]
        self.ids, self.home, self.goal, self.sent = [], {}, {}, {}
        self.log, self.seq, self.now, self.lost = collections.deque(maxlen=KEEP), 0, {}, False
        self.brain, self.host, self.sensors, self.auto, self.linking = None, "microduck", {}, True, threading.Lock()
        self.cal, self.calibrating, self.volts, self.want = self._local(), False, None, "setup"
        threading.Thread(target=self._ramp, daemon=True).start()
        threading.Thread(target=self._sample, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    # ---- the duck's computer -------------------------------------------------------------------------------------
    def _watch(self):
        """Once a second: which sensors the duck reports; a dead connection drops the brain. With no brain, and
        unless Duck was switched off by hand, look for the duck on the network every few seconds and connect."""
        addr, tick = None, 0
        while True:
            time.sleep(1)
            tick += 1
            try:
                if self.brain:
                    self.sensors = self.brain.status()
                elif self.auto and (self.mock or tick % 4 == 0) and not self.linking.locked():
                    if not self.mock:
                        if addr is None:                 # the alias's real host name, from ~/.ssh/config
                            out = subprocess.run(["ssh", "-G", self.host], capture_output=True, text=True, timeout=5).stdout
                            addr = next(l.split()[1] for l in out.splitlines() if l.startswith("hostname "))
                        socket.create_connection((addr, 22), 1.5).close()        # cheap: is it on the Wi-Fi at all?
                    self.brain_connect({})
            except Exception:
                if self.brain:
                    self.brain_disconnect(auto=True)

    def brain_connect(self, q):
        with self.linking:                       # one at a time: two starts would kill each other's agent
            self.brain_disconnect(auto=True)
            self.host = q.get("host") or self.host
            brain = duck_agent.Mock() if self.mock else Brain(self.host)
            try:
                self.sensors = brain.status()
            except Exception:
                brain.close() if isinstance(brain, Brain) else None
                raise
            self.brain = brain
            self.cal = brain.calibration() or self.cal       # the duck's own file is the one that counts
            self._online()
        return self.state()

    def _online(self):
        """Bring everything up without a click: take the devices from the duck's runtime (Setup) and connect the
        servo bus through the duck. Skipped if Run was chosen by hand, or a USB adapter is already in use."""
        try:
            if self.want == "setup" and not self.mock:
                if self.sensors.get("mode") == "run":
                    self.brain.set_mode("setup")
                    self.sensors = self.brain.status()
                if self.bus is None and self.sensors.get("servos", {}).get("ok"):
                    self.connect(dict(port="duck"))
        except Exception as e:
            print("bring-up:", e, flush=True)

    def brain_disconnect(self, _=None, auto=False):
        """`auto` False = the Duck button: stay off until it is pressed again."""
        brain, self.brain, self.sensors, self.auto = self.brain, None, {}, auto
        if isinstance(brain, Brain):
            brain.close()
        return self.state()

    def mode(self, q):
        """Setup: the duck's runtime is paused and the studio has the servo bus, ToF and camera. Run: the runtime
        has them back. Going to Run lets go of the studio's servo connection first."""
        to = self.want = "run" if q.get("to") == "run" else "setup"
        if to == "run" and self.port == "duck":
            self.disconnect()
        self.brain.set_mode(to)
        self.sensors = self.brain.status()
        return self.state()

    def say(self, q):
        """Play one of the duck's own sounds: a random variant of q["tag"] (chirp = the quack)."""
        tag = q.get("tag", "chirp")
        if tag not in duck_voice.TAGS:
            raise ValueError(f"no sound called {tag}")
        played = self.brain.say(tag)
        if played is None:
            raise ValueError("the duck is still rendering its voice: try again in a minute")
        return dict(played=played)

    # ---- calibration ---------------------------------------------------------------------------------------------
    # One file, kept on the duck (~/microduck/calibration.json) with a copy beside this script. It holds, per joint,
    # the servo ticks at the joint's zero pose, and the IMU's mounting rotation. Only `cal_save` and `cal_level`
    # write it: nothing else ever changes a calibration.
    def _local(self):
        """The copy beside this script: what a duck reached only through a USB adapter is calibrated by. A mock
        studio neither reads nor writes it, so simulated numbers can never reach a real duck."""
        try:
            return {} if self.mock else json.loads((HERE / "calibration.json").read_text())
        except (OSError, ValueError):
            return {}

    def _zero(self, sid):
        return next((j["zero_raw"] for j in self.cal.get("joints", []) if j["id"] == sid), None)

    def _write(self, **parts):
        cal = dict(self.cal, updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **parts)
        if self.brain:
            self.brain.save_calibration(cal)
        if not self.mock:
            (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
        self.cal = cal

    def _mount(self, pose):
        """The IMU's mounting rotation, from one still reading taken while the duck's body orientation is `pose`
        ([w, x, y, z]): mount = raw^-1 x pose, so that raw x mount gives `pose` again. None with no IMU."""
        if not self.brain:
            return None
        imu = self.brain.imu()
        if not imu:
            return None
        if max(map(abs, imu["gyro"])) > STILL:
            raise ValueError("hold the duck still")
        w, x, y, z = imu["raw"]
        return dict(mount=[round(v, 6) for v in duck_agent.qmul((w, -x, -y, -z), pose)], calibrated=True)

    def _hold_all(self, on):
        """Every servo at once: three packets on the bus instead of three per servo."""
        for d in (self.goal, self.sent, self.home):
            d.clear()
        held = self.bus.hold_all(self.ids, on)
        self.home.update(held)
        self.sent.update(held)

    def cal_start(self, _=None):
        """Begin: every servo limp, so the duck can be laid on its back and posed at home by hand."""
        self.calibrating = True
        self._hold_all(False)
        return self._fresh()

    def cal_hold(self, q):
        """Torque on (hold where each joint is now) or off, for all of them."""
        self._hold_all(bool(q["on"]))
        return self._fresh()

    def cal_save(self, _=None):
        """The duck is at home (lying flat on its back, every joint at the model's zero pose): record it. Each
        joint's present ticks become its zero, and the IMU's mount is taken from that same pose."""
        now = self.bus.sample(self.ids)
        missing = [i for i in DUCK if i not in now]
        if missing:
            raise ValueError(f"not answering: {missing}")
        joints = [dict(name=DUCK[i], id=i, zero_raw=now[i]["pos"], direction=feetech.sign(i), calibrated=True,
                       min_rad=feetech.RANGE[DUCK[i]][0], max_rad=feetech.RANGE[DUCK[i]][1]) for i in DUCK]
        imu = self._mount(BACK)
        self._write(joints=joints, pose="home: duck flat on its back, every joint at the model's zero pose",
                    **(dict(imu=imu) if imu else {}))
        return self.cal_cancel()

    def cal_cancel(self, _=None):
        self.calibrating = False
        self._hold_all(False)
        return self._fresh()

    def cal_level(self, _=None):
        """IMU only: the duck is standing upright on a level surface. Its present reading becomes 'level'."""
        imu = self._mount((1.0, 0.0, 0.0, 0.0))
        if imu is None:
            raise ValueError("no IMU reading")
        self._write(imu=imu)
        return self.state()

    def sense(self, name):
        """One reading from the duck: camera -> NV12 bytes, the rest -> JSON-able. None if there is none."""
        return getattr(self.brain, name)() if self.brain and name in SENSORS else None

    # ---- telemetry: every found servo, HZ times a second, into a ring the page reads from ------------------------
    def _sample(self):
        seen, look = {}, 0.0
        while True:
            t0 = time.time()
            bus, ids = self.bus, self.ids
            try:
                if bus and ids:
                    self.now = bus.sample(ids)
                    self.seq += 1
                    self.log.append((self.seq, round(t0, 3), {i: [m[k] for k in SIGNALS] for i, m in self.now.items()}))
                if bus and t0 - look > WATCH:                 # hot-plug: forget the silent, look for the new
                    look = t0
                    seen.update(dict.fromkeys(self.now, t0))
                    gone = [i for i in ids if t0 - seen.setdefault(i, t0) > GONE]
                    new = bus.scan([i for i in feetech.LIKELY if i not in ids or i in gone])
                    if (gone or new) and bus is self.bus:
                        for i in gone:
                            for d in (self.home, self.goal, self.sent, seen):
                                d.pop(i, None)
                        self.ids = sorted({*ids, *new} - (set(gone) - set(new)))
                elif not bus and self.port and not self.mock and t0 - look > WATCH:
                    look = t0                                  # the adapter came back: pick up where we were
                    if self.lost and self.port in feetech.ports():
                        self.connect(dict(port=self.port, baud=self.baud))
            except Exception:                                  # the adapter was pulled
                self._lose()
            time.sleep(max(0.002, 1 / HZ - (time.time() - t0)))

    def _lose(self):
        try:
            self.bus and self.bus.close()
        except Exception:
            pass
        self.bus, self.ids, self.goal, self.sent, self.home, self.now, self.lost = None, [], {}, {}, {}, {}, True

    # ---- motion: goals are approached at RATE, never jumped to --------------------------------------------------
    def _ramp(self):
        dt = 0.02
        while True:
            time.sleep(dt)
            for sid, goal in list(self.goal.items()):
                cur = self.sent.get(sid)
                if self.bus is None or cur is None or cur == goal:
                    continue
                self.sent[sid] = cur + max(-RATE * dt, min(RATE * dt, goal - cur))
                try:
                    self.bus.goto(sid, self.sent[sid])
                except Exception:
                    pass

    def _hold(self, sid, on):
        """Torque on = hold the present position; that position becomes the slider's centre."""
        self.goal.pop(sid, None)
        if on:
            fb = self.bus.feedback(sid)
            if fb is None:
                raise ValueError(f"{sid} is not answering")
            self.home[sid] = self.sent[sid] = fb["pos"]
            self.bus.goto(sid, fb["pos"])
        self.bus.torque(sid, on)

    def _span(self, sid, pos):
        """(lo, hi) ticks the slider and Test may send: the model's joint range, or +-SPAN for a non-joint ID."""
        home = self.home.get(sid, pos)
        if self.calibrating:                     # not calibrated yet: small trims around where it was caught
            return (home - TRIM, home + TRIM)
        return feetech.limits(sid, self._zero(sid) or feetech.ZERO) or (home - SPAN, home + SPAN)

    def _reach(self, sid, pos):
        """`pos` clamped to the servo's span. Refuses a servo that is not inside its joint range to begin with:
        it has not been zeroed, and clamping would swing it across to a limit."""
        lo, hi = self._span(sid, self.sent[sid])
        if not lo - SLACK <= self.sent[sid] <= hi + SLACK:
            self._hold(sid, False)
            raise ValueError(f"{sid} reads {round(self.sent[sid])}, outside its joint range {lo}-{hi}: pose it at zero, then Zero")
        return max(lo, min(hi, int(pos)))

    # ---- routes -------------------------------------------------------------------------------------------------
    def state(self, q=None):
        """Everything the page shows. `since` = last sample number it has; newer ones ride along."""
        since = int((q or {}).get("since", self.seq))
        motors = {i: dict(m, span=self._span(i, m["pos"]),
                          deg=None if self._zero(i) is None else round(feetech.angle(i, m["pos"], self._zero(i)), 1))
                  for i, m in self.now.items() if i in self.ids}
        seen = sorted(m["volt"] for m in motors.values()) or [v for v in [self.sensors.get("volts")] if v]
        if seen:                                 # median servo voltage, smoothed: one reading is only good to 0.1 V
            self.volts = seen[len(seen) // 2] if self.volts is None else self.volts + 0.05 * (seen[len(seen) // 2] - self.volts)
        elif not self.brain and not self.bus:
            self.volts = None
        return dict(battery=None if self.volts is None else dict(volts=round(self.volts, 2), percent=feetech.battery(self.volts)),
                    connected=self.bus is not None, port=self.port, baud=self.baud, mock=self.mock,
                    ports=["mock"] if self.mock else feetech.ports() + ["duck"] * bool(self.sensors.get("servos", {}).get("ok")),
                    bauds=feetech.BAUDS,
                    ids=self.ids, motors=motors, duck=DUCK, signals=SIGNALS, seq=self.seq,
                    cal=dict(on=self.calibrating, saved=self.cal.get("updated") if self.cal.get("joints") else None,
                             imu=bool(self.cal.get("imu")),
                             edge=[i for i, m in motors.items() if self.calibrating and not EDGE < m["pos"] < 4095 - EDGE]),
                    brain=dict(on=self.brain is not None, host=self.host, sensors=self.sensors,
                               size=(duck_agent.W, duck_agent.H)),
                    samples=[r for r in list(self.log) if r[0] > since][-600:])

    def _fresh(self):
        """State after a command, re-read first so the reply already shows its effect."""
        self.now = self.bus.sample(self.ids)
        return self.state()

    def connect(self, q):
        self.disconnect()
        self.port, self.baud = q.get("port"), int(q.get("baud") or feetech.BAUDS[0])
        if self.port == "duck":                  # through the duck's computer, not a USB adapter on this Mac
            if self.sensors.get("mode") == "run":            # the duck's runtime has the bus: take it (Setup mode)
                self.brain.set_mode("setup")
            self.bus = feetech.Bus(None, self.baud, Wire(self.brain))
        else:
            self.bus = feetech.MockBus() if self.mock else feetech.Bus(self.port, self.baud)
        return self.scan({})

    def disconnect(self, _=None):
        if self.bus:
            try:
                self.bus.hold_all(self.ids, False)
            except Exception:
                pass
            self.bus.close()
        self.bus, self.ids, self.goal, self.sent, self.home, self.now, self.lost = None, [], {}, {}, {}, {}, False
        self.calibrating = False
        return self.state()

    def scan(self, _):
        self.ids = self.bus.scan()
        return self._fresh()

    def torque(self, q):
        self._hold(int(q["id"]), q["on"])
        return self._fresh()

    def move(self, q):
        """Slider: a position in ticks, clamped to the joint's range (`_span`). Sent straight to the servo, so what
        you see is the servo's own position loop (its P and D), not a ramp of ours."""
        sid = int(q["id"])
        if sid not in self.sent:
            self._hold(sid, True)
        self.goal[sid] = self.sent[sid] = self._reach(sid, q["pos"])
        self.bus.goto(sid, self.sent[sid])
        return {}

    def wiggle(self, q):
        """Test: +-10 deg around the present position (never past the joint's range), then release."""
        sid = int(q["id"])
        self._hold(sid, True)
        stops = [self._reach(sid, self.home[sid] + off) for off in (114, -114, 0)]

        def run():
            for stop in stops:
                self.goal[sid] = stop
                while self.goal.get(sid) is not None and self.sent.get(sid) != self.goal[sid]:
                    time.sleep(0.02)
                time.sleep(0.15)
            if self.bus and sid in self.goal:
                self._hold(sid, False)
        threading.Thread(target=run, daemon=True).start()
        return {}

    def center(self, q):
        """Zero the servo where it stands: the present pose becomes 2048."""
        sid = int(q["id"])
        self._hold(sid, False)
        self.sent.pop(sid, None)
        self.home.pop(sid, None)
        self.bus.center(sid)
        return self._fresh()

    def gains(self, q):
        p, d = self.bus.gains(int(q["id"]))
        return dict(p=p, d=d, duck=feetech.replica_gains(int(q["id"])))

    def set_gains(self, q):
        sid = int(q["id"])
        self._hold(sid, False)                       # never rewrite the loop gains under torque
        self.bus.set_gains(sid, int(q["p"]), int(q["d"]))
        return self.gains(q)

    def set_id(self, q):
        old, new = int(q["id"]), int(q["to"])
        self._hold(old, False)
        self.bus.set_id(old, new)
        self.ids = sorted({new if i == old else i for i in self.ids})
        return self._fresh()

    def set_baud(self, q):
        """Move every found servo to 1 Mbit (the robot's bus speed), then reopen there."""
        for sid in self.ids:
            self.bus.set_baud(sid, feetech.BAUDS[0])
        return self.connect(dict(port=self.port))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/api/state":
            return self.reply(200, json.dumps(STUDIO.state(dict(p.split("=") for p in query.split("&") if "=" in p))).encode())
        if path.startswith("/sense/"):
            try:
                out = STUDIO.sense(path[7:])
            except Exception:
                out = None
            return self.reply(200, *((out, "application/octet-stream") if isinstance(out, bytes) else (json.dumps(out).encode(),)))
        f = FILES.get(path)
        if f is None or not f.exists():
            return self.reply(404, b"{}")
        self.reply(200, f.read_bytes(), TYPES[f.suffix])

    def do_POST(self):
        name = self.path.rsplit("/", 1)[-1]
        fn = None if name.startswith("_") or name in ("state", "sense") else getattr(STUDIO, name, None)
        if fn is None or self.headers.get("Origin", "").split("//")[-1] not in ("", self.headers.get("Host")):
            return self.reply(404, b"{}")       # unknown route, or a POST from another site
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if name in ("say", "mode") and STUDIO.brain is None:
                raise ValueError("duck not connected")
            if name not in ("connect", "disconnect", "say", "cal_level", "mode") and not name.startswith("brain_") and STUDIO.bus is None:
                raise ValueError("not connected")
            self.reply(200, json.dumps(fn(q)).encode())
        except Exception as e:
            self.reply(400, json.dumps(dict(error=str(e))).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mock", action="store_true", help="simulated servos, no hardware")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--lan", action="store_true", help="also serve the phone app (ios/): every interface + Bonjour; no login")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    addr = ("0.0.0.0" if a.lan else "127.0.0.1", a.port)
    try:                                             # the port first: nothing touches the duck before we own it
        srv = ThreadingHTTPServer(addr, Handler)
    except OSError:                                  # a studio is already there (left in the background): take over
        import os, signal
        old = subprocess.run(["lsof", "-ti", f"tcp:{a.port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
        ours = [int(p) for p in old if "studio.py" in subprocess.run(["ps", "-o", "command=", "-p", p],
                                                                     capture_output=True, text=True).stdout]
        if not ours:
            raise SystemExit(f"port {a.port} is in use by another program")
        print("Taking over from the studio that was already running.")
        for pid in ours:
            os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            time.sleep(0.2)
            try:
                srv = ThreadingHTTPServer(addr, Handler)
                break
            except OSError:
                pass
        else:
            raise SystemExit("the old studio did not stop")
    srv.daemon_threads = True
    STUDIO = Studio(a.mock)

    def bye(*_):                                     # stopped from outside: release the servos and the duck first
        STUDIO.disconnect()
        STUDIO.brain_disconnect()
        raise SystemExit(0)
    import signal
    signal.signal(signal.SIGTERM, bye)
    if a.lan:
        import sys
        sys.path.insert(0, str(HERE.parent / "ios"))
        import lan
        print(lan.announce("studio", a.port, mock=int(a.mock)), f"http://{lan.address()}:{a.port}")   # the app prefers a real studio to a mock
    print(url)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        STUDIO.disconnect()
        STUDIO.brain_disconnect()
