"""Duck Servo Studio: find, rename and test the Microduck's servos from a browser.

    python firmware/studio.py            # opens http://127.0.0.1:8770
    python firmware/studio.py --mock     # no hardware

Needs pyserial. Stdlib HTTP + JSON; every route is in `Studio`. The duck's own computer and its sensors (camera,
ToF, IMU, microphone, speaker) are reached through `duck_agent.py`, which `Brain` copies over SSH and runs there.
"""
import argparse, collections, hashlib, json, math, socket, subprocess, sys, threading, time, urllib.request, webbrowser
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


PILOT = ("pilot_start", "pilot_stop", "pilot_drive", "pilot_press")


class Link:
    """Talks to a duck agent listening on 127.0.0.1:`port`. The agent is the one program on the duck that holds the
    camera, microphone, ToF and servo port, so every studio (the Mac's, and the duck's own for the phone) goes
    through the same one instead of each starting its own and fighting over the devices."""
    port = duck_agent.PORT

    def _get(self, name):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/{name}", timeout=4) as r:
            body = r.read()
            return body if r.headers["Content-Type"] == "application/octet-stream" else json.loads(body)

    def __getattr__(self, name):                 # status, camera, tof, mic, imu
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda: self._get(name)

    def _post(self, name, data=b"{}"):
        return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/{name}", data=data), timeout=12).read()

    def say(self, tag="chirp"):
        return json.loads(self._post(f"say/{tag}"))

    def save_calibration(self, data):
        self._post("calibration", json.dumps(data).encode())

    def set_cal(self, on):
        self._post(f"cal/{int(bool(on))}")

    def pilot(self, what, data=b"{}"):
        """`start/dry`, `start/live`, `stop`, `drive` (JSON sticks), `press/<key>`, `pack`, `install` (the pack)."""
        out = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/pilot/{what}", data=data), timeout=60).read())
        if "error" in out:
            raise ValueError(out["error"])
        return out

    def anim(self, what, data=b"{}"):
        """`play/<name>`, `stop`, or `install` with the pack as `data`. Raises with the duck's own reason."""
        out = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/anim/{what}", data=data), timeout=60).read())
        if "error" in out:
            raise ValueError(out["error"])
        return out

    def set_mode(self, to):
        out = json.loads(self._post(f"mode/{to}"))
        if "error" in out:
            raise ValueError(out["error"])
        return out["mode"]

    def bus(self, tx, n, timeout):
        url = f"http://127.0.0.1:{self.port}/bus/{n}/{round(timeout * 1000)}"
        return urllib.request.urlopen(urllib.request.Request(url, data=tx), timeout=timeout + 4).read()

    def close(self):
        pass


def listening(port):
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        return True
    except OSError:
        return False


class Local(Link):
    """`--duck`: this studio runs on the duck. It uses the duck's agent, the service `duck-agent`, which starts at
    boot beside it; it never starts one itself and never opens a device."""

    def __init__(self):
        for _ in range(120):                     # at boot, or while the agent restarts, it is there within seconds
            if listening(self.port):
                return
            time.sleep(0.25)
        raise ValueError("the duck agent is not running: systemctl --user status duck-agent")


class Brain(Link):
    """The agent on the duck's computer, from the Mac, through one SSH tunnel. `host` is anything `ssh` accepts
    without a password prompt (an alias in ~/.ssh/config). The agent is the duck's own service `duck-agent`; this
    only makes sure the duck has this studio's version of it (installing or updating the service if not) and then
    talks to it. Closing the studio leaves the agent running for the duck's own studio and the next connection."""
    port = TUNNEL
    DIR = "~/microduck/studio/firmware"          # where the duck keeps the agent (the same place duck_serve.sh uses)
    FILES = ("duck_agent.py", "duck_voice.py", "duck_pilot.py", "policy_net.py", "run_log.py")

    def __init__(self, host):
        # Two SSH connections in all: one copies both files and checks the ToF readers, one is the tunnel.
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        subprocess.run(["pkill", "-f", f"{TUNNEL}:127.0.0.1:{duck_agent.PORT}"])       # a tunnel left by an earlier studio
        import io, tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:        # what the duck's agent is made of
            for n in self.FILES:
                tar.add(HERE / n, arcname=n)
        have = subprocess.run(                       # the files in one go, and the answer to "are the ToF readers built?"
            [*ssh, host, f"mkdir -p {self.DIR}; rm -rf /tmp/duck_new; mkdir /tmp/duck_new; tar xzf - -C /tmp/duck_new 2>/dev/null; "
                         "test -x ~/microduck/tof_vl53l5cx -a -x ~/microduck/tof_vl53l8cx"],
            input=buf.getvalue(), timeout=30, capture_output=True)
        if have.returncode == 255:
            raise ValueError(f"cannot reach {host} over SSH")
        if have.returncode:
            src = subprocess.run(["tar", "czf", "-", "-C", str(HERE), "tof"], capture_output=True, check=True,
                                 env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"}).stdout
            subprocess.run([*ssh, host, "rm -rf /tmp/tof && tar xzf - -C /tmp 2>/dev/null; sh /tmp/tof/build.sh"],
                           input=src, timeout=180, capture_output=True)      # the ToF readers, built once (~15 s)
        # The service is left alone if it already runs this very code (the phone may be using it); otherwise the new
        # files go in and the service is installed or restarted. Then wait until it answers.
        d = self.DIR
        same = " && ".join(f"cmp -s /tmp/duck_new/{n} {d}/{n}" for n in self.FILES)
        start = (f"{same} && systemctl --user -q is-active duck-agent "
                 f"|| {{ cp /tmp/duck_new/* {d}/; python3 {d}/duck_agent.py --install; }}; "
                 f"for i in $(seq 120); do curl -s -m 2 -o /dev/null http://127.0.0.1:{duck_agent.PORT}/calibration "
                 "&& echo duck agent on && break; sleep 0.25; done; sleep 100000000")
        self.ssh = subprocess.Popen(
            [*ssh, "-o", "ExitOnForwardFailure=yes", "-L", f"{TUNNEL}:127.0.0.1:{duck_agent.PORT}", host, start],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        late = threading.Timer(45, self.ssh.terminate)     # never wait on a start that hangs
        late.start()
        for line in iter(self.ssh.stdout.readline, b""):
            if b"duck agent on" in line:
                late.cancel()
                return
        late.cancel()
        raise ValueError(f"the agent did not start on {host}")

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
    def __init__(self, mock=False, local=False):
        self.local = local                       # `--duck`: this studio runs on the duck itself and serves the phone
        self.mock, self.bus, self.port, self.baud = mock, None, None, feetech.BAUDS[0]
        self.ids, self.home, self.goal, self.sent = [], {}, {}, {}
        self.log, self.seq, self.now, self.lost = collections.deque(maxlen=KEEP), 0, {}, False
        self.brain, self.host, self.sensors, self.auto, self.linking = None, "microduck", {}, True, threading.Lock()
        self.cal, self._cal, self.volts, self.want, self.asked = self._local(), False, None, "setup", 0.0
        threading.Thread(target=self._ramp, daemon=True).start()
        threading.Thread(target=self._sample, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    # ---- the duck's computer -------------------------------------------------------------------------------------
    def _watch(self):
        """Once a second: which sensors the duck reports; a dead connection drops the brain. With no brain, and
        unless Duck was switched off by hand, look for the duck on the network every few seconds and connect."""
        addr, tick, was = None, 0, None
        while True:
            time.sleep(1)
            tick += 1
            try:
                if self.brain:
                    self.sensors = self.brain.status()
                    now = self.sensors.get("mode")
                    if now == "run" and self.bus and self.port == "duck":
                        self.disconnect()        # the runtime was started from elsewhere: it has the servo bus now
                    if was == "run" and now == "setup" and self.bus is None and self.sensors.get("servos", {}).get("ok"):
                        self.connect(dict(port="duck"))      # back in Setup, from here or elsewhere: the servos return
                    was = now
                    if tick % 3 == 0 and not self.calibrating:       # a calibration saved from another studio
                        self.cal = self.brain.calibration() or self.cal
                    if tick % 10 == 0 and isinstance(self.brain, Link) and not getattr(self, "anims", None):
                        self.anims = self.brain.anims()["clips"]      # a pack installed since (by a Mac studio)
                elif self.auto and (self.mock or self.local or tick % 3 == 1) and not self.linking.locked():
                    if not self.mock and not self.local:
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
            brain = duck_agent.Mock() if self.mock else Local() if self.local else Brain(self.host)
            try:
                self.sensors = brain.status()
            except Exception:
                brain.close() if isinstance(brain, Link) else None
                raise
            self.brain = brain
            self.cal = brain.calibration() or self.cal       # the duck's own file is the one that counts
            threading.Thread(target=self._anims, daemon=True).start()
            self._online()
        return self.state()

    def _anims(self):
        """What the duck can play. From a Mac, first make sure the duck holds this repo's clips: the pack is built
        here (anim_pack.py, needs numpy) and sent only if the duck's differs. The duck keeps it for good."""
        try:
            if isinstance(self.brain, Brain):
                import anim_pack
                data = anim_pack.pack()[0]
                if self.brain.anims()["pack"] != hashlib.sha256(data).hexdigest():
                    self.brain.anim("install", data)
            self.anims = self.brain.anims()["clips"] if isinstance(self.brain, Link) else []
            pack = HERE / "policies.npz"                 # the policies too (policy_pack.py writes this file)
            if isinstance(self.brain, Brain) and pack.exists():
                data = pack.read_bytes()
                if self.brain.pilot("pack")["pack"] != hashlib.sha256(data).hexdigest():
                    # installing stops the pilot, and a stopped pilot lets every joint go limp: never under live policies
                    if ((self.brain.status() or {}).get("pilot") or {}).get("live"):
                        print("policies: a newer pack is waiting; not installed while the duck's policies are live", flush=True)
                    else:
                        self.brain.pilot("install", data)
        except Exception as e:
            print("animations:", e, flush=True)

    def anim_play(self, q):
        """Play one of the duck's emotion clips on the real head, beak and speaker. The duck plays it by itself."""
        self.brain.anim("play/" + q["name"])
        self.sensors = self.brain.status()
        return dict(anim=self.sensors.get("anim"))

    # ---- driving: the policies run on the duck (duck_agent.Drive); these only pass the driver's inputs on ----
    def pilot_start(self, q):
        """q["live"]: drive the servos. Without it a dry run: the duck reads and computes, nothing moves."""
        if q.get("live"):
            self.goal.clear(); self.sent.clear()         # the studio's own holds are over: the policy has the joints
        return self.brain.pilot("start/live" if q.get("live") else "start/dry")

    def pilot_stop(self, _=None):
        return self.brain.pilot("stop")

    def pilot_drive(self, q):
        self.brain.pilot("drive", json.dumps({k: q.get(k, 0) for k in ("x", "y", "w", "run")}).encode())
        return {}

    def pilot_press(self, q):
        return self.brain.pilot("press/" + q["key"])

    def anim_stop(self, _=None):
        """Stop the clip and let the head go limp."""
        self.brain.anim("stop")
        self.sensors = self.brain.status()
        return dict(anim=None)

    def _online(self):
        """Bring everything up without a click: connect the servo bus through the duck. Skipped if the duck is in
        Run (here or from elsewhere), or a USB adapter is already in use."""
        try:
            # A duck found in Run was put there by someone (after a boot the bridge itself takes the devices, see
            # duck_agent.Sensors.__init__), perhaps the phone driving it: left alone. Connect or Setup takes it back.
            if self.want == "setup" and not self.mock and not self.local:      # on the duck: never take over unasked
                if self.bus is None and self.sensors.get("servos", {}).get("ok"):
                    self.connect(dict(port="duck"))
        except Exception as e:
            print("bring-up:", e, flush=True)

    def brain_disconnect(self, _=None, auto=False):
        """`auto` False = the Duck button: stay off until it is pressed again."""
        brain, self.brain, self.sensors, self.auto = self.brain, None, {}, auto
        if isinstance(brain, Link):
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
        if to == "setup" and self.bus is None and self.sensors.get("servos", {}).get("ok"):
            self.connect(dict(port="duck"))      # Setup gives the servos back: take them, no second click
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

    def _dir(self, sid):
        """Which way the joint's positive angle runs on its servo: the calibration's word, else the built-in guess."""
        return next((j["direction"] for j in self.cal.get("joints", []) if j["id"] == sid), feetech.sign(sid))

    def flip(self, q):
        """Reverse one joint's direction in the saved calibration. Its zero stays; its angle changes sign."""
        sid = int(q["id"])
        if self._zero(sid) is None:
            raise ValueError("calibrate first: a direction is stored with the calibration")
        self._hold(sid, False)                   # its limits are about to mirror: never under torque
        self._write(joints=[dict(j, direction=-j["direction"]) if j["id"] == sid else j for j in self.cal["joints"]])
        return self._fresh()

    def _write(self, **parts):
        cal = dict(self.cal, updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **parts)
        if self.brain:
            self.brain.save_calibration(cal)
        if not self.mock:
            (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
        self.cal = cal

    @property
    def calibrating(self):
        """A calibration is under way. With a duck connected this is the duck's word (its agent keeps it), so a
        calibration begun in one studio, the phone's say, shows in every other one too."""
        return self.sensors.get("calibrating", self._cal) if self.brain else self._cal

    @calibrating.setter
    def calibrating(self, on):
        self._cal = on
        if self.brain:
            self.brain.set_cal(on)
            self.sensors = dict(self.sensors, calibrating=on)      # at once, not at the next status reading

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
        joints = [dict(name=DUCK[i], id=i, zero_raw=now[i]["pos"], direction=self._dir(i), calibrated=True,
                       min_rad=feetech.RANGE[DUCK[i]][0], max_rad=feetech.RANGE[DUCK[i]][1]) for i in DUCK]
        # The IMU is not touched here: one pose says which way is down but not which way is front (`imu_cal`).
        self._write(joints=joints, pose="home: duck flat on its back, every joint at the model's zero pose")
        return self.cal_cancel()

    def cal_cancel(self, _=None):
        self.calibrating = False
        self._hold_all(False)
        return self._fresh()

    def cal_level(self, _=None):
        """IMU only: the duck is upright with its trunk level. Its present reading becomes 'level'.

        One level reading says which way is down and nothing about which way is front: the IMU's own heading is
        wherever it was when it was switched on. So the front is then squared to the IMU board: of the directions
        in the level plane, the one along a chip axis nearest to where the reading put it (the board is mounted
        square to the trunk). That leaves four possible fronts, a quarter turn apart: `imu_turn` picks among them."""
        imu = self._mount((1.0, 0.0, 0.0, 0.0))
        if imu is None:
            raise ValueError("no IMU reading")
        w, x, y, z = imu["mount"]
        front = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))      # body front and left, in chip axes
        left = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
        best = max(((abs(f * math.cos(a) + l * math.sin(a)), -abs(a), a) for f, l in zip(front, left)
                    for a in (math.atan2(l, f), math.atan2(-l, -f))), default=(0, 0, 0.0))[2]
        imu["mount"] = [round(v, 6) for v in duck_agent.qmul(imu["mount"], (math.cos(best / 2), 0.0, 0.0, math.sin(best / 2)))]
        self._write(imu=imu)
        return self.state()

    def imu_cal(self, q):
        """The IMU's orientation in the trunk, from gravity in two poses. Gravity is the one direction the IMU knows
        for certain (its heading is arbitrary), so each pose pins one trunk axis in the chip's own axes:

          step 1  the duck upright, trunk level        ->  "up" as the chip sees it is the trunk's z (up)
          step 2  the duck on its back, beak straight up  ->  "up" as the chip sees it is the trunk's x (front)

        Left is then z cross x. The second reading is squared against the first, so a few degrees off in step 2
        only turns the front slightly; the poses must be 60 to 120 deg apart or the result is refused. Step 1 alone
        changes nothing. Each reading needs the duck still."""
        imu = self.brain.imu() if self.brain else None
        if not imu:
            raise ValueError("no IMU reading")
        if max(map(abs, imu["gyro"])) > STILL:
            raise ValueError("hold the duck still")
        w, x, y, z = imu["raw"]
        up = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))      # the world's up, in the chip's axes
        if int(q.get("step", 1)) == 1:
            self.imu_up = up
            return self.state()
        if getattr(self, "imu_up", None) is None:
            raise ValueError("step 1 first: the duck upright, trunk level")
        zb, dot = self.imu_up, sum(a * b for a, b in zip(self.imu_up, up))
        if abs(dot) > 0.5:
            raise ValueError(f"the two poses are only {math.degrees(math.acos(max(-1, min(1, dot)))):.0f} deg apart: step 2 is on its back, beak straight up")
        xb = [a - dot * b for a, b in zip(up, zb)]
        n = math.sqrt(sum(v * v for v in xb))
        xb = [v / n for v in xb]
        yb = (zb[1] * xb[2] - zb[2] * xb[1], zb[2] * xb[0] - zb[0] * xb[2], zb[0] * xb[1] - zb[1] * xb[0])
        m = [[xb[i], yb[i], zb[i]] for i in range(3)]                  # columns: the trunk's front, left, up in chip axes
        t = m[0][0] + m[1][1] + m[2][2]
        if t > 0:
            k = math.sqrt(t + 1) * 2
            quat = (k / 4, (m[2][1] - m[1][2]) / k, (m[0][2] - m[2][0]) / k, (m[1][0] - m[0][1]) / k)
        else:
            i = max(range(3), key=lambda a: m[a][a])
            j, l = (i + 1) % 3, (i + 2) % 3
            k = math.sqrt(1 + m[i][i] - m[j][j] - m[l][l]) * 2
            v = [0.0, 0.0, 0.0]
            v[i], v[j], v[l] = k / 4, (m[j][i] + m[i][j]) / k, (m[l][i] + m[i][l]) / k
            quat = ((m[l][j] - m[j][l]) / k, *v)
        self.imu_up = None
        self._write(imu=dict(mount=[round(v, 6) for v in quat], calibrated=True, how="two poses",
                             apart_deg=round(math.degrees(math.acos(max(-1, min(1, dot)))), 1)))
        return self.state()

    def imu_turn(self, q):
        """Turn the IMU's body frame about one of its own axes: q["axis"] "x" (front), "y" (left) or "z" (up), by
        q["deg"] (180 = flip the other two axes; 90 about z = the front a quarter turn to the left). For a front
        that points the wrong way after Level, or an IMU that reads upside down."""
        imu = dict(self.cal.get("imu") or {})
        if not imu.get("mount"):
            raise ValueError("level the IMU first")
        h = math.radians(float(q.get("deg", 180))) / 2
        axis = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))[q["axis"]]
        imu["mount"] = [round(v, 6) for v in duck_agent.qmul(imu["mount"], (math.cos(h), *(math.sin(h) * c for c in axis)))]
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
            # nobody looking (no page or phone has asked for 5 s): once a second is enough, and the duck's small
            # computer runs cooler. The first request finds a reading at most a second old; the rate is back at once.
            time.sleep(max(0.002, (1 / HZ if time.time() - self.asked < 5 else 1.0) - (time.time() - t0)))

    def _lose(self):
        try:
            self.bus and self.bus.close()
        except Exception:
            pass
        self.bus, self.ids, self.goal, self.sent, self.home, self.now, self.lost = None, [], {}, {}, {}, {}, True
        if self.port == "duck":                  # the duck went, not an adapter: `_online` reconnects when it is back.
            self.port = None                     # Kept, the page showed a port that no longer exists and Connect sent ""

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
        return feetech.limits(sid, self._zero(sid) or feetech.ZERO, self._dir(sid)) or (home - SPAN, home + SPAN)

    def _reach(self, sid, pos):
        """`pos` clamped to the servo's span. Refuses a servo that is not inside its joint range to begin with:
        it has not been zeroed, and clamping would swing it across to a limit."""
        lo, hi = self._span(sid, self.sent[sid])
        if not lo - SLACK <= self.sent[sid] <= hi + SLACK:
            self._hold(sid, False)
            raise ValueError(f"{sid} reads {round(self.sent[sid])}, outside its joint range {lo}-{hi}: pose it at zero, then Zero")
        return max(lo, min(hi, int(pos)))

    def _clips(self):
        """The emotion clips this studio would give the duck (anim_pack.py, from app/animations), without their motion
        and sound: so the page lists every clip even with no duck connected (it greys them out until one is). On the
        duck itself without app/animations, an empty list."""
        if not hasattr(self, "_clip_list"):
            try:
                import anim_pack
                self._clip_list = [{k: c[k] for k in ("name", "label", "emoji", "group", "T", "needs")} for c in anim_pack.pack()[1]]
            except Exception:
                self._clip_list = []
        return self._clip_list

    # ---- routes -------------------------------------------------------------------------------------------------
    def state(self, q=None):
        """Everything the page shows. `since` = last sample number it has; newer ones ride along."""
        since = int((q or {}).get("since", self.seq))
        self.asked = time.time()
        motors = {i: dict(m, span=self._span(i, m["pos"]),
                          deg=None if self._zero(i) is None else round(feetech.angle(i, m["pos"], self._zero(i), self._dir(i)), 1),
                          dir=self._dir(i))
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
                             imu=bool(self.cal.get("imu")), imu_step=1 if getattr(self, "imu_up", None) else 0,
                             imu_how=(self.cal.get("imu") or {}).get("how", "level only" if self.cal.get("imu") else None),
                             edge=[i for i, m in motors.items() if self.calibrating and not EDGE < m["pos"] < 4095 - EDGE]),
                    anims=(getattr(self, "anims", None) or self._clips()) if self.brain else self._clips(),
                    brain=dict(on=self.brain is not None, linking=self.linking.locked(), host=self.host, sensors=self.sensors,
                               size=(duck_agent.W, duck_agent.H)),
                    samples=[r for r in list(self.log) if r[0] > since][-600:])

    def _fresh(self):
        """State after a command, re-read first so the reply already shows its effect."""
        self.now = self.bus.sample(self.ids)
        return self.state()

    def connect(self, q):
        self.disconnect()
        port = q.get("port")
        if port == "duck" and not self.brain:
            raise ValueError("the duck is not connected: choose the USB adapter")
        if not port or not (self.mock or port == "duck" or port in feetech.ports()):
            raise ValueError(f"no such port: {port or 'none chosen'}")
        self.port, self.baud = port, int(q.get("baud") or feetech.BAUDS[0])
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
        self._cal = False                        # this studio's own; one going through the duck stays with the duck
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
        """Zero the servo where it stands: the present pose becomes 2048. This renumbers the servo's positions, so a
        saved calibration is renumbered with it: the joint's zero moves by the same amount and still means the same
        physical pose. (Without that, a joint zeroed after Save sat "outside its range" and could not be moved.)
        The exception is a servo that stands outside its joint's range to begin with: the saved zero cannot be right
        for it (a horn refitted, a servo swapped or tested off the duck), so the present pose becomes the joint's
        zero, which is what `_reach` asks for: "pose it at zero, then Zero"."""
        sid = int(q["id"])
        self._hold(sid, False)
        self.sent.pop(sid, None)
        self.home.pop(sid, None)
        before = self.bus.feedback(sid)
        zero = self._zero(sid)
        lo, hi = feetech.limits(sid, zero, self._dir(sid)) if before and zero is not None and sid in DUCK else (0, 0)
        after = self.bus.center(sid)
        if before and zero is not None:
            inside = lo - SLACK <= before["pos"] <= hi + SLACK
            new = zero + after - before["pos"] if inside else after
            self._write(joints=[dict(j, zero_raw=new) if j["id"] == sid else j for j in self.cal["joints"]])
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
            if name in ("say", "mode", "anim_play", "anim_stop") + PILOT and STUDIO.brain is None:
                raise ValueError("duck not connected")
            if name not in ("connect", "disconnect", "say", "cal_level", "imu_cal", "imu_turn", "mode", "anim_play", "anim_stop") + PILOT and not name.startswith("brain_") and STUDIO.bus is None:
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
    ap.add_argument("--duck", action="store_true", help="run on the duck itself (duck_serve.sh): its own sensors and servo port, no SSH")
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
    STUDIO = Studio(a.mock, a.duck)

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
        print(lan.announce("studio", a.port, mock=int(a.mock), duck=int(a.duck)), f"http://{lan.address()}:{a.port}", flush=True)   # the app prefers a real studio to a mock
    print(url)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        STUDIO.disconnect()
        STUDIO.brain_disconnect()
