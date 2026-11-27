"""Duck agent: serves the sensors on the duck's computer (Radxa Zero 3W) to the studio. Python stdlib only.

    python3 duck_agent.py            # on the duck; listens on 127.0.0.1:8771, reached through an SSH tunnel
    python3 duck_agent.py --install  # on the duck: run it as the service `duck-agent`, now and at every boot
    python3 duck_agent.py --mock     # anywhere: simulated sensors

It is the duck's one hardware bridge: the only program that opens the camera, the microphone, the speaker, the ToF
ranger and the servo bus (servos and IMU). Every client (the Mac's studio through an SSH tunnel, the duck's own
studio for the phone) asks it over HTTP and none opens a device itself. It runs as a systemd user service started
at boot and restarted if it dies, and it carries on with any device absent: each one is opened when first asked
for and opened again after it fails, so a camera, a sensor or the servo power that comes back is picked up without
anyone restarting anything. This is the pattern of Pollen's own runtime (one owner per device, systemd-supervised
daemons that retry their hardware: pollen-robotics/microduck `docs/design/architecture.md`), with one bridge
instead of their three because a setup tool needs every device from one place.

The duck's own runtime (robotd, tofd) uses the servo bus and the ToF ranger while it runs, and only one program
can. (Its camera service, mediad, is switched off by `mode_install.sh`: on this board it leaves the camera broken.) So the duck is in one of two modes (`Sensors.mode`): "run", the runtime has them; "setup",
the runtime is paused and this agent does. Switching needs the one sudoers rule `mode_install.sh` adds.

At the first start after a boot the agent puts the duck in "setup" (the runtime's units are enabled in the image
and start by themselves), so that it comes up owning every device; "run" is then a choice made in the studio.

A Mac studio that connects installs or updates this file and the service over SSH (`Studio.brain_connect`);
nothing needs root. It listens on loopback only, so the camera and microphone are never open to the Wi-Fi.

Sensors, each `{ok, note}` in `/status`:
  camera   IMX219 on CSI. Frames come from `v4l2-ctl` as raw NV12, W x H; the page converts them.
  mic      the HAT's TLV320AIC3104 codec (I2C3, 0x18): microphone on MIC3R -> right channel, read with `arecord`.
  speaker  the same codec's line outputs, driven with `aplay`. It plays the duck's own voice: a bank of sounds
           rendered once by `duck_voice.py` (Pollen's synth) into ~/microduck/sounds and kept there.
           The image has no kernel driver for the codec, so `Sensors.codec` sets its registers over I2C itself and
           the audio travels on the plain I2S card ("dummy-sound"). Both streams must be 32-bit: the codec's PLL
           is fed from the bit clock, which is 64 x 48 kHz only then.
  imu      the `imu_to_servo` board (LSM6DSV16X) on the servo bus, ID 200: gyro and the chip's own orientation
           quaternion, read from register 124 (layout from pollen-robotics/microduck `duck-control/src/imu.rs`).
  servos   the servo bus on the duck's UART (/dev/ttyS2, 1 Mbit). `/bus` passes raw Feetech packets through, so the
           studio's own `feetech.Bus` drives it. The user must be in the `dialout` group to open the port.
  tof      VL53L5CX / L8CX 8 x 8 ranger at 0x29 on I2C3 (the HAT socket marked IO2/3). Ranging is done by a small C
           program built from ST's driver (`tof/`, compiled on the duck by `tof/build.sh` into ~/microduck); this
           file starts it and reads its frames.
"""
import argparse, array, fcntl, glob, json, math, os, random, select, struct, subprocess, termios, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

W, H = 320, 240                     # camera frame the studio is sent
PORT = 8771
IDLE = 3.0                          # s without a request before the camera / microphone is released
STALL = 30.0                        # s a reader may give nothing while asked before it is started afresh
UNIT = """[Unit]
Description=Duck hardware bridge: the one owner of the camera, microphone, speaker, ToF and servo bus

[Service]
ExecStart=/usr/bin/python3 -u {path}
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
"""
RATE, CHUNK = 48000, 4800           # microphone: Hz, frames per reading (0.1 s)
CODEC = (3, 0x18)                   # I2C bus and address of the TLV320AIC3104 on the HAT
# Codec registers (page 0, TI SLAS510): clocks from BCLK through the PLL (P 1, J 32, R 1: 3.072 MHz x 32 / 2048 =
# 48 kHz), I2S slave, both ADCs and DACs on, MIC3R -> right ADC at 20 dB with 2.5 V mic bias, DACs -> line outputs
# (the speaker amplifier) at 9 dB. Routing follows pollen-robotics/microduck `deploy/audio/aic3104-init.sh`.
CODEC_SETUP = ((0, 0), (102, 0xA2), (3, 0x91), (4, 32 << 2), (5, 0), (6, 0), (11, 0x01), (101, 0x00), (2, 0x00),
               (7, 0x0A), (8, 0x00), (9, 0x00), (10, 0), (15, 0x28), (16, 0x28), (17, 0x0F), (18, 0xF0), (19, 0x7C),
               (22, 0x7C), (25, 0x80), (37, 0xC0), (41, 0x00), (43, 0x00), (44, 0x00), (82, 0x80), (86, 0x99),
               (92, 0x80), (93, 0x99))
HOME = os.path.expanduser("~/microduck")      # what the duck keeps: its voice bank and its calibration
BANK = HOME + "/sounds"
# /data is the duck's small always-writable partition (safe_install.sh); the rest of the card is read-only then
CALIBRATION = ("/data" if os.path.ismount("/data") else HOME) + "/calibration.json"
ANIM = ("/data" if os.path.ismount("/data") else HOME) + "/animations"     # the emotion clips (anim_pack.py)
POLICIES = ("/data" if os.path.ismount("/data") else HOME) + "/policies.npz"     # the policy pack (policy_pack.py)
RUNS = ("/data" if os.path.ismount("/data") else HOME) + "/runs"     # every policy run, tick by tick (run_log.py)
RUNTIME = ("robotd", "tofd")        # the duck's own services that hold the servo bus and the ToF ranger
UART = "/dev/ttyS2"                 # the servo bus on a Radxa Zero 3W (UART2 on the 40-pin header)


def _i2c(bus, addr, reg=None):
    """True if a device answers at `addr` (or the byte in `reg`); "busy" if a kernel driver holds it; else None."""
    try:
        fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, 0x0703, addr)                       # I2C_SLAVE
        if reg is not None:
            os.write(fd, bytes([reg]))
        return os.read(fd, 1)[0] if reg is not None else bool(os.read(fd, 1))
    except OSError as e:
        return "busy" if e.errno == 16 else None
    finally:
        os.close(fd)


class Pipe:
    """A child process whose stdout is cut into fixed-size records (or lines, with size None); keeps the newest.
    Starts on the first `get`, stops after IDLE seconds with nobody asking."""

    def __init__(self, cmd, size):
        self.cmd, self.size, self.last, self.asked, self.proc, self.got = cmd, size, None, 0.0, None, 0.0

    def get(self):
        self.asked = time.time()
        proc = self.proc
        if proc is None:
            self.got = self.asked
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        elif self.asked - self.got > STALL:      # alive but silent (a device that went away): end it; the next
            proc.kill()                          # call starts a new one
        return self.last

    def _read(self, proc):
        while time.time() - self.asked < IDLE:
            rec = proc.stdout.read(self.size) if self.size else proc.stdout.readline()
            if len(rec) < (self.size or 1):
                break
            self.last, self.got = rec, time.time()
        proc.kill()
        proc.wait()
        self.proc = self.last = None


def qmul(a, b):
    """Hamilton product of two [w, x, y, z] quaternions."""
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def pose(raw, mount, gyro):
    """An IMU reading as the studio wants it. `raw` is the chip's quaternion (sensor -> its own gravity-aligned world).
    `mount` is the fixed rotation body -> sensor found by calibration: the inverse of `raw` when the duck stood level
    at home, so that the body reads identity there. Body orientation = raw x mount; without a mount it is `raw`.
    Roll, pitch, yaw are ZYX Euler angles in degrees. Yaw has no absolute reference: it drifts and restarts."""
    w, x, y, z = q = qmul(raw, mount) if mount else raw
    # the gyro measures in the sensor's axes; `rate` is the same turn rate in the body's axes (about front, left, up)
    # (mount takes body axes to sensor axes, so a sensor-frame vector comes back with its inverse: mount^-1 g mount.
    # Checked on the duck, 2026-11-18: against how roll and pitch really changed over 160 samples of live runs this
    # order correlates 0.60 and 0.54; the opposite order, used until then, 0.03 and 0.13.)
    rate = qmul(qmul((mount[0], -mount[1], -mount[2], -mount[3]), (0.0, *gyro)), mount)[1:] if mount else gyro
    return dict(quat=[round(v, 5) for v in q], raw=[round(v, 5) for v in raw], gyro=gyro, calibrated=bool(mount),
                rate=[round(v, 2) for v in rate],
                roll=round(math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))), 2),
                pitch=round(math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x))))), 2),
                yaw=round(math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), 2))


class Uart:
    """The servo bus: a raw serial port at 1 Mbit, one packet out and its reply back per call."""

    def __init__(self):
        self.fd = os.open(UART, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = getattr(termios, "B1000000", termios.B115200)
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        self.lock = threading.Lock()

    def _read(self, n, end):
        buf = b""
        while len(buf) < n and select.select([self.fd], [], [], max(0, end - time.time()))[0]:
            buf += os.read(self.fd, n - len(buf))
        return buf

    def xfer(self, tx, n, timeout):
        """Send `tx`, return up to `n` reply bytes. A half-duplex adapter that echoes what was sent is handled:
        the echo is recognised and dropped."""
        with self.lock:
            fcntl.flock(self.fd, fcntl.LOCK_EX)         # two programs on the duck may share the bus (this agent and
            try:                                        # `studio.py --duck`, the phone's server): one packet at a time
                return self._xfer(tx, n, timeout)
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _xfer(self, tx, n, timeout):
        termios.tcflush(self.fd, termios.TCIFLUSH)
        os.write(self.fd, tx)
        if not n:
            return b""
        end = time.time() + timeout + (len(tx) + 2 * n) * 1e-5         # a byte takes 10 us at 1 Mbit
        buf = self._read(n, end)
        if tx and buf[:len(tx)] == tx[:len(buf)]:                       # our own bytes came back first
            buf = (buf + self._read(len(tx), end + timeout))[len(tx):]
        if len(buf) < n:                                                # a slow or partial answer: let the bus
            while select.select([self.fd], [], [], 0.003)[0]:           # go quiet before anyone talks again,
                os.read(self.fd, 256)                                   # or the next packet collides with it
        return buf


class Player:
    """Plays the duck's own emotion clips: the head's four joints, the beak and the sound, from the pack kept in
    ANIM (built by anim_pack.py, sent once by a Mac studio). One clip at a time, on a 50 Hz thread of the bridge, so
    it runs the same whoever asked (the Mac, the phone) and with nobody connected at all.

    Only servos 30-34 are ever written. The legs are not touched: they stay as they are, limp or held. A clip is a
    list of keyframes [ms, neck, head pitch, head yaw, head roll (mrad from the default pose), beak 0..100], linear in
    between. A joint's target is its default angle plus the offset, kept inside the calibrated joint range, and
    becomes ticks through the calibration (zero and direction): so a duck that is not calibrated does not play.
    The head eases from wherever it is to the clip's first pose (0.5 s), plays, eases back to the default pose
    (0.4 s) and stays held there; `stop` lets it go limp.

    While the policies drive the duck (Drive, live) a clip still plays at once: it is mixed in on top (see `_run`)."""
    IDS = (30, 31, 32, 33, 34)
    DEFAULT = (0.3491, 0.3491, 0.0, 0.0, 0.0)      # rad at rest: neck and head pitch as the policies' default pose
    BEAK = 0.42                                    # rad, fully open (Duck Live's JAW_OPEN_RAD)
    # No head limits of its own: every head angle is bounded only by the joint ranges of the model the policies were
    # trained on (Pollen's MicroDuck USD in Isaac Lab = robot_groundcontact.xml = calibration.json min_rad / max_rad:
    # neck_pitch -90..+60 deg, head_pitch -90..+90, head_yaw -170..+170, head_roll -25..+25). `ticks` and the runner's
    # clip apply them. (Until 2026-11-20 the clips were also held to Duck Live's offset limits, head_pitch -0.75 rad
    # behind its 20 deg default: about -23 deg, a quarter of the joint's real travel.)

    def __init__(self, duck):
        self.duck, self.now, self.halt, self.lock, self.lib, self.stamp = duck, None, threading.Event(), threading.Lock(), None, None
        self.mix = None                            # while the policies drive: (weight 0..1, the four head angles [rad])
        self.twist = None                          # ... and the clip's walk command (vx, vy, yaw), duck_pilot.ClipBody

    def library(self):
        """{pack, clips} as the pack on the card says; read again when the file changes."""
        try:
            stamp = os.stat(ANIM + "/clips.json").st_mtime
        except OSError:
            return dict(pack=None, clips=[])
        if stamp != self.stamp:
            lib = json.load(open(ANIM + "/clips.json"))
            pack = open(ANIM + "/pack.sha").read().strip() if os.path.exists(ANIM + "/pack.sha") else None
            self.lib, self.stamp = dict(pack=pack, clips=lib["clips"], rate=lib["rate"]), stamp
        return self.lib

    def install(self, data):
        """A new pack (zip bytes): unpacked beside the old one, synced to the card, then swapped in."""
        import hashlib, io, shutil, zipfile
        self.stop()
        new = ANIM + ".new"
        shutil.rmtree(new, ignore_errors=True)
        os.makedirs(new)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if "/" in name or name.startswith("."):
                    raise OSError("bad pack")
                with open(f"{new}/{name}", "wb") as f:
                    f.write(z.read(name))
                    f.flush()
                    os.fsync(f.fileno())
        with open(new + "/pack.sha", "w") as f:
            f.write(hashlib.sha256(data).hexdigest())
            f.flush()
            os.fsync(f.fileno())
        shutil.rmtree(ANIM + ".old", ignore_errors=True)
        if os.path.exists(ANIM):
            os.rename(ANIM, ANIM + ".old")
        os.rename(new, ANIM)
        fd = os.open(os.path.dirname(ANIM), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        shutil.rmtree(ANIM + ".old", ignore_errors=True)
        self.stamp = None
        return self.library()["pack"]

    # ---- the five servos, in raw Feetech packets (the bridge has no other servo code) ----
    def _tx(self, sid, instr, params, reply=0):
        body = [sid, len(params) + 2, instr, *params]
        r = self.duck.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), reply, 0.02)
        return r if reply and len(r) == reply and r[:3] == bytes([0xFF, 0xFF, sid]) and (sum(r[2:]) & 0xFF) == 0xFF else None

    def _sync(self, addr, rows):                   # one packet for all five: {id: bytes}
        n = len(next(iter(rows.values())))
        self._tx(0xFE, 0x83, [addr, n, *[b for sid, v in rows.items() for b in (sid, *v)]])

    def _goto(self, ticks):
        sm = lambda t: abs(t) | (0x8000 if t < 0 else 0)          # sign-magnitude, as the HD1910 takes it
        self._sync(42, {sid: bytes([sm(t) & 0xFF, sm(t) >> 8, 0, 0, 0, 0]) for sid, t in zip(self.IDS, ticks)})

    def _present(self):
        out = []
        for sid in self.IDS:
            r = self._tx(sid, 2, [56, 2], 8) or self._tx(sid, 2, [56, 2], 8)
            if r is None:
                raise OSError(f"servo {sid} does not answer")
            v = r[5] | r[6] << 8
            out.append(-(v & 0x7FFF) if v & 0x8000 else v)
        return out

    def play(self, name):
        lib = self.library()
        clip = next((c for c in lib["clips"] if c["name"] == name), None)
        if clip is None:
            raise OSError(f"no animation called {name}")
        if self.duck.mode() != "setup":
            raise OSError("the duck is in Run: its runtime has the servos")
        if getattr(self.duck, "cal_on", False):
            raise OSError("a calibration is under way")
        cal = {j["id"]: j for j in (self.duck.calibration() or {}).get("joints", [])}
        if any(i not in cal for i in self.IDS):
            raise OSError("calibrate first: the head's zero is not known")
        self._present()                            # the head servos answer, or this raises before anything moves
        self.stop(release=False)
        self.halt.clear()
        self.now = dict(name=name, t=0.0, T=clip["T"])
        threading.Thread(target=self._run, args=(clip, [cal[i] for i in self.IDS], lib["rate"], self.now), daemon=True).start()
        return self.now

    def driven(self):
        """The policies have the joints right now (duck_agent.Drive, live)."""
        d = getattr(self.duck, "drive", None)
        return bool(d and d.info and d.info["live"] and not d.info.get("ended"))

    def stop(self, release=True):
        """End the clip now. `release`: also let the head go limp."""
        self.halt.set()
        with self.lock:                            # waits for the clip's thread to let go
            self.now = self.mix = None
            if release and self.duck.mode() == "setup" and not self.driven():      # under the policies the head is theirs
                try:
                    self._sync(40, {sid: bytes([0]) for sid in self.IDS})
                except OSError:
                    pass

    def _run(self, clip, cal, rate, mine):
        # In the turn each servo is counting in (the HD1910 counts past one turn; see Drive._run): set from the
        # first reading below, so a head servo read a turn away is never sent round a whole turn.
        turns = [0] * len(cal)
        ticks = lambda q: [round(j["zero_raw"] + n + j["direction"] * min(j["max_rad"], max(j["min_rad"], a)) * 4096 / (2 * math.pi))
                           for j, a, n in zip(cal, q, turns)]

        def pose(row):                             # a keyframe -> the five joint angles [rad]
            head = [v / 1000 for v in row[1:5]]                    # offsets; the model's joint ranges bound them
            return [d + o for d, o in zip(self.DEFAULT, head + [row[5] / 100 * self.BEAK])]

        keys, sound = clip["k"], None

        def at(ms):
            i = max(0, min(len(keys) - 2, next((n for n, k in enumerate(keys) if k[0] > ms), len(keys)) - 1))
            a, b = keys[i], keys[i + 1]
            w = min(1.0, max(0.0, (ms - a[0]) / (b[0] - a[0])))
            return pose([0] + [x + (y - x) * w for x, y in zip(a[1:], b[1:])])

        ease = lambda u: u * u * u * (u * (6 * u - 15) + 10)
        with self.lock:
            try:
                start = self._present()
                turns[:] = [round((p - j["zero_raw"]) / 4096) * 4096 for p, j in zip(start, cal)]
                first, rest, T = ticks(at(0)), ticks(list(self.DEFAULT)), clip["T"]
                over = self.driven()
                if not over:
                    self._goto(start)
                    self._sync(40, {sid: bytes([1]) for sid in self.IDS})
                else:                              # only the beak is ours to switch on
                    self._sync(42, {34: (lambda v: bytes([v & 0xFF, v >> 8, 0, 0, 0, 0]))(abs(start[4]) | (0x8000 if start[4] < 0 else 0))})
                    self._sync(40, {34: bytes([1])})
                import duck_pilot
                body = duck_pilot.ClipBody(clip) if over else None
                t0 = prev = time.monotonic()
                while not self.halt.is_set():
                    now = time.monotonic()
                    if body is not None and body.waiting:
                        t0 += now - prev                                   # a jump / flip / stand-up: the clip waits
                    prev = now
                    t = now - t0 - 0.5                                     # < 0: easing in; > T: easing out
                    if t >= T + 0.4:
                        break
                    if over and self.driven():
                        # The policies are driving. The clip jumps in on top, as in Duck Live: the head's four
                        # joints are handed to the runner as (weight, angles) and it writes them with its own
                        # packet, the policy balancing against them; the clip's legs (ClipBody: its walk command
                        # and its sit / jump / flip) go to the pilot, so the whole duck moves; the beak and the
                        # sound are played from here. The runner keeps the head inside the model's joint ranges.
                        pilot = self.duck.drive.pilot
                        for key in body.step(t, 0.02, pilot.state if pilot else "stand", pilot.still() if pilot else False):
                            self.duck.drive.press(key)
                        self.twist = body.twist
                        if body.done:
                            print("animation:", body.note, flush=True)
                            break
                        w = ease((t + 0.5) / 0.5) if t < 0 else 1.0 if t <= T else 1.0 - ease((t - T) / 0.4)
                        if body.waiting:
                            w = 0.0                                        # the skill's policy has the head too
                        q = at(min(T, max(0.0, t)) * 1000)
                        head = list(q[:4])
                        self.mix = (w, head)
                        self._sync(42, {34: (lambda v: bytes([v & 0xFF, v >> 8, 0, 0, 0, 0]))((lambda t: abs(t) | (0x8000 if t < 0 else 0))(ticks(q)[4] if 0 <= t <= T else rest[4]))})
                        if sound is None and clip["snd"] and t >= clip["snd_at"]:
                            sound = self._sound(random.choice(clip["snd"]), rate)
                        mine["t"] = round(min(T, max(0.0, t)), 2)
                        time.sleep(max(0.001, 0.02 - (time.monotonic() - t0) % 0.02))
                        continue
                    if over:                       # the policies stopped mid-clip: the clip ends with them
                        break
                    if t < 0:
                        w = ease((t + 0.5) / 0.5)
                        goal = [round(a + (b - a) * w) for a, b in zip(start, first)]
                    elif t <= T:
                        goal = ticks(at(t * 1000))
                        if sound is None and clip["snd"] and t >= clip["snd_at"]:
                            sound = self._sound(random.choice(clip["snd"]), rate)
                    else:
                        w, last = ease((t - T) / 0.4), ticks(at(T * 1000))
                        goal = [round(a + (b - a) * w) for a, b in zip(last, rest)]
                    self._goto(goal)
                    mine["t"] = round(min(T, max(0.0, t)), 2)
                    time.sleep(max(0.001, 0.02 - (time.monotonic() - t0) % 0.02))
            except OSError as e:
                print("animation:", e, flush=True)
            finally:
                self.mix = self.twist = None
                if sound and self.halt.is_set():
                    sound.kill()
                if self.now is mine:
                    self.now = None

    def _sound(self, name, rate):
        """Start one of the pack's sounds: 8-bit mu-law -> what the codec needs (48 kHz, 32-bit, stereo, -6 dB)."""
        import audioop
        pcm = audioop.ulaw2lin(open(f"{ANIM}/{name}", "rb").read(), 2)
        pcm = audioop.ratecv(pcm, 2, 1, rate, RATE, None)[0]
        pcm = audioop.tostereo(audioop.mul(audioop.lin2lin(pcm, 2, 4), 4, 0.5), 4, 1, 1)
        self.duck.sound = self.duck.codec()
        p = subprocess.Popen(["aplay", "-q", "-D", "hw:0,0", "-f", "S32_LE", "-r", str(RATE), "-c", "2", "-t", "raw"],
                             stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def feed():
            try:
                p.stdin.write(pcm)
                p.stdin.close()
            except OSError:
                pass
            p.wait()

        threading.Thread(target=feed, daemon=True).start()
        return p


class Drive:
    """Runs the duck's policies on the duck: 50 times a second it reads the IMU and the 14 joints, lets the pilot
    (duck_pilot.py: which policy, what command) compute 14 joint targets with the numpy policy pack (policy_net.py,
    POLICIES, built by policy_pack.py and sent once by a Mac studio), and writes them to the servos.

    `start(live=False)` is a dry run: everything is read and computed, nothing is written, the servos stay as they
    are. It shows what the policy sees and would do, and how well the loop keeps its 20 ms. `start(live=True)` drives
    the legs and head: torque on where each joint is, then the first BLEND seconds blend from there to the policy's target.
    `stop` ends it and lets every joint go limp. A driver who stops sending (0.5 s) counts as sticks released; ten
    control ticks in a row without a full reading from the joints or the IMU stop a live run.

    Every run, live or dry, is recorded tick by tick in RUNS (run_log.py: what the policy was given and answered,
    what was written, each servo's position, speed, load, current, temperature and voltage, the IMU, the loop's
    timing). The servos' load, current, temperature and voltage take a second read of the bus; it is made after the
    goals are written, so the policy gets its reading no later than without the log, and it is skipped on a tick
    that is already running long.

    Units: a joint angle is (ticks - zero) * 2 pi / 4096 * direction from the calibration, a speed is the servo's
    count * 50 ticks/s the same way; targets go back to ticks through the same numbers and are not clipped to the
    joint's range for the legs (the policies stand up by asking past it; the simulator does not clip either). The neck
    and head are held to the model's range, where the simulated joints stop."""
    IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14)        # duck_pilot.JOINTS order
    NECK_PAST = math.radians(3)                                           # how far past its range the simulated neck is pushed
    BLEND = 2.0                                                           # s from the pose it starts in to the policy's
    ALL = IDS + (34,)                                                     # the log's servo order: the joints, then the beak
    EXTRAS_BY = 0.014                                                     # s into a tick after which the log's read is skipped

    def __init__(self, duck):
        self.duck, self.halt, self.lock, self.pilot, self.info = duck, threading.Event(), threading.Lock(), None, None
        self.stick, self.stick_t, self.keys = (0.0, 0.0, 0.0, False), 0.0, []

    def pack(self):
        try:
            return open(POLICIES + ".sha").read().strip()
        except OSError:
            return None

    def install(self, data):
        import hashlib
        self.stop()
        for path, body in ((POLICIES + ".part", data), (POLICIES + ".sha.part", hashlib.sha256(data).hexdigest().encode())):
            with open(path, "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
        os.replace(POLICIES + ".part", POLICIES)
        os.replace(POLICIES + ".sha.part", POLICIES + ".sha")
        return self.pack()

    def start(self, live):
        if self.duck.mode() != "setup":
            raise OSError("the duck is in Run: its runtime has the servos")
        if getattr(self.duck, "cal_on", False):
            raise OSError("a calibration is under way")
        cal = {j["id"]: j for j in (self.duck.calibration() or {}).get("joints", [])}
        if any(i not in cal for i in self.IDS) or not ((self.duck.calibration() or {}).get("imu") or {}).get("mount"):
            raise OSError("calibrate first: joints and IMU")
        if not os.path.exists(POLICIES):
            raise OSError("no policies on the duck yet: connect a Mac studio once")
        if self._read() is None or self.duck.imu() is None:
            raise OSError("the joints or the IMU do not answer: is the servo power on?")
        self.stop(release=False)
        self.duck.player.stop(release=False)
        self.halt.clear()
        self.info = dict(live=bool(live), ticks=0, late=0, missed=0, ms=0.0, hz=0.0)
        threading.Thread(target=self._run, args=(bool(live), [cal[i] for i in self.IDS], self.info), daemon=True).start()
        return self.status()

    def stop(self, release=True):
        was = self.info
        self.halt.set()
        with self.lock:
            self.info = self.pilot = None
            if release and was and was["live"] and self.duck.mode() == "setup":
                try:
                    self._sync(40, {sid: bytes([0]) for sid in self.IDS})
                except OSError:
                    pass

    def drive(self, x=0.0, y=0.0, w=0.0, run=False):
        self.stick, self.stick_t = (float(x), float(y), float(w), bool(run)), time.monotonic()

    def press(self, key):
        if key not in ("sit", "jump", "flip", "kick", "faster", "slower"):
            raise OSError(f"no such key: {key}")
        self.keys.append(key)

    def status(self):
        p, i = self.pilot, self.info
        return None if i is None else dict(i, **(p.status() if p else {}))

    # ---- the bus ----
    def _sync(self, addr, rows):
        n = len(next(iter(rows.values())))
        body = [0xFE, len(rows) * (n + 1) + 4, 0x83, addr, n, *[b for sid, v in rows.items() for b in (sid, *v)]]
        self.duck.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), 0, 0)

    def _read(self):
        """{id: (position ticks, speed counts)} for all 14 from one sync read of registers 56-59, or None if any is missing."""
        body = [0xFE, len(self.IDS) + 4, 0x82, 56, 4, *self.IDS]
        r = self.duck.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), 10 * len(self.IDS), 0.008)
        sm = lambda v: -(v & 0x7FFF) if v & 0x8000 else v
        out = {r[i + 2]: (sm(r[i + 5] | r[i + 6] << 8), sm(r[i + 7] | r[i + 8] << 8)) for i in range(0, len(r) - 9, 10)
               if r[i:i + 2] == b"\xff\xff" and (sum(r[i + 2:i + 10]) & 0xFF) == 0xFF}
        return out if all(i in out for i in self.IDS) else None

    def _block(self, addr, n, timeout=0.004):
        """{id: the n bytes from register `addr`} for the servos of ALL that answered one sync read."""
        body = [0xFE, len(self.ALL) + 4, 0x82, addr, n, *self.ALL]
        r = self.duck.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), (n + 6) * len(self.ALL), timeout)
        return {r[i + 2]: r[i + 5:i + 5 + n] for i in range(0, len(r) - n - 5, n + 6)
                if r[i:i + 2] == b"\xff\xff" and r[i + 3] == n + 2 and (sum(r[i + 2:i + n + 6]) & 0xFF) == 0xFF}

    def _extras(self):
        """Per servo of ALL (NaN where one did not answer): load -1000..1000, current counts, temperature deg C,
        supply V, status byte. Registers 60-70 in one sync read."""
        got, nan = self._block(60, 11), float("nan")
        sm = lambda v, bit: -(v & ((1 << bit) - 1)) if v & (1 << bit) else v
        rows = [(sm(b[0] | b[1] << 8, 10), sm(b[9] | b[10] << 8, 15), b[3], b[2] / 10, b[5]) if b else (nan,) * 5
                for b in (got.get(i) for i in self.ALL)]
        return list(zip(*rows))

    def _recorder(self, live, cal):
        """Open this run's log, or None (and why in the status) if it cannot be: a run never waits on its log."""
        import duck_pilot, run_log
        try:
            gains = {i: list(b) for i, b in self._block(50, 2, 0.01).items()}        # P, D as the servos hold them now
        except OSError:
            gains = {}
        names = {j["id"]: j.get("name") for j in (self.duck.calibration() or {}).get("joints", [])}
        return run_log.Recorder(RUNS, dict(
            live=live, host=os.uname().nodename, pack=self.pack(), dt=duck_pilot.DT, ids=list(self.IDS),
            joints=list(duck_pilot.JOINTS), servo_ids=list(self.ALL), servos=[*duck_pilot.JOINTS, "mouth"],
            default_pose=[float(v) for v in duck_pilot.DEFAULT_POSE], calibration=self.duck.calibration(),
            names=names, gains_p_d=gains, blend_s=self.BLEND))

    def _run(self, live, cal, info):
        import numpy as np
        import duck_pilot, policy_net
        k = np.array([j["direction"] * 2 * math.pi / 4096 for j in cal])
        zero = np.array([j["zero_raw"] for j in cal], dtype=float)
        # The policies were trained on a model whose joints stop at their limits whatever is asked: getting up, the
        # walk policy asks the neck for -125 to -135 deg and the simulated neck stops at -93 (its range is -90..60).
        # The real neck has no such stop at the same place, so its targets, and the head's, are held to the model's
        # range (plus NECK_PAST, what the simulated joint gives under load). The legs are deliberately not: the
        # policies stand up by asking past the range, and holding those back takes the force away.
        lo = np.array([j["min_rad"] for j in cal]) - self.NECK_PAST
        hi = np.array([j["max_rad"] for j in cal]) + self.NECK_PAST
        word = lambda t: (lambda w: bytes([w & 0xFF, w >> 8, 0, 0, 0, 0]))(abs(t) | (0x8000 if t < 0 else 0))
        # The HD1910 counts on past one turn (it reads e.g. 4940 or 6412, not 0-4095), so a joint can be read a whole
        # turn away from its zero. 2026-11-20: the left hip yaw was read at -196.6 deg, the policies saw that, and its
        # goal (written in the frame of the turn it should have been in) sent the servo the long way round into its
        # stop at full load. Every joint's range is within +-180 deg of its zero, so the reading is taken in the turn
        # nearest the zero (`turn`, from each reading) and every goal is written in the servo's own turn.
        turn_of = lambda p: np.round((p - zero) / 4096.0) * 4096.0
        rec, pilot, keymap = None, None, {}
        with self.lock:
            try:
                pilot = self.pilot = duck_pilot.Pilot(policy_net.load(POLICIES))
                first = self._read()
                start = np.array([first[i][0] for i in self.IDS], dtype=float)
                if live:
                    self._sync(42, {sid: word(int(t)) for sid, t in zip(self.IDS, start)})
                    self._sync(40, {sid: bytes([1]) for sid in self.IDS})
                try:
                    rec, keymap = self._recorder(live, cal), duck_pilot_keys()
                    info["log"] = rec.name
                except Exception as e:                                   # no card, no room, no module: drive on
                    info["log_error"] = str(e)
                t0 = nxt = time.monotonic()
                gyro, quat, pos, vel, bad = np.zeros(3), (1.0, 0.0, 0.0, 0.0), start.copy(), np.zeros(14), 0
                raw, sens, late, cpu = (float("nan"),) * 4, (float("nan"),) * 3, 0.0, (float("nan"),) * 2
                while not self.halt.is_set():
                    t = time.monotonic()
                    got, imu = self._read(), self.duck.imu()
                    if got is None or imu is None:
                        bad += 1
                        info["missed"] += 1
                        if bad >= 10 and live:
                            raise OSError("lost the joints or the IMU: stopped")
                    else:
                        bad = 0
                        pos = np.array([got[i][0] for i in self.IDS], dtype=float)
                        vel = np.array([got[i][1] for i in self.IDS], dtype=float)
                        gyro, quat, raw, sens = np.radians(imu["rate"]), imu["quat"], imu["raw"], imu["gyro"]
                    x, y, w, run = self.stick if time.monotonic() - self.stick_t < 0.5 else (0.0, 0.0, 0.0, False)
                    pilot.drive(x, y, w, run)
                    pilot.twist = self.duck.player.twist                 # a clip's wiggle / side-step, if one plays
                    pressed = 0
                    while self.keys:
                        key = self.keys.pop(0)
                        pressed |= keymap.get(key, 0)
                        pilot.press(key)
                    turn = turn_of(pos)
                    target = pilot.tick(gyro, quat, (pos - turn - zero) * k, vel * 50 * k)
                    mix = self.duck.player.mix                           # a clip playing on top: it has the head
                    if mix and pilot.state in ("stand", "walk", "stopping", "seated", "run", "sprint"):
                        target = target.copy()
                        target[5:9] = (1 - mix[0]) * target[5:9] + mix[0] * np.array(mix[1])
                    target[5:9] = np.clip(target[5:9], lo[5:9], hi[5:9])       # neck and head: the model's own stops
                    ticks = np.clip(np.round(zero + turn + target / k), -32767, 32767)
                    u = 1.0
                    if live:
                        u = min(1.0, (t - t0) / self.BLEND)                    # the first seconds: from where it was
                        u = u * u * (3 - 2 * u)
                        ticks = np.round(start + (ticks - start) * u)
                        self._sync(42, {sid: word(int(v)) for sid, v in zip(self.IDS, ticks)})
                    n = info["ticks"] = info["ticks"] + 1
                    ms = (time.monotonic() - t) * 1e3
                    if rec and rec.error is None:
                        try:
                            if n % 50 == 1:
                                cpu = (cpu_temp(), os.getloadavg()[0])
                            for name, value in (
                                    ("t", t - t0), ("ms", ms), ("late", late), ("missed", float(bad > 0)),
                                    ("state", rec.meta["states"].index(pilot.state) if pilot.state in rec.meta["states"] else -1),
                                    ("role", rec.meta["roles"].index(pilot.role) if pilot.role in rec.meta["roles"] else -1),
                                    ("level", pilot.level), ("tilt", pilot.tilt), ("falls", pilot.falls), ("blend", u),
                                    ("stick", (x, y, w, float(run))), ("keys", pressed), ("cmd", pilot.v),
                                    ("clip", mix[0] if mix else 0.0), ("quat", quat), ("quat_raw", raw), ("gyro", sens),
                                    ("rate", np.degrees(gyro)), ("obs", pilot.obs), ("action", pilot.last),
                                    ("target", target), ("goal", ticks), ("pos", pos), ("vel", vel), ("cpu_temp", cpu[0]),
                                    ("cpu_load", cpu[1])):
                                rec.put(name, value)
                            if time.monotonic() - t < self.EXTRAS_BY:    # the servos' own measurements, after the write
                                for name, value in zip(("load", "current", "temp", "volt", "status"), self._extras()):
                                    rec.put(name, value)
                            rec.commit()
                        except Exception as e:                           # the log must never stop a run
                            rec.error = info["log_error"] = str(e)
                    elif rec and "log_error" not in info:
                        info["log_error"] = rec.error
                    info["ms"] = round(info["ms"] + (ms - info["ms"]) / min(n, 100), 2)
                    info["hz"] = round(n / max(1e-3, time.monotonic() - t0), 1)
                    if n % 5 == 0:                                       # for the page: the largest step asked of a joint
                        info["off_deg"] = round(float(np.degrees(np.abs((ticks - pos) * k)).max()), 1)
                        info["ask_deg"] = [round(float(v), 1) for v in np.degrees(target)]      # per joint, the policies' order:
                        info["at_deg"] = [round(float(v), 1) for v in np.degrees((pos - turn - zero) * k)]  # asked, and where it is
                    nxt += duck_pilot.DT
                    wait = nxt - time.monotonic()
                    late = float(wait < 0)
                    if wait < 0:
                        info["late"] += 1
                        nxt = time.monotonic()
                    else:
                        time.sleep(wait)
            except Exception as e:
                info["error"] = str(e)
                print("drive:", e, flush=True)
            finally:
                if live:
                    try:
                        self._sync(40, {sid: bytes([0]) for sid in self.IDS})
                    except OSError:
                        pass
                info["ended"] = True
                if rec:
                    rec.close(ticks=info["ticks"], late=info["late"], missed=info["missed"], error=info.get("error"),
                              falls=getattr(pilot, "falls", None), note=getattr(pilot, "note", None))


def cpu_temp():
    """The computer's temperature in deg C, NaN where there is no such sensor."""
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except (OSError, ValueError):
        return float("nan")


def duck_pilot_keys():
    """{key: its bit in the run log's `keys` column}."""
    import run_log
    return {k: 1 << i for i, k in enumerate(run_log.KEYS)}


class Sensors:
    def set_cal(self, on):
        """A calibration is under way (or over). Kept here, by the one bridge, so that every studio shows the same."""
        self.cal_on = bool(on)
        if on and hasattr(self, "player"):
            self.player.stop(release=False)

    def bus(self, tx, n, timeout):
        if getattr(self, "running", False):      # Run: the runtime has the port. A packet of ours would collide with
            raise OSError("the duck's runtime has the servo bus (Run mode)")       # its own and take its replies
        if getattr(self, "uart", None) is None:
            self.uart = Uart()
        uart = self.uart
        try:
            return uart.xfer(tx, n, timeout)
        except OSError:                          # the port failed under us (or was handed to the runtime meanwhile):
            if self.uart is uart:                # drop it, the next packet opens it again
                self.uart = None
                try:
                    os.close(uart.fd)
                except OSError:
                    pass
            raise

    def mode(self):
        """ "run" while the duck's runtime is up, "setup" when it is paused and the devices are ours."""
        up = subprocess.run(["systemctl", "is-active", "--quiet", RUNTIME[0]]).returncode == 0
        if up and not getattr(self, "running", False):       # started by someone else: let go of the port now
            self._release()
        self.running = up
        return "run" if up else "setup"

    def _release(self):
        uart, self.uart = getattr(self, "uart", None), None
        if uart:
            os.close(uart.fd)

    def set_mode(self, to):
        """Hand the devices over. To "run": let go of the bus and the ToF, then start the runtime.
        To "setup": stop the runtime (its servos go limp). Raises if the sudoers rule is not installed."""
        to = "run" if to == "run" else "setup"
        if to == "run":
            self.player.stop()                   # a clip in progress ends and the head is let go first
            self.drive.stop()
            self.running = True                  # from here on no packet of ours goes out
            self._release()
            if self.eye and self.eye.proc:
                self.eye.proc.kill()
        # Not waited for to the end: the runtime's ToF service can take 90 s to stop, and a start queued behind
        # that stop waits as long. What decides the mode is the control daemon, which switches in under a second.
        p = subprocess.Popen(["sudo", "-n", "/usr/bin/systemctl", "start" if to == "run" else "stop", *RUNTIME],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        end = time.time() + 9
        while time.time() < end:
            if p.poll():                         # it failed outright
                err = p.stderr.read()
                self.mode()
                raise OSError("not allowed to switch: run mode_install.sh on the duck once" if "password" in err
                              else err.strip() or "systemctl failed")
            if self.mode() == to:
                return to
            time.sleep(0.2)
        raise OSError("the duck is still switching: try again in a minute")

    def codec(self):
        """Program the audio codec. True if it is there. Safe to repeat: no reset, the same values again."""
        try:
            fd = os.open(f"/dev/i2c-{CODEC[0]}", os.O_RDWR)
        except OSError:
            return False
        try:
            fcntl.ioctl(fd, 0x0703, CODEC[1])
            for reg, value in CODEC_SETUP:
                os.write(fd, bytes([reg, value]))
            return True
        except OSError:
            return False
        finally:
            os.close(fd)

    def __init__(self):
        once = f"/run/user/{os.getuid()}/duck-agent.booted"      # /run is emptied by a reboot
        if not os.path.exists(once) and os.path.isdir(os.path.dirname(once)):
            open(once, "w").close()              # first start since boot: take the devices from the runtime
            subprocess.run(["sudo", "-n", "/usr/bin/systemctl", "stop", *RUNTIME], capture_output=True, timeout=40)
        self.sound = self.codec()
        self.player = Player(self)
        self.drive = Drive(self)
        threading.Thread(target=self._bank, daemon=True).start()
        self.cam = Pipe(["v4l2-ctl", "-d", "/dev/video0", f"--set-fmt-video=width={W},height={H},pixelformat=NV12",
                         "--stream-mmap=3", "--stream-to=-"], W * H * 3 // 2)
        self.eye = None                      # the ToF reader, once a sensor has been seen
        self.ear = Pipe(["arecord", "-q", "-D", "hw:0,0", "-f", "S32_LE", "-r", str(RATE), "-c", "2", "-t", "raw"],
                        CHUNK * 8)

    def status(self):
        buses = [int(p.rsplit("-", 1)[1]) for p in glob.glob("/dev/i2c-*")]
        names = [open(p).read().strip() for p in glob.glob("/sys/bus/i2c/devices/*/name")]
        sensor = next((n for n in names if n.startswith(("imx", "ov", "gc"))), None)
        mode = self.mode()
        bus = mode == "setup" and os.access(UART, os.R_OK | os.W_OK)
        imu = bus and self.imu() is not None
        held = "the duck's runtime has it (Run mode)"
        ranging = self.eye is not None and self.eye.proc is not None
        kind = self.kind if ranging else self.tof_kind() if mode == "setup" else None
        cards = open("/proc/asound/cards").read() if os.path.exists("/proc/asound/cards") else ""
        ear = self.mic() if self.ear.proc else None
        return dict(
            mode=mode, calibrating=getattr(self, "cal_on", False), anim=self.player.now, pilot=self.drive.status(),
            camera=dict(ok=bool(sensor), note=sensor or "no sensor on CSI"),
            tof=dict(ok=bool(kind) and os.path.exists(f"{HOME}/tof_{kind}"),
                     note=held if mode == "run" else "no sensor at 0x29 on the HAT socket IO2/3" if not kind else
                     kind if os.path.exists(f"{HOME}/tof_{kind}") else f"{kind} found, reader not built"),
            imu=dict(ok=bool(imu), note="" if imu else held if mode == "run" else
                     "ID 200 silent on the servo bus" if bus else "servo bus not readable"),
            mic=dict(ok=self.sound and (ear is None or ear["live"]),
                     note="no signal" if ear and not ear["live"] else "" if self.sound else "no codec on I2C3"),
            speaker=dict(ok=self.sound, note="" if self.sound else "no codec on I2C3"),
            servos=dict(ok=bus, note="" if bus else held if mode == "run" else
                        f"no permission for {UART}: sudo usermod -aG dialout $USER, then reboot"),
            volts=self.volts() if bus else None, host=os.uname().nodename, load=os.getloadavg()[0],
            temp=int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000)

    def camera(self):
        """Newest frame, NV12, W x H; None until the first one arrives."""
        return self.cam.get()

    def mic(self):
        """{level: RMS in dBFS, wave: 64 samples -1..1, live: the line carries data}, or None until audio arrives."""
        rec = self.ear.get()
        if rec is None:
            return None
        s = array.array("i", rec)[1::2]                     # right channel: the microphone is on MIC3R
        mean = sum(s) / len(s)
        rms = math.sqrt(sum((x - mean) ** 2 for x in s) / len(s)) / 2 ** 31
        step = len(s) // 64
        return dict(level=round(20 * math.log10(max(rms, 1e-9)), 1), live=len(set(s[:2000])) > 8,
                    wave=[round((s[i * step] - mean) / 2 ** 31, 4) for i in range(64)])

    def tof_kind(self):
        """Which ranger answers at 0x29 on I2C3: "vl53l5cx", "vl53l8cx" or None. Not to be called while it ranges:
        the ID registers sit in another register bank, and switching banks would upset a running sensor."""
        try:
            fd = os.open(f"/dev/i2c-{CODEC[0]}", os.O_RDWR)
        except OSError:
            return None
        try:
            fcntl.ioctl(fd, 0x0703, 0x29)
            os.write(fd, b"\x7f\xff\x00")
            os.write(fd, b"\x00\x00")
            dev, rev = os.read(fd, 2)
            os.write(fd, b"\x7f\xff\x02")
            return {0x02: "vl53l5cx", 0x0C: "vl53l8cx"}.get(rev) if dev == 0xF0 else None
        except OSError:
            return None
        finally:
            os.close(fd)

    def tof(self):
        """64 distances in mm, row by row in the sensor's own order; -1 where a zone saw nothing valid. None until
        the first frame (the sensor takes a few seconds to load its firmware) or with no sensor."""
        if self.eye is None or self.eye.proc is None:
            self.kind = self.tof_kind()
            if not self.kind or not os.path.exists(f"{HOME}/tof_{self.kind}"):
                return None
            self.eye = Pipe([f"{HOME}/tof_{self.kind}", f"/dev/i2c-{CODEC[0]}", "15"], None)
        line = self.eye.get()
        return [int(v) for v in line.split()] if line else None

    def imu(self):
        """The body IMU: {quat: [w, x, y, z], roll, pitch, yaw in degrees, gyro in deg/s}, or None if it is silent.
        Register 124 of bus ID 200 holds gyro x/y/z as int16 (17.5 mdps per count) and the quaternion's x/y/z as
        half-precision floats; w is what makes it a unit quaternion. Yaw has no absolute reference and drifts."""
        body = [200, 4, 2, 124, 12]
        for _ in range(3):                                   # the bus is shared with the servos: a miss is retried
            try:
                r = self.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), 18, 0.02)
            except OSError:
                return None
            if len(r) == 18 and r[:3] == b"\xff\xff\xc8" and (sum(r[2:]) & 0xFF) == 0xFF:
                break
        else:
            return None
        gx, gy, gz, x, y, z = struct.unpack("<3h3e", r[5:17])
        raw = (math.sqrt(max(0.0, 1 - x * x - y * y - z * z)), x, y, z)
        return pose(raw, self.mount(), [round(g * 0.0175, 2) for g in (gx, gy, gz)])

    def volts(self):
        """The servo supply (the battery) in volts, as the first joint servo that answers reports it. None if none do."""
        for sid in (10, 20, 30, 11, 21, 31):
            body = [sid, 4, 2, 62, 1]
            try:
                r = self.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), 7, 0.01)
            except OSError:
                return None
            if len(r) == 7 and r[2] == sid and (sum(r[2:]) & 0xFF) == 0xFF:
                return r[5] / 10
        return None

    def mount(self):
        """The IMU's calibrated mounting rotation [w, x, y, z], or None. Read from the calibration file when it
        changes, so a new calibration takes effect at once."""
        try:
            stamp = os.stat(CALIBRATION).st_mtime
        except OSError:
            return None
        if stamp != getattr(self, "_stamp", None):
            self._stamp, self._mount = stamp, ((self.calibration() or {}).get("imu") or {}).get("mount")
        return self._mount

    def _bank(self):
        """Render the duck's voice bank if it is not there yet (about a minute, once)."""
        try:
            import duck_voice
            os.makedirs(HOME, exist_ok=True)
            duck_voice.render_all(BANK)
        except Exception as e:
            print("voice bank:", e, flush=True)

    def say(self, tag="chirp"):
        """Play one of the duck's own sounds: a random variant of `tag` (chirp = the quack, greet, coo, peck,
        inquire, alarm). Returns the file played, or None if that tag has not been rendered yet."""
        files = sorted(glob.glob(f"{BANK}/{tag}/{tag}_?.wav"))
        if not files:
            return None
        name = random.choice(files)
        with wave.open(name) as f:
            mono = array.array("h", f.readframes(f.getnframes()))
        pcm = array.array("i", (v << 15 for s in mono for v in (s, s)))       # 16-bit mono -> 32-bit stereo, -6 dB
        self.sound = self.codec()
        subprocess.run(["aplay", "-q", "-D", "hw:0,0", "-f", "S32_LE", "-r", str(RATE), "-c", "2", "-t", "raw"],
                       input=pcm.tobytes(), timeout=8, stderr=subprocess.DEVNULL)
        return os.path.basename(name)

    def calibration(self):
        """The saved calibration, or None. Written only by `save_calibration`."""
        try:
            return json.load(open(CALIBRATION))
        except (OSError, ValueError):
            return None

    def save_calibration(self, data):
        os.makedirs(os.path.dirname(CALIBRATION), exist_ok=True)
        with open(CALIBRATION + ".part", "w") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())                              # on the card before it replaces the old one
        os.replace(CALIBRATION + ".part", CALIBRATION)        # all or nothing: never a half-written file
        d = os.open(os.path.dirname(CALIBRATION), os.O_RDONLY)
        os.fsync(d)                                           # and the rename itself: a power cut now keeps the new file
        os.close(d)


class Mock(Sensors):
    """Made-up sensors with the same answers, so the studio can be tried with no duck."""

    def __init__(self):
        self.t0 = time.time()

    def status(self):
        ok = dict(ok=True, note="mock")
        return dict(mode=getattr(self, "_mode", "setup"), calibrating=getattr(self, "cal_on", False), anim=None,
                    camera=ok, tof=ok, imu=ok, mic=ok, speaker=ok,
                    servos=dict(ok=False, note="mock"), volts=7.4, host="mock",
                    load=0.2, temp=41.0)

    def camera(self):
        t = time.time() - self.t0
        cx, cy = W / 2 + 90 * math.sin(t), H / 2 + 50 * math.cos(1.3 * t)
        y = bytearray(W * H)
        for j in range(H):
            row = j * W
            for i in range(0, W, 2):
                v = 60 + (i * 120) // W + (40 if (i // 40 + j // 40) % 2 else 0)
                if (i - cx) ** 2 + (j - cy) ** 2 < 900:
                    v = 235
                y[row + i] = y[row + i + 1] = v
        uv = bytearray(W * H // 2)
        for j in range(H // 2):
            for i in range(0, W, 2):
                uv[j * W + i], uv[j * W + i + 1] = 90 + (i * 70) // W, 110 + (j * 80) // H
        return bytes(y + uv)

    def tof(self):
        """64 distances in mm, row 0 = top, column 0 = the sensor's left: a wall at 0.9 m, a blob passing at 0.3 m."""
        t = time.time() - self.t0
        cx, cy = 3.5 + 3 * math.sin(0.8 * t), 3.5 + 1.5 * math.cos(0.5 * t)
        return [300 if (c - cx) ** 2 + (r - cy) ** 2 < 2.2 else 900 + 12 * r for r in range(8) for c in range(8)]

    def mic(self):
        t = time.time() - self.t0
        a = 0.05 + 0.4 * abs(math.sin(0.7 * t)) ** 3
        return dict(level=round(20 * math.log10(a / math.sqrt(2)), 1), live=True,
                    wave=[round(a * math.sin(2 * math.pi * (i / 9 + 3 * t)), 4) for i in range(64)])

    def imu(self):
        """Roll and pitch, degrees."""
        t = time.time() - self.t0
        r, p = math.radians(8 * math.sin(0.6 * t) + 6) / 2, math.radians(5 * math.sin(0.9 * t + 1) - 4) / 2      # a tilted mount
        raw = qmul((math.cos(p), 0, math.sin(p), 0), (math.cos(r), math.sin(r), 0, 0))
        return pose(raw, ((self.calibration() or {}).get("imu") or {}).get("mount"), [0.0, 0.0, 0.0])

    def say(self, tag="chirp"):
        return f"{tag}_a.wav"

    set_cal = Sensors.set_cal

    def set_mode(self, to):
        self._mode = to
        return to

    def calibration(self):
        return getattr(self, "saved", None)

    def save_calibration(self, data):
        self.saved = data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        fn = getattr(DUCK, self.path.strip("/"), None) if self.path.strip("/") in ("status", "camera", "tof", "mic", "imu", "calibration") else None
        if self.path == "/anims" and hasattr(DUCK, "player"):
            fn = lambda: (lambda lib: dict(pack=lib["pack"], clips=[{k: v for k, v in c.items() if k != "k"} for c in lib["clips"]]))(DUCK.player.library())
        out = fn() if fn else None
        body, kind = (out, "application/octet-stream") if isinstance(out, bytes) else (json.dumps(out).encode(), "application/json")
        self.send_response(200 if fn else 404)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        out = b"{}"
        if self.path.startswith("/say/"):
            out = json.dumps(DUCK.say(self.path[5:])).encode()
        elif self.path.startswith("/pilot/") and hasattr(DUCK, "drive"):
            try:
                what, d = self.path[7:], DUCK.drive
                out = json.dumps(dict(pack=d.install(body)) if what == "install" else
                                 dict(pilot=d.start(what == "start/live")) if what.startswith("start/") else
                                 dict(pilot=d.stop()) if what == "stop" else
                                 dict(pilot=d.drive(**json.loads(body))) if what == "drive" else
                                 dict(pack=d.pack()) if what == "pack" else
                                 dict(pilot=d.press(what[6:]))).encode()
            except (OSError, ValueError, TypeError) as e:
                out = json.dumps(dict(error=str(e))).encode()
        elif self.path.startswith("/anim/") and hasattr(DUCK, "player"):
            try:
                what = self.path[6:]
                out = json.dumps(dict(pack=DUCK.player.install(body)) if what == "install" else
                                 dict(now=DUCK.player.stop()) if what == "stop" else
                                 dict(now=DUCK.player.play(what[5:]))).encode()
            except (OSError, ValueError) as e:
                out = json.dumps(dict(error=str(e))).encode()
        elif self.path.startswith("/cal/"):
            DUCK.set_cal(self.path[5:] == "1")
        elif self.path == "/calibration":
            DUCK.save_calibration(json.loads(body))
        elif self.path.startswith("/mode/"):
            try:
                out = json.dumps(dict(mode=DUCK.set_mode(self.path[6:]))).encode()
            except OSError as e:
                out = json.dumps(dict(error=str(e))).encode()
        elif self.path.startswith("/bus/"):                  # /bus/<reply bytes>/<timeout ms>, body = the packet
            n, ms = self.path.split("/")[2:4]
            out = DUCK.bus(body, int(n), int(ms) / 1000)
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--install", action="store_true", help="run this file as the service duck-agent, now and at boot")
    a = ap.parse_args()
    if a.install:
        unit = os.path.expanduser("~/.config/systemd/user/duck-agent.service")
        os.makedirs(os.path.dirname(unit), exist_ok=True)
        with open(unit, "w") as f:
            f.write(UNIT.format(path=os.path.abspath(__file__)))
        subprocess.run(["pkill", "-f", "^python3 -u /tmp/duck_agent.py"])      # an agent from before the service
        subprocess.run(["loginctl", "enable-linger"])                           # user services run without a login
        for cmd in ("daemon-reload", "enable duck-agent", "restart duck-agent"):
            subprocess.run(["systemctl", "--user", *cmd.split()], check=True, capture_output=True)
        raise SystemExit(0)
    for _ in range(40):                          # the agent this one replaces may still be letting go of the port
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
            break
        except OSError:
            time.sleep(0.25)
    else:
        raise SystemExit(f"port {a.port} is taken")
    DUCK = Mock() if a.mock else Sensors()       # only the agent that owns the port takes the devices
    print(f"duck agent on 127.0.0.1:{a.port}", flush=True)       # only once it is listening: the studio acts on this line
    srv.serve_forever()
