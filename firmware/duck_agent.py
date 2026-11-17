"""Duck agent: serves the sensors on the duck's computer (Radxa Zero 3W) to the studio. Python stdlib only.

    python3 duck_agent.py            # on the duck; listens on 127.0.0.1:8771, reached through an SSH tunnel
    python3 duck_agent.py --mock     # anywhere: simulated sensors

The duck's own runtime (robotd, tofd) uses the servo bus and the ToF ranger while it runs, and only one program
can. (Its camera service, mediad, is switched off by `mode_install.sh`: on this board it leaves the camera broken.) So the duck is in one of two modes (`Sensors.mode`): "run", the runtime has them; "setup",
the runtime is paused and this agent does. Switching needs the one sudoers rule `mode_install.sh` adds.

The studio copies this file to the duck and starts it over SSH (`Studio.brain_connect`); nothing is installed and
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
BANK, CALIBRATION = HOME + "/sounds", HOME + "/calibration.json"
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
        self.cmd, self.size, self.last, self.asked, self.proc = cmd, size, None, 0.0, None

    def get(self):
        self.asked = time.time()
        if self.proc is None:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return self.last

    def _read(self, proc):
        while time.time() - self.asked < IDLE:
            rec = proc.stdout.read(self.size) if self.size else proc.stdout.readline()
            if len(rec) < (self.size or 1):
                break
            self.last = rec
        proc.kill()
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
    return dict(quat=[round(v, 5) for v in q], raw=[round(v, 5) for v in raw], gyro=gyro, calibrated=bool(mount),
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
            termios.tcflush(self.fd, termios.TCIFLUSH)
            os.write(self.fd, tx)
            if not n:
                return b""
            end = time.time() + timeout + len(tx) * 1e-5
            buf = self._read(n, end)
            if tx and buf[:len(tx)] == tx[:len(buf)]:                       # our own bytes came back first
                buf = (buf + self._read(len(tx), end + timeout))[len(tx):]
            return buf


class Sensors:
    def bus(self, tx, n, timeout):
        if getattr(self, "uart", None) is None:
            self.uart = Uart()
        return self.uart.xfer(tx, n, timeout)

    def mode(self):
        """ "run" while the duck's runtime is up, "setup" when it is paused and the devices are ours."""
        up = subprocess.run(["systemctl", "is-active", "--quiet", RUNTIME[0]]).returncode == 0
        return "run" if up else "setup"

    def set_mode(self, to):
        """Hand the devices over. To "run": let go of the bus and the ToF, then start the runtime.
        To "setup": stop the runtime (its servos go limp). Raises if the sudoers rule is not installed."""
        if to == "run":
            uart, self.uart = getattr(self, "uart", None), None
            if uart:
                os.close(uart.fd)
            if self.eye and self.eye.proc:
                self.eye.proc.kill()
        r = subprocess.run(["sudo", "-n", "/usr/bin/systemctl", "start" if to == "run" else "stop", *RUNTIME],
                           capture_output=True, text=True, timeout=40)
        if r.returncode:
            raise OSError("not allowed to switch: run mode_install.sh on the duck once" if "password" in r.stderr
                          else r.stderr.strip() or "systemctl failed")
        return self.mode()

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
        self.sound = self.codec()
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
            mode=mode,
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
        try:
            r = self.bus(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]), 18, 0.02)
        except OSError:
            return None
        if len(r) != 18 or r[:3] != b"\xff\xff\xc8" or (sum(r[2:]) & 0xFF) != 0xFF:
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
        os.makedirs(HOME, exist_ok=True)
        with open(CALIBRATION + ".part", "w") as f:
            json.dump(data, f, indent=1)
        os.replace(CALIBRATION + ".part", CALIBRATION)        # all or nothing: never a half-written file


class Mock(Sensors):
    """Made-up sensors with the same answers, so the studio can be tried with no duck."""

    def __init__(self):
        self.t0 = time.time()

    def status(self):
        ok = dict(ok=True, note="mock")
        return dict(mode=getattr(self, "_mode", "setup"), camera=ok, tof=ok, imu=ok, mic=ok, speaker=ok,
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
    a = ap.parse_args()
    DUCK = Mock() if a.mock else Sensors()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"duck agent on 127.0.0.1:{a.port}", flush=True)       # only once it is listening: the studio acts on this line
    srv.serve_forever()
