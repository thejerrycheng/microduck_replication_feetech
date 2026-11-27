"""Feetech STS/SCS serial bus for the Microduck's HD1910 servos. pyserial only.

Packet: FF FF ID LEN INST PARAMS.. CHK, LEN = len(params) + 2, CHK = ~(ID + LEN + INST + params) & 0xFF.
Registers and the 1 Mbit bus follow kissqy/Microduck-kissqy `duck-control/src/feetech.rs`; the ID map is its
`hardware.json` (same numbering as the official Pollen robot).
"""
import re
import threading
import time
from math import pi, radians
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:                 # on the duck itself (`studio.py --duck`): no pyserial, and no USB adapter either
    serial = list_ports = None

BAUDS = (1_000_000, 500_000, 250_000, 128_000, 115_200, 76_800, 57_600, 38_400)   # index = register 6
PING, READ, WRITE, OFSCAL, SYNC_READ, SYNC_WRITE = 1, 2, 3, 0x0B, 0x82, 0x83
ID, BAUD, KP, KD, OFFSET, LOCK, TORQUE, GOAL, KP_RAM, KD_RAM = 5, 6, 21, 22, 31, 55, 40, 42, 50, 51
BLOCK, BLOCK_LEN = 40, 31          # torque .. present current, read as one block

# servo ID -> joint. Right leg 10-14, left leg 20-24, neck / head / mouth 30-34.
DUCK = {
    10: "right_hip_yaw", 11: "right_hip_roll", 12: "right_hip_pitch", 13: "right_knee", 14: "right_ankle",
    20: "left_hip_yaw", 21: "left_hip_roll", 22: "left_hip_pitch", 23: "left_knee", 24: "left_ankle",
    30: "neck_pitch", 31: "head_pitch", 32: "head_yaw", 33: "head_roll", 34: "mouth",
}
# Position-loop gains the HD1910 replica walks with: P 5, D 20, and P 10 on the beak (kissqy console design note
# v1.0.131, "this historical Walk used P5/D20 and beakP10"). P 5 is also the firmware gain the policies here were
# trained against. The official Pollen duck (Dynamixel XL330) runs P 200, I 0, D 0 in Dynamixel units -- a different
# servo and scale, so its numbers do not carry over; only "D is set on purpose, not left at factory" does.
# 2026-11-20 (user: "make the P gain higher and D gain smaller"): P 8, D 10 on the joints (set_gains.py), P 8 being
# near the top of the policies' trained firmware-P range (4.5-8.5, microduck_rl/realism.py KP_SCALE_RANGE x P 5).
def replica_gains(sid):
    return (10, 20) if DUCK.get(sid) == "mouth" else (8, 10)


# Joint travel, read from the MuJoCo model the policies train on (the URDF's limits), so the studio cannot command
# past it. The mouth is not in the model; its travel is kissqy's `hardware.json`.
MODEL = Path(__file__).resolve().parent.parent / "app/assets/microduck/robot_groundcontact.xml"
RANGE = {n: (float(a), float(b)) for n, a, b in
         re.findall(r'name="(\w+)" type="hinge" range="(\S+) (\S+)"', MODEL.read_text())}
RANGE["mouth"] = (radians(-5), radians(30))
# Servos whose encoder counts against the model's joint axis (direction -1 in kissqy's calibrated `hardware.json`,
# same skeleton and servo). Only matters where the travel is lopsided: hip yaw, neck pitch, mouth.
FLIP = {11, 12, 13, 14, 21, 22, 23, 24, 32, 34}
ZERO = 2048                         # ticks at the joint's zero pose, once the servo has been zeroed there


def sign(sid):
    return -1 if sid in FLIP else 1


def limits(sid, zero=ZERO, direction=None):
    """(lo, hi) ticks a joint's servo may be sent to: the model's range about `zero`, the tick count at the joint's
    zero pose (from the calibration, else ZERO). `direction` is +1 or -1 (from the calibration, else `sign`).
    None if `sid` is not a joint."""
    if sid not in DUCK:
        return None
    k = (direction or sign(sid)) * 4096 / (2 * pi)
    return tuple(sorted(round(zero + k * r) for r in RANGE[DUCK[sid]]))


def angle(sid, ticks, zero, direction=None):
    """Joint angle in degrees for a servo position, given the calibrated zero and direction. The HD1910 counts on
    past one turn, so the reading is taken in the turn nearest the zero (every joint's range is within +-180 deg)."""
    return ((ticks - zero + 2048) % 4096 - 2048) * 360 / 4096 * (direction or sign(sid))


# Charge left in a lithium-ion cell against its resting voltage (a typical 18650 discharge curve). Rough by nature:
# under load the voltage sags, so a working duck reads a few percent low, and 0.1 V per servo reading is ~8 %.
CELL = ((3.0, 0), (3.3, 4), (3.4, 8), (3.5, 14), (3.6, 25), (3.7, 42), (3.8, 58), (3.9, 72), (4.0, 83), (4.1, 92), (4.2, 100))


def battery(volts, cells=2):
    """Percent of charge left in the duck's pack (2 cells in series) at `volts`."""
    v = volts / cells
    for (v0, p0), (v1, p1) in zip(CELL, CELL[1:]):
        if v <= v1:
            return max(0, round(p0 + (p1 - p0) * (v - v0) / (v1 - v0)))
    return 100


LIKELY = (0, 1, *DUCK)              # where a servo can be: factory IDs, then the joint IDs


def ports():
    """USB serial adapters, never the macOS pseudo-ports."""
    if list_ports is None:
        return []
    return [p.device for p in list_ports.comports()
            if not any(s in p.device.lower() for s in ("bluetooth", "debug", "wlan"))
            and "monitor" not in (p.description or "").lower()]          # a display's control port is no servo adapter


def _sm(v, bit):
    """Feetech sign-magnitude word -> int."""
    return -(v & ((1 << bit) - 1)) if v >> bit & 1 else v & ((1 << bit) - 1)


def _word(pos):
    """A position as the servo's two bytes: magnitude, low byte first, with bit 15 as the sign."""
    pos = max(-32767, min(32767, int(pos)))
    w = abs(pos) | (0x8000 if pos < 0 else 0)
    return w & 0xFF, w >> 8


class Bus:
    def __init__(self, port, baud=BAUDS[0], ser=None):
        """`ser`: anything with write / read(n) / timeout / reset_input_buffer / close, in place of a local port."""
        self.ser = ser or serial.Serial(port, baud, timeout=0.02, exclusive=True)
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            self.ser.close()

    def _tx(self, sid, inst, params=(), reply=0):
        """Send one packet; return the reply's data bytes, or None if nothing valid came back."""
        body = [sid, len(params) + 2, inst, *params]
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]))
            if sid == 0xFE:
                return b""
            r = self.ser.read(6 + reply)
        if len(r) != 6 + reply or r[:3] != bytes([0xFF, 0xFF, sid]) or (sum(r[2:]) & 0xFF) != 0xFF:
            return None
        return r[5:-1]

    def ping(self, sid):
        return self._tx(sid, PING) is not None

    def read(self, sid, addr, n=1):
        r = self._tx(sid, READ, [addr, n], n)
        return None if r is None else int.from_bytes(r, "little")

    def write(self, sid, addr, value, n=1):
        return self._tx(sid, WRITE, [addr, *int(value).to_bytes(n, "little")]) is not None

    def scan(self, ids=None):
        """IDs that answer a ping. Default: only where a duck servo can be -- factory 0 and 1, then the 15 joint
        IDs -- which takes ~0.1 s; if none of those answer, sweep all of 0-253 (~1.5 s). A silent ID costs 5 ms."""
        with self.lock:
            self.ser.timeout = 0.005
            try:
                found = [i for i in (ids or LIKELY) if self.ping(i) or self.ping(i)]
                if not found and ids is None:
                    found = [i for i in range(254) if self.ping(i)]
            finally:
                self.ser.timeout = 0.02
        return found

    @staticmethod
    def _decode(r):
        """Registers 40..70 -> one sample. Raw units: pos/goal ticks, vel counts (x50 ticks/s on the HD1910),
        load -1000..1000, cur counts (scale not published)."""
        return dict(torque=bool(r[0]), goal=_sm(r[2] | r[3] << 8, 15), pos=_sm(r[16] | r[17] << 8, 15),
                    vel=_sm(r[18] | r[19] << 8, 15), load=_sm(r[20] | r[21] << 8, 10), volt=r[22] / 10, temp=r[23],
                    moving=bool(r[26]), cur=_sm(r[29] | r[30] << 8, 15))

    def feedback(self, sid):
        """One servo's sample (see `_decode`), or None."""
        r = self._tx(sid, READ, [BLOCK, BLOCK_LEN], BLOCK_LEN)
        return None if r is None else self._decode(r)

    def sample(self, ids):
        """{id: sample} for every servo that answers: one Sync Read for the lot, single reads for any it missed."""
        ids, out, n = list(ids), {}, 6 + BLOCK_LEN
        if len(ids) > 1:
            body = [0xFE, len(ids) + 4, SYNC_READ, BLOCK, BLOCK_LEN, *ids]
            with self.lock:
                self.ser.reset_input_buffer()
                self.ser.write(bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]))
                buf = self.ser.read(n * len(ids))
            for i in range(0, len(buf) - n + 1, n):
                r = buf[i:i + n]
                if r[:2] == b"\xff\xff" and r[3] == BLOCK_LEN + 2 and (sum(r[2:]) & 0xFF) == 0xFF:
                    out[r[2]] = self._decode(r[5:-1])
        for sid in ids:
            if sid not in out:
                fb = self.feedback(sid)
                if fb:
                    out[sid] = fb
        return out

    def torque(self, sid, on):
        return self.write(sid, TORQUE, int(bool(on)))

    def sync_write(self, addr, values):
        """One packet that writes `addr` on many servos at once: {id: bytes}, every value the same length. Nobody
        answers a sync write, so this is as fast as a single packet however many servos there are."""
        if values:
            n = len(next(iter(values.values())))
            self._tx(0xFE, SYNC_WRITE, [addr, n, *[b for sid, v in values.items() for b in (sid, *v)]])

    def hold_all(self, ids, on):
        """Torque on (each servo holding where it is now) or off for all of `ids`, in three packets, not three per
        servo. Returns {id: position held} when switching on."""
        held = {}
        if on:
            held = {sid: fb["pos"] for sid, fb in self.sample(ids).items()}
            self.sync_write(GOAL, {sid: bytes([*_word(p), 0, 0, 0, 0]) for sid, p in held.items()})
        self.sync_write(TORQUE, {sid: bytes([1]) for sid in held} if on else {sid: bytes([0]) for sid in ids})
        return held

    def goto(self, sid, pos):
        """Goal position in ticks; the 6-byte command at 42 (position, time 0, speed 0) the robot itself sends.
        The HD1910 reports and takes sign-magnitude positions that can lie outside 0-4095, so nothing is clamped
        to one turn here: a clamp would make "hold where you are" jump a servo that reads e.g. 4798."""
        return self._tx(sid, WRITE, [GOAL, *_word(pos), 0, 0, 0, 0]) is not None

    def center(self, sid, target=2048):
        """Zero: make the servo's present pose read `target` (2048 = mid-travel, so it can swing a half turn either
        way). Torque off, then the Feetech offset-calibration instruction 0x0B; the servo works out its own encoder
        offset and stores it in EEPROM. Same sequence as the MABEL ORCA hand's re-centre. Returns the pose read back."""
        self.torque(sid, False)
        time.sleep(0.05)
        self._tx(sid, OFSCAL, [target & 0xFF, target >> 8])
        time.sleep(0.15)
        fb = self.feedback(sid)
        if fb is None:
            raise ValueError(f"{sid} stopped answering")
        if abs(fb["pos"] - target) > 120:
            # The servo keeps its offset within +-2047 and will not go past that, so a pose more than half a turn
            # from where the old offset put it cannot be reached by the instruction (seen on a real HD1910: home
            # read 1 with an offset of -2025). Offsets repeat every turn, so write the wrapped value ourselves.
            # position = encoder - offset, measured: the instruction moved both by the same amount.
            old = self.read(sid, OFFSET, 2)
            if old is None:
                raise ValueError(f"{sid} stopped answering")
            want = (_sm(old, 15) + fb["pos"] - target + 2048) % 4096 - 2048
            self._eeprom(sid, OFFSET, abs(want) | (0x8000 if want < 0 else 0), n=2)
            time.sleep(0.15)
            fb = self.feedback(sid)
            if fb is None or abs(fb["pos"] - target) > 120:
                self._eeprom(sid, OFFSET, old, n=2)          # leave the servo as it was
                raise ValueError(f"zero did not take: {sid} reads {fb and fb['pos']}")
        self.goto(sid, fb["pos"])        # the goal register is in the old frame; torque is off, so this moves nothing
        return fb["pos"]

    def _eeprom(self, sid, addr, value, after=None, n=1):
        with self.lock:
            self.write(sid, LOCK, 0)
            time.sleep(0.02)
            self.write(sid, addr, value, n)
            time.sleep(0.05)
            self.write(sid if after is None else after, LOCK, 1)

    def set_id(self, old, new):
        """Rename a servo. Refuses if `new` already answers, so two servos can never end up sharing an ID."""
        if not 0 <= new <= 253:
            raise ValueError("ID must be 0-253")
        if not self.ping(old):
            raise ValueError(f"{old} is not answering")
        if new != old and self.ping(new):
            raise ValueError(f"{new} is already on the bus")
        self._eeprom(old, ID, new, after=new)
        if not self.ping(new):
            raise ValueError("servo did not take the new ID")

    def gains(self, sid):
        """Position-loop (P, D) the servo is running: RAM registers 50 and 51. They load from EEPROM 21 and 22 at
        power-up (read on a factory HD1910: 32 / 40 in both places)."""
        p, d = self.read(sid, KP_RAM), self.read(sid, KD_RAM)
        if p is None or d is None:
            raise ValueError(f"{sid} is not answering")
        return p, d

    def set_gains(self, sid, p, d):
        """Write P and D (0-254) to EEPROM, so they survive a power cycle, and to RAM, so they act now; then read
        both back. Raises if the servo did not keep them."""
        if not (0 <= p <= 254 and 0 <= d <= 254):
            raise ValueError("gains must be 0-254")
        self._eeprom(sid, KP, p)
        self._eeprom(sid, KD, d)
        self.write(sid, KP_RAM, p)
        self.write(sid, KD_RAM, d)
        got = (self.read(sid, KP), self.read(sid, KD)), self.gains(sid)
        if got != ((p, d), (p, d)):
            raise ValueError(f"gains did not take: EEPROM {got[0]}, running {got[1]}")

    def set_baud(self, sid, baud):
        self._eeprom(sid, BAUD, BAUDS.index(baud), after=0xFE)


class MockBus(Bus):
    """In-memory servos with the same API, for trying the studio with nothing plugged in."""

    def __init__(self, ids=(1, *[i for i in DUCK if i != 24])):
        self.lock = threading.RLock()
        self.m = {i: {ID: i, BAUD: 0, KP: 32, KD: 40, KP_RAM: 32, KD_RAM: 40, TORQUE: 0, "pos": 2048 + 37 * (i % 7), "goal": None} for i in ids}

    def close(self):
        pass

    def ping(self, sid):
        return sid in self.m

    def scan(self, ids=None):
        return [i for i in (ids or range(254)) if i in self.m]

    def read(self, sid, addr, n=1):
        return self.m[sid].get(addr, 0) if sid in self.m else None

    def write(self, sid, addr, value, n=1):
        if sid not in self.m:
            return False
        s = self.m[sid]
        s[addr] = value
        if addr == ID:
            self.m[value] = self.m.pop(sid)
        return True

    def feedback(self, sid):
        s = self.m.get(sid)
        if s is None:
            return None
        now, step = time.time(), 0
        if s[TORQUE] and s["goal"] is not None:
            step = max(-12, min(12, s["goal"] - s["pos"]))
            s["pos"] += step
        dt, s["t"] = max(1e-3, now - s.get("t", now - 0.03)), now
        return dict(torque=bool(s[TORQUE]), goal=s["goal"] or s["pos"], pos=s["pos"], vel=round(step / dt / 50),
                    load=30 * s[TORQUE] + 8 * abs(step), volt=7.9, temp=31 + sid % 5, moving=step != 0, cur=3 * abs(step))

    def sample(self, ids):
        return {i: self.feedback(i) for i in ids if i in self.m}

    def goto(self, sid, pos):
        if sid in self.m:
            self.m[sid]["goal"] = int(pos)
        return sid in self.m

    def sync_write(self, addr, values):
        for sid, v in values.items():
            if sid in self.m and addr == TORQUE:
                self.m[sid][TORQUE] = v[0]
            elif sid in self.m and addr == GOAL:
                self.m[sid]["goal"] = _sm(v[0] | v[1] << 8, 15)

    def _tx(self, sid, inst, params=(), reply=0):
        if inst == OFSCAL and sid in self.m:
            self.m[sid]["pos"], self.m[sid]["goal"] = params[0] | params[1] << 8, None

    def _eeprom(self, sid, addr, value, after=None, n=1):
        self.write(sid, addr, value)
