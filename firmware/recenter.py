"""Move every servo's calibrated zero to the middle of its encoder (tick 2048), on the duck, through duck_agent's bus.

    python3 recenter.py            # dry run: prints what it would do
    python3 recenter.py --apply

Why (2026-11-20): a joint whose zero sat near an end of the 0-4095 encoder wrapped inside or just outside its range
(left hip yaw: zero 576, wrapped at +50.6 deg; right knee: zero 3080, wrapped at -89.2 deg inside its +-90; head yaw:
167.8 deg inside +-170). Twisted past it, the reading jumps a full turn: the policies saw the left hip yaw at -196.6
deg and its servo drove the long way round into the stop at full load. With the zero at 2048 the wrap is 180 deg from
the zero on every joint, outside every model range.

How: the servo reports position = encoder - offset (register 31, 2 bytes, sign-magnitude, bit 15). Adding
d = zero_raw - 2048 to the offset moves every reading by -d (mod 4096), and zero_raw becomes 2048: the joint angle
(ticks - zero_raw) * direction is unchanged for every pose. Torque is switched off first; the pilot must be off.
The calibration file is backed up beside itself (.bak-<time>) and replaced atomically.
"""
import json, os, sys, time, urllib.request

AGENT = "http://127.0.0.1:8771"
CAL = "/data/calibration.json" if os.path.exists("/data/calibration.json") else os.path.expanduser("~/calibration.json")
MID = 2048


def bus(pkt, n):
    r = urllib.request.Request(f"{AGENT}/bus/{n}/30", data=pkt, method="POST")
    return urllib.request.urlopen(r, timeout=2).read()


def packet(body):
    return bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])


def write(sid, addr, data):
    bus(packet([sid, len(data) + 3, 3, addr, *data]), 6)


def read(sid, addr, n):
    for _ in range(3):
        r = bus(packet([sid, 4, 2, addr, n]), 6 + n)
        if len(r) == 6 + n:
            return r[5:5 + n]
    raise OSError(f"servo {sid} does not answer")


sm = lambda v: -(v & 0x7FFF) if v & 0x8000 else v
word = lambda v: [abs(v) & 0xFF, (abs(v) >> 8) & 0x7F | (0x80 if v < 0 else 0)]
wrap = lambda v: (v + 2048) % 4096 - 2048             # an offset, within +-2047 (offsets repeat every turn)


def pos(sid):
    b = read(sid, 56, 2)
    return sm(b[0] | b[1] << 8)


def main(apply):
    st = json.load(urllib.request.urlopen(AGENT + "/status", timeout=3))
    if st.get("pilot") and st["pilot"].get("live"):
        sys.exit("the policies are driving: turn them off first")
    cal = json.load(open(CAL))
    if apply:
        bak = f"{CAL}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        open(bak, "w").write(json.dumps(cal, indent=1))
        print("backup:", bak)
    bad = []
    for j in cal["joints"]:
        sid, z = j["id"], j["zero_raw"]
        d = z - MID
        o = read(sid, 31, 2)
        old = sm(o[0] | o[1] << 8)
        new = wrap(old + d)
        p0 = pos(sid)
        print(f"{sid:3d} {j['name']:16s} zero {z:5d} -> {MID}  offset {old:+5d} -> {new:+5d}  reading now {p0:5d}", end="")
        if not apply:
            print(f"  -> about {(p0 - d) % 4096}")
            continue
        write(sid, 40, [0]); time.sleep(0.02)           # torque off
        write(sid, 55, [0]); time.sleep(0.02)           # EEPROM unlock
        write(sid, 31, word(new)); time.sleep(0.05)
        write(sid, 55, [1]); time.sleep(0.15)
        o2 = read(sid, 31, 2)
        p1 = pos(sid)
        moved = (p1 - p0) % 4096
        ok = sm(o2[0] | o2[1] << 8) == new and min(abs(moved - (-d % 4096)), 4096 - abs(moved - (-d % 4096))) <= 25
        if ok:
            j["zero_raw"] = MID
        else:
            bad.append(sid)
        print(f"  -> reads {p1}  {'ok' if ok else 'DID NOT TAKE (calibration for it left as it was)'}")
    if apply:
        cal["updated"] = time.strftime("%Y-%m-%d %H:%M:%S") + " recenter.py"
        with open(CAL + ".part", "w") as f:
            f.write(json.dumps(cal, indent=1)); f.flush(); os.fsync(f.fileno())
        os.replace(CAL + ".part", CAL)
        print("calibration saved:", CAL, "| not done on:", bad or "none")


if __name__ == "__main__":
    main("--apply" in sys.argv)
