"""Set the position-loop gains (P, D) of the duck's joint servos, on the duck, through duck_agent's raw bus.

    python3 set_gains.py [P D]          # default: the values below; prints before / after for every servo

Writes EEPROM 21 / 22 (unlock 55 = 0, write, lock 55 = 1), so the gains survive a power cycle, and RAM 50 / 51, so
they act now; then reads all four back. Refuses while the policies drive (the agent's pilot is live).

Training vs deployment (2026-11-20): the policies' servo model (BAM, fitted at firmware P 5, no D term in the drive
model) is trained with each joint's P scaled 0.9-1.7 (round 2; 1.0-1.6 before; stand-up 0.8-1.7), i.e. firmware
P 4.5-8.5. The duck ran P 5, D 20 (set 2026-11-18; factory P 32, D 40). User, 2026-11-20: "make the P gain higher and
D gain smaller": P 8, near the top of what every deployed policy was trained on (P 8.5 is the highest), D 10.
The beak (34) is left at P 10.
"""
import json, sys, time, urllib.request

P, D = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (8, 10)
IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)
AGENT = "http://127.0.0.1:8771"


def bus(pkt, n):
    r = urllib.request.Request(f"{AGENT}/bus/{n}/30", data=pkt, method="POST")
    return urllib.request.urlopen(r, timeout=2).read()


def packet(body):
    return bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])


def write(sid, addr, value):
    return bus(packet([sid, 4, 3, addr, value]), 6)


def read(sid, addr):
    r = bus(packet([sid, 4, 2, addr, 1]), 7)
    return r[5] if len(r) == 7 else None


st = json.load(urllib.request.urlopen(AGENT + "/status", timeout=3))
if st.get("pilot") and st["pilot"].get("live"):
    sys.exit("the policies are driving: turn them off first")
print(f"setting P {P}, D {D}")
bad = []
for sid in IDS:
    before = (read(sid, 21), read(sid, 22), read(sid, 50), read(sid, 51))
    write(sid, 40, 0); time.sleep(0.02)            # torque off: the loop gains are never rewritten under torque
    write(sid, 55, 0); time.sleep(0.02)
    write(sid, 21, P); time.sleep(0.03)
    write(sid, 22, D); time.sleep(0.03)
    write(sid, 55, 1); time.sleep(0.02)
    write(sid, 50, P); write(sid, 51, D)
    after = (read(sid, 21), read(sid, 22), read(sid, 50), read(sid, 51))
    ok = after == (P, D, P, D)
    bad += [] if ok else [sid]
    print(f"servo {sid}: EEPROM P,D / running P,D  {before} -> {after}{'' if ok else '  DID NOT TAKE'}")
print("done" if not bad else f"not set on {bad}")
