"""Does the run log hold what the policy saw and did? Runs the duck's own drive loop (duck_agent.Drive, the real
pilot and policy pack) against a simulated servo bus on this machine, dry and live, then reads the log back.

    ~/.venvs/microduck_live/bin/python firmware/test_run_log.py [--seconds 6]

Checks, per run: one row per tick; the logged observation through the pack gives the logged action (to float32
rounding); the logged joint angles, goals, load, current, temperature and voltage are the simulated bus's own; a
silent servo gives NaN in its columns and nothing else; a log folder that cannot be written does not stop the run.
Also times the recorder's share of a tick. Writes docs/run_log_results.json and docs/figures/run_log_test.pdf.
"""
import argparse, json, math, shutil, struct, tempfile, time
from pathlib import Path

import numpy as np

import duck_agent, duck_pilot, policy_net, run_log

HERE = Path(__file__).resolve().parent
IDS = duck_agent.Drive.ALL


class Player:
    mix = twist = now = None

    def stop(self, release=True):
        pass


class FakeDuck:
    """A bus of 15 servos that go where they are told with a first-order lag, and an IMU that sways a little.
    It is a bus, not a duck: nothing the servos do moves the IMU, so the policies' actions mean nothing here."""

    def __init__(self, silent=()):
        self.player, self.silent, self.t0, self.calls = Player(), set(silent), time.monotonic(), 0
        self.cal = dict(joints=[dict(id=i, name=n, zero_raw=2048 + 7 * k, direction=-1 if k % 2 else 1, min_rad=-2.0, max_rad=2.0)
                                for k, (i, n) in enumerate(zip(IDS, [*duck_pilot.JOINTS, "mouth"]))],
                        imu=dict(mount=[1.0, 0.0, 0.0, 0.0]))
        zero = {j["id"]: j for j in self.cal["joints"]}
        self.pos = {i: zero[i]["zero_raw"] + (duck_pilot.DEFAULT_POSE[k] if k < 14 else 0) / (zero[i]["direction"] * 2 * math.pi / 4096)
                    for k, i in enumerate(IDS)}
        self.goal, self.on = dict(self.pos), {i: 0 for i in IDS}

    def mode(self):
        return "setup"

    def calibration(self):
        return self.cal

    def imu(self):
        t = time.monotonic() - self.t0
        r, p = math.radians(0.4 * math.sin(3 * t)) / 2, math.radians(0.3 * math.sin(2 * t)) / 2
        raw = duck_agent.qmul((math.cos(p), 0, math.sin(p), 0), (math.cos(r), math.sin(r), 0, 0))
        return duck_agent.pose(raw, self.cal["imu"]["mount"], [round(1.2 * math.cos(3 * t), 2), round(0.6 * math.cos(2 * t), 2), 0.0])

    def reg(self, sid):
        """Registers 0..70 of one servo, now."""
        sm = lambda v, bit: (abs(int(v)) | (1 << bit if v < 0 else 0)) & 0xFFFF
        m = bytearray(71)
        vel = (self.goal[sid] - self.pos[sid]) * 0.2
        m[40], m[50], m[51] = self.on[sid], 8, 10
        m[56:58] = struct.pack("<H", sm(round(self.pos[sid]), 15))
        m[58:60] = struct.pack("<H", sm(round(vel), 15))
        m[60:62] = struct.pack("<H", sm(self.load(sid), 10))
        m[62], m[63], m[65] = 78, 30 + sid % 7, 0
        m[69:71] = struct.pack("<H", sm(self.current(sid), 15))
        return m

    load = lambda self, sid: (sid * 13) % 400 - 200
    current = lambda self, sid: (sid * 7) % 90 - 40

    def bus(self, tx, n, timeout):
        self.calls += 1
        inst, addr, size = tx[4], tx[5], tx[6]
        if inst == 0x83:                                       # sync write
            for i in range(7, len(tx) - 1, size + 1):
                sid, data = tx[i], tx[i + 1:i + 1 + size]
                if addr == 40:
                    self.on[sid] = data[0]
                elif addr == 42:
                    w = data[0] | data[1] << 8
                    self.goal[sid] = -(w & 0x7FFF) if w & 0x8000 else w
                    self.pos[sid] += (self.goal[sid] - self.pos[sid]) * 0.2
            return b""
        out = b""
        for sid in tx[7:-1]:                                   # sync read
            if sid in self.silent:
                continue
            body = [sid, size + 2, 0, *self.reg(sid)[addr:addr + size]]
            out += bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])
        return out


def one(live, seconds, root, silent=()):
    duck = FakeDuck(silent)
    drive = duck_agent.Drive(duck)
    duck_agent.RUNS = str(root)
    drive.start(live)
    time.sleep(1.0)
    drive.drive(1.0, 0.0, 0.3)                                  # walk forward, turning
    end = time.monotonic() + seconds - 1.0
    while time.monotonic() < end:
        drive.drive(1.0, 0.0, 0.3)
        time.sleep(0.1)
    drive.press("slower")
    time.sleep(0.2)
    info = drive.info
    drive.stop()
    time.sleep(0.3)
    return duck, info


def check(duck, info, root, nets, silent=()):
    run = run_log.load(Path(root) / info["log"])
    m, n = run.meta, len(run["t"])
    worst = max(run_log.replay(run).values())
    cal = m["calibration"]["joints"]
    k = np.array([j["direction"] * 2 * math.pi / 4096 for j in cal[:14]])
    zero = np.array([j["zero_raw"] for j in cal[:14]])
    turn = np.round((run["pos"] - zero) / 4096.0) * 4096.0                 # the turn the servo is in (Drive._run)
    jpos = (run["pos"] - turn - zero) * k - np.array(m["default_pose"])    # the joint part of obs, rebuilt from the ticks
    goal = np.clip(zero + turn + run["target"] / k, -32767, 32767)         # the goal ticks, rebuilt from the target
    ok = [i for i, s in enumerate(m["servo_ids"]) if s not in silent]
    out = dict(
        live=bool(m["live"]), ticks_logged=n, ticks_run=info["ticks"], rows_match_ticks=n == info["ticks"],
        ended_recorded=m.get("ended") is not None, hz=round(float(1 / np.diff(run["t"]).mean()), 1),
        tick_ms=round(float(run["ms"].mean()), 3),
        replay_worst_rad=worst, obs_pos_worst_rad=float(np.abs(jpos - run["obs"][:, 6:20]).max()),
        goal_worst_ticks=float(np.abs(goal - run["goal"])[run["blend"] >= 1].max()) if (run["blend"] >= 1).any() else None,
        blend_reaches_1=bool(run["blend"].max() == 1.0), walked=bool((run["state"] == m["states"].index("walk")).any()),
        key_logged=bool((run["keys"] == 1 << m["keys"].index("slower")).any()), level_end=float(run["level"][-1]),
        load_exact=bool(np.array_equal(run["load"][0, ok], [duck.load(m["servo_ids"][i]) for i in ok])),
        current_exact=bool(np.array_equal(run["current"][0, ok], [duck.current(m["servo_ids"][i]) for i in ok])),
        temp_exact=bool(np.array_equal(run["temp"][0, ok], [30 + m["servo_ids"][i] % 7 for i in ok])),
        volt=float(np.nanmedian(run["volt"])), battery_pct=float(np.nanmedian(run_log.battery(run)[0])),
        gains_p_d=m["gains_p_d"].get("10"), extras_read_pct=round(100 * float(np.isfinite(run["load"][:, ok[0]]).mean()), 1),
        silent_are_nan=bool(all(np.isnan(run[c][:, m["servo_ids"].index(s)]).all() for s in silent for c in ("load", "temp", "volt"))),
        bytes_per_s=round(m["width"] * 4 / m["dt"]))
    return out, run


def cost(n=3000):
    """Seconds of one tick spent filling and committing a row (no bus), on this machine."""
    root = tempfile.mkdtemp()
    rec = run_log.Recorder(root, dict(live=False))
    vals = [(name, np.zeros(w) if w > 1 else 0.0) for name, w in run_log.LAYOUT]
    t = time.perf_counter()
    for _ in range(n):
        for name, v in vals:
            rec.put(name, v)
        rec.commit()
    dt = (time.perf_counter() - t) / n
    rec.close()
    shutil.rmtree(root)
    return dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp())
    duck_agent.POLICIES = str(HERE / "policies.npz")
    nets = policy_net.load(duck_agent.POLICIES)
    res, runs = dict(when=time.strftime("%Y-%m-%d %H:%M"), seconds=a.seconds, row_floats=run_log.WIDTH, cases={}), {}
    for name, live, silent in (("dry run", False, ()), ("live", True, ()), ("live, the beak's servo silent", True, (34,))):
        duck, info = one(live, a.seconds, root, silent)
        res["cases"][name], runs[name] = check(duck, info, root, nets, silent)
    # a log folder that cannot be made: the run goes on without a log
    duck_agent.RUNS = "/dev/null/runs"
    d = duck_agent.Drive(FakeDuck())
    d.start(True)
    time.sleep(1.0)
    info = d.info
    d.stop()
    res["no_log_folder"] = dict(ticks=info["ticks"], log_error=info.get("log_error"), run_error=info.get("error"))
    res["recorder_ms_per_tick"] = round(cost() * 1e3, 4)
    res["hour_MB"] = round(run_log.WIDTH * 4 * 50 * 3600 / 1e6, 1)
    (HERE.parent / "docs/run_log_results.json").write_text(json.dumps(res, indent=1))
    fig = run_log.plot(runs["live"], HERE.parent / "docs/figures/run_log_test.pdf")
    shutil.rmtree(root)
    print(f"row: {run_log.WIDTH} float32 = {run_log.WIDTH * 4} B/tick, {res['hour_MB']} MB per hour; recorder {res['recorder_ms_per_tick']} ms per tick (this machine)")
    for name, c in res["cases"].items():
        print(f"\n{name}")
        for key, v in c.items():
            print(f"  {key:<20} {v}")
    print("\nno log folder:", res["no_log_folder"])
    print(fig)
    bad = [n for n, c in res["cases"].items() if not (c["rows_match_ticks"] and c["ended_recorded"] and c["replay_worst_rad"] < 1e-5
                                                      and c["obs_pos_worst_rad"] < 1e-5 and (c["goal_worst_ticks"] or 0) <= 0.5 and c["load_exact"] and c["current_exact"]
                                                      and c["temp_exact"] and c["silent_are_nan"] and c["walked"] and c["key_logged"])]
    ok = not bad and res["no_log_folder"]["ticks"] > 20 and res["no_log_folder"]["log_error"] and not res["no_log_folder"]["run_error"]
    print("PASS" if ok else f"FAIL: {bad or 'no log folder'}")
    raise SystemExit(0 if ok else 1)
