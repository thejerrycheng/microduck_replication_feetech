"""Which fall handling in the duck's pilot gets the (simulated) duck up and keeps it up? Deployment logic only: the
same policy pack (firmware/policies.npz), the same drop into each lying pose as test_getup.py, three pilot settings:

    walk first   the walk policy (VelStand) for GETUP_S = 3 s, then the stand-up policy (the pilot until 2026-11-20)
    stand-up     the stand-up policy at once (GETUP_S = 0.25 s, just past the 0.2 s fall test)
    stand-up+hold  the same, and once up the stand-up policy keeps the stand (HOLD_STANDUP) instead of the walk policy

    ~/.venvs/microduck_live/bin/python firmware/test_getup_logic.py [--trials 6] [--seconds 10]

Conditions: nominal (7.7 V, 50 Hz) and real (6.6 V pack under load, the loop at 25 Hz). Per trial: up (tilt < 20 deg,
trunk > 0.095 m, 0.5 s) and when, standing at the end, and falls counted by the pilot after the first get-up.
Writes docs/getup_logic_results.json.
"""
import argparse, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
POSES = dict(back=(0, -90), front=(0, 90), left=(-90, 0), right=(90, 0))
LOGIC = {"walk first": (3.0, False), "stand-up": (0.25, False), "stand-up+hold": (0.25, True)}


def trial(job):
    logic, cond, pose, seed, seconds = job
    import engine, mujoco, duck_pilot, policy_net
    duck_pilot.GETUP_S, duck_pilot.HOLD_STANDUP = LOGIC[logic]
    rng = np.random.default_rng(seed)
    weak, slow = cond == "real", cond == "real"
    duck = engine.Duck(vin=6.6) if weak else engine.Duck()
    if weak:
        duck.servos.vin_min = 6.2
    duck.servos.set_lag(int(rng.integers(engine.DELAY_RANGE[0], engine.DELAY_RANGE[1] + 1)))
    duck.reset()
    d, s, a = duck.data, duck.servos, duck.free_qadr
    roll, pitch = np.radians(POSES[pose]) + rng.uniform(-0.1, 0.1, 2)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, rng.uniform(-3.1, 3.1)]), "xyz")
    d.qpos[a + 2], d.qpos[a + 3:a + 7] = 0.14, q
    d.qpos[s.qpos_idx] += rng.uniform(-0.2, 0.2, size=14)
    mujoco.mj_forward(duck.model, d)
    for _ in range(300):
        d.ctrl[:] = 0
        mujoco.mj_step(duck.model, d)
    s.reset(d.qpos[s.qpos_idx].copy())
    pilot = duck_pilot.Pilot(policy_net.load(HERE / "policies.npz"))
    tilt_of = lambda z: math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (z[1] ** 2 + z[2] ** 2)))))
    step = duck_pilot.DT * (2 if slow else 1)
    up_t, up_since, falls_at_up, ok = None, None, None, False
    for n in range(round(seconds / step)):
        target = pilot.tick(d.sensordata[duck.gyro_adr:duck.gyro_adr + 3].copy(), d.qpos[a + 3:a + 7].copy(),
                            d.qpos[s.qpos_idx] + d.qpos[duck.bl_qidx], d.qvel[s.dof_idx] + d.qvel[duck.bl_didx])
        for _ in range(duck.decimation * (2 if slow else 1)):
            s.command(target)
            s.substep()
        ok = tilt_of(d.qpos[a + 3:a + 7]) < 20 and d.qpos[a + 2] > 0.095
        up_since = (up_since if up_since is not None else n) if ok else None
        if up_t is None and up_since is not None and (n - up_since) * step >= 0.5:
            up_t, falls_at_up = up_since * step, pilot.falls
    return dict(logic=logic, cond=cond, pose=pose, seed=seed, up_s=up_t, standing_end=bool(ok),
                refalls=None if up_t is None else pilot.falls - falls_at_up, role_end=pilot.role)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    jobs = [(l, c, p, k, a.seconds) for l in LOGIC for c in ("nominal", "real") for p in POSES for k in range(a.trials)]
    with Pool() as pool:
        rows = pool.map(trial, jobs)
    import policy_net
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), trials=a.trials, seconds=a.seconds, pack=sorted(policy_net.load(HERE / "policies.npz")),
               rows=rows, summary={})
    print(f"{'logic':14s} {'cond':8s} " + " ".join(f"{p:>22s}" for p in POSES))
    for l in LOGIC:
        for c in ("nominal", "real"):
            cells = []
            for p in POSES:
                R = [r for r in rows if r["logic"] == l and r["cond"] == c and r["pose"] == p]
                up = [r["up_s"] for r in R if r["up_s"] is not None]
                cell = dict(up=len(up), n=len(R), up_s=float(np.mean(up)) if up else None, standing_end=sum(r["standing_end"] for r in R),
                            refalls=sum(r["refalls"] or 0 for r in R))
                out["summary"][f"{l} | {c} | {p}"] = cell
                cells.append(f"up {cell['up']}/{cell['n']} {cell['up_s'] or 0:4.1f}s end {cell['standing_end']}/{cell['n']}")
            print(f"{l:14s} {c:8s} " + " ".join(f"{x:>22s}" for x in cells))
    (HERE.parent / "docs/getup_logic_results.json").write_text(json.dumps(out, indent=1))
