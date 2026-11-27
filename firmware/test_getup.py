"""Which joint's real travel stops the duck getting up? The pilot gets the simulated duck up from four lying poses,
with the real servos' limits put on one at a time.

    ~/.venvs/microduck_live/bin/python firmware/test_getup.py [--trials 8] [--seconds 20]

A real servo takes a goal of 0..4095 encoder ticks and nothing outside it, so each joint can only be *asked* for the
angles its encoder covers about the calibrated zero (firmware/calibration.json). The model in training has no such
limit on what is asked: only the joint itself stops, at its range, and the policies ask past the range to push harder.

Variants (same seeds for each):
  trained     nothing added: the simulator as the policies were trained
  encoder     every joint's target held to what its servo's encoder covers (the real duck today)
  only <id>   the same, for that one servo only (every joint whose encoder cuts into what the get-up asks)
  wrap        encoder + what the servo does past the end of its encoder: the reading jumps a full turn and the servo
              pushes on into the stop
  neck        encoder + the neck and head targets held to the model's range (duck_agent.py, NECK_PAST)
  free <id>   that joint's stop moved out by FREE each way: the model stops every joint at its range, the real joints
              were logged travelling well past it (hip pitch, knee, neck, hip roll), so a target past the range, which
              in training presses the joint on its stop, on the real duck carries the joint on
  free all    every joint's stop moved out
  6.6 V       the pack as measured under load on the real duck (6.5-6.8 V; training: 7.4-8.0 V, never under 7.0)
  25 Hz       the control loop at half rate, as on the overheated, throttled board (26-33 Hz measured; training: 50 Hz)
  6.6 V 25 Hz both
  soft stop   free all + the fix in the runner: a joint measured past the model's range has its target held to the
              range, so it is pulled back; inside the range the target is left as asked

Start poses: on its back, on its front, on its left side, on its right side; dropped limp from 2 cm, then the pilot
runs for --seconds. Up = tilt under 20 deg and trunk above 0.095 m for 0.5 s (the simulator's truth).
Writes docs/getup_results.json and docs/figures/getup_limits.pdf.
"""
import argparse, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14)            # the policies' joint order
POSES = dict(back=(0, -90), front=(0, 90), left=(-90, 0), right=(90, 0))  # roll, pitch [deg]
NECK_PAST = math.radians(3)
FREE = math.radians(40)


def encoder_span():
    """Per joint, in the policies' order: (lowest, highest) angle its servo can be asked for [rad], and the model's range."""
    cal = {j["id"]: j for j in json.loads((HERE / "calibration.json").read_text())["joints"]}
    k = 2 * math.pi / 4096
    span = np.array([sorted(((0 - cal[i]["zero_raw"]) * k * cal[i]["direction"], (4095 - cal[i]["zero_raw"]) * k * cal[i]["direction"])) for i in IDS])
    rng = np.array([(cal[i]["min_rad"], cal[i]["max_rad"]) for i in IDS])
    return span, rng, [cal[i]["name"] for i in IDS]


def trial(job):
    variant, pose, seed, seconds = job
    import engine, mujoco, duck_pilot, policy_net
    span, rng_, _ = encoder_span()
    lo, hi = np.full(14, -np.inf), np.full(14, np.inf)
    if variant in ("encoder", "wrap", "neck"):
        lo, hi = span[:, 0].copy(), span[:, 1].copy()
    elif variant.startswith("only "):
        i = IDS.index(int(variant[5:]))
        lo[i], hi[i] = span[i]
    if variant == "neck":
        lo[5:9], hi[5:9] = np.maximum(lo[5:9], rng_[5:9, 0] - NECK_PAST), np.minimum(hi[5:9], rng_[5:9, 1] + NECK_PAST)
    rng = np.random.default_rng(seed)
    weak, slow = "6.6 V" in variant, "25 Hz" in variant
    duck = engine.Duck(vin=6.6) if weak else engine.Duck()
    if weak:
        duck.servos.vin_min = 6.2
    free = list(range(14)) if variant in ("free all", "soft stop") else [IDS.index(int(variant[5:]))] if variant.startswith("free ") else []
    for i in free:
        duck.model.jnt_range[mujoco.mj_name2id(duck.model, mujoco.mjtObj.mjOBJ_JOINT, engine.JOINTS[i])] = rng_[i] + (-FREE, FREE)
    duck.servos.set_lag(int(rng.integers(engine.DELAY_RANGE[0], engine.DELAY_RANGE[1] + 1)))
    duck.reset()
    d, s, a = duck.data, duck.servos, duck.free_qadr
    roll, pitch = np.radians(POSES[pose]) + rng.uniform(-0.1, 0.1, 2)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, rng.uniform(-3.1, 3.1)]), "xyz")
    d.qpos[a + 2], d.qpos[a + 3:a + 7] = 0.14, q
    d.qpos[s.qpos_idx] += rng.uniform(-0.2, 0.2, size=14)
    mujoco.mj_forward(duck.model, d)
    for _ in range(300):                                 # dropped limp: 1.5 s to come to rest
        d.ctrl[:] = 0
        mujoco.mj_step(duck.model, d)
    s.reset(d.qpos[s.qpos_idx].copy())
    pilot = duck_pilot.Pilot(policy_net.load(HERE / "policies.npz"))
    tilt_of = lambda z: math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (z[1] ** 2 + z[2] ** 2)))))
    tilt0, z0 = tilt_of(d.qpos[a + 3:a + 7]), float(d.qpos[a + 2])
    ask, got, up_t, up_since = [], [], None, None
    step = duck_pilot.DT * (2 if slow else 1)              # simulated time per control tick
    for n in range(round(seconds / step)):
        jq = d.qpos[s.qpos_idx] + d.qpos[duck.bl_qidx]
        seen = jq.copy()
        past = (jq < span[:, 0]) | (jq > span[:, 1]) if variant == "wrap" else np.zeros(14, bool)
        seen[past] += np.where(jq[past] < span[past, 0], 2 * math.pi, -2 * math.pi)       # the encoder has gone round
        target = pilot.tick(d.sensordata[duck.gyro_adr:duck.gyro_adr + 3].copy(), d.qpos[a + 3:a + 7].copy(), seen, d.qvel[s.dof_idx] + d.qvel[duck.bl_didx])
        ask.append(target.copy())
        got.append(jq.copy())
        sent = np.clip(target, lo, hi)
        if variant == "soft stop":
            sent = np.where(jq > rng_[:, 1], np.minimum(sent, rng_[:, 1]), np.where(jq < rng_[:, 0], np.maximum(sent, rng_[:, 0]), sent))
        sent[past] = np.where(jq[past] < span[past, 0], -10.0, 10.0)                      # ... and the servo pushes on
        for _ in range(duck.decimation * (2 if slow else 1)):
            s.command(sent)
            s.substep()
        z = d.qpos[a + 3:a + 7]
        tilt = tilt_of(z)
        ok = tilt < 20 and d.qpos[a + 2] > 0.095
        up_since = (up_since if up_since is not None else n) if ok else None
        if up_since is not None and (n - up_since) * step >= 0.5:
            up_t = up_since * step
            break
    ask, got = np.array(ask), np.array(got)
    return dict(variant=variant, pose=pose, seed=seed, up_s=up_t, tilt0=round(tilt0, 1), z0=round(z0, 3), ask_lo=ask.min(0).tolist(), ask_hi=ask.max(0).tolist(),
                got_lo=got.min(0).tolist(), got_hi=got.max(0).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    span, rng, names = encoder_span()
    deg = np.degrees
    with Pool() as pool:
        base = pool.map(trial, [("trained", p, k, a.seconds) for p in POSES for k in range(a.trials)])
        # the joints whose servo cannot be asked for what the get-up asks of them
        lo = np.min([r["ask_lo"] for r in base], 0)
        hi = np.max([r["ask_hi"] for r in base], 0)
        cut = [i for i in range(14) if lo[i] < span[i, 0] or hi[i] > span[i, 1]]
        variants = ["encoder"] + [f"only {IDS[i]}" for i in cut] + ["wrap", "neck"] + [f"free {i}" for i in sorted(IDS)] + ["free all", "soft stop", "6.6 V", "25 Hz", "6.6 V 25 Hz"]
        rows = base + pool.map(trial, [(v, p, k, a.seconds) for v in variants for p in POSES for k in range(a.trials)])
    print(f"\n{'id':>3} {'joint':16} {'model range':>15} {'servo can be asked':>19} {'get-up asks':>15} {'joint reaches':>15}")
    glo, ghi = np.min([r["got_lo"] for r in base], 0), np.max([r["got_hi"] for r in base], 0)
    table = []
    for i in range(14):
        table.append(dict(id=IDS[i], joint=names[i], model_deg=deg(rng[i]).round(1).tolist(), servo_deg=deg(span[i]).round(1).tolist(),
                          asked_deg=[round(deg(lo[i]), 1), round(deg(hi[i]), 1)], reached_deg=[round(deg(glo[i]), 1), round(deg(ghi[i]), 1)], cut=i in cut))
        f = lambda p: f"{deg(p[0]):>6.0f}..{deg(p[1]):<6.0f}"
        print(f"{IDS[i]:>3} {names[i]:16} {f(rng[i]):>15} {f(span[i]):>19} {f((lo[i], hi[i])):>15} {f((glo[i], ghi[i])):>15}" + ("   <-- cut" if i in cut else ""))
    res = {}
    start = {p: [r["tilt0"] for r in base if r["pose"] == p] for p in POSES}
    print("\ntilt when the pilot starts (deg): " + "  ".join(f"{p} {min(v):.0f}..{max(v):.0f}" for p, v in start.items()))
    print(f"\n{'variant':10}" + "".join(f"{p:>16}" for p in POSES) + "    (got up / trials, median s)")
    for v in ["trained"] + variants:
        res[v] = {}
        line = f"{v:10}"
        for p in POSES:
            t = [r["up_s"] for r in rows if r["variant"] == v and r["pose"] == p]
            ok = [x for x in t if x is not None]
            res[v][p] = dict(up=len(ok), trials=len(t), median_s=round(float(np.median(ok)), 2) if ok else None)
            line += f"{len(ok):>7}/{len(t)} {res[v][p]['median_s'] if ok else '-':>6}"
        print(line)
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), trials=a.trials, seconds=a.seconds, calibration=json.loads((HERE / "calibration.json").read_text())["updated"],
               joints=table, start_tilt_deg={p: [min(v), max(v)] for p, v in start.items()}, got_up=res)
    (HERE.parent / "docs/getup_results.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(14)
    ax.barh(y, deg(rng[:, 1] - rng[:, 0]), left=deg(rng[:, 0]), height=0.62, color="#c9ccd3", label="model range")
    ax.barh(y, deg(hi - lo), left=deg(lo), height=0.26, color="#1a73e8", label="asked while getting up")
    for i in range(14):
        for e in span[i]:
            if abs(deg(e)) < 215:
                ax.plot([deg(e)] * 2, [i - 0.42, i + 0.42], color="#d93025", lw=1.6, label="end of the servo's encoder" if i == 12 and e == span[i, 0] else None)
    ax.set_yticks(y, [f"{IDS[i]} {names[i]}" for i in range(14)], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(-215, 215); ax.set_xlabel("joint angle (deg)", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    vs, w = ["trained"] + variants, 0.8 / 4
    for n, p in enumerate(POSES):
        bx.bar(np.arange(len(vs)) + (n - 1.5) * w, [100 * res[v][p]["up"] / res[v][p]["trials"] for v in vs], w - 0.02, label=p,
               color=["#1a73e8", "#34a853", "#f9ab00", "#9334e6"][n])
    bx.set_xticks(np.arange(len(vs)), vs, fontsize=7, rotation=20); bx.set_ylabel(f"got up within {a.seconds:.0f} s (% of {a.trials})", fontsize=8)
    bx.tick_params(labelsize=7); bx.legend(fontsize=7, frameon=False, title="lying on its", title_fontsize=7); bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs/figures/getup_limits.pdf")
