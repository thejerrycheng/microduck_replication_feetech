"""Does the duck's pilot (duck_pilot.py + the numpy policy pack) drive the simulated duck as Duck Live does?

    ~/.venvs/microduck_live/bin/python firmware/test_pilot.py [--trials 10]

The pilot only gets what the real duck can measure (gyro, orientation, joint angles and speeds); the simulator's
physics and servo model (app/engine.py) stand in for the duck, and the simulator's own truth (trunk height, position)
judges the result. Per trial (random start jitter and servo delay, fixed seeds): walk forward 4 s, turn 2 s, run 3 s,
sprint 2.5 s, sit, stand, walk back, sidestep, jump, flip, then a kick (the ball placed at the right toe as in training). Reported over the trials: falls, forward speed reached
against the command for walk / run / sprint, seated and standing heights, jump rise, flips landed upright, ball speed and the kick's peak tilt.
Writes docs/pilot_results.json and docs/figures/pilot_sim.pdf.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import duck_pilot, policy_net

STEPS = [("stand", 1.0, {}), ("walk", 4.0, dict(y=1)), ("turn", 2.0, dict(y=1, w=-1)), ("stop1", 1.5, {}), ("run", 3.0, dict(y=1, run=True)),
         ("stop2", 2.0, {}), ("level5", 0, "faster2"), ("sprint", 2.5, dict(y=1, run=True)), ("stop3", 2.5, {}), ("sit", 2.5, "sit"),
         ("up", 2.5, "sit"), ("back", 2.0, dict(y=-1)), ("side", 2.0, dict(x=1)), ("stop4", 1.5, {}), ("jump", 3.0, "jump"), ("flip", 4.5, "flip"), ("settle", 2.0, {}), ("kick", 3.5, "kick")]


def trial(nets, seed):
    import engine, mujoco
    rng = np.random.default_rng(seed)
    duck = engine.Duck()
    duck.servos.set_lag(int(rng.integers(engine.DELAY_RANGE[0], engine.DELAY_RANGE[1] + 1)))
    duck.reset(yaw=float(rng.uniform(-0.035, 0.035)))
    d, s = duck.data, duck.servos
    d.qpos[s.qpos_idx] += rng.uniform(-0.0175, 0.0175, size=14)
    mujoco.mj_forward(duck.model, d)
    pilot, log, marks = duck_pilot.Pilot(nets), [], {}
    for name, secs, what in STEPS:
        marks[name] = len(log)
        if what == "faster2":
            pilot.press("faster"); pilot.press("faster")
        elif isinstance(what, str):
            if what == "kick":
                duck.place_ball()                 # the operator puts the ball down; the pilot cannot see it
                ball_yaw = engine.yaw_of(d.qpos[duck.free_qadr + 3:duck.free_qadr + 7]); ball_peak = 0.0
            pilot.drive(); pilot.press(what)
        else:
            pilot.drive(**what)
        for _ in range(round(secs / duck_pilot.DT)):
            q = d.qpos[duck.free_qadr + 3:duck.free_qadr + 7]
            target = pilot.tick(d.sensordata[duck.gyro_adr:duck.gyro_adr + 3].copy(), q.copy(),
                                d.qpos[s.qpos_idx] + d.qpos[duck.bl_qidx], d.qvel[s.dof_idx] + d.qvel[duck.bl_didx])
            for _ in range(duck.decimation):
                s.command(target)
                s.substep()
            v = engine.quat_rotate_inverse(q, d.qvel[duck.free_dadr:duck.free_dadr + 3])
            log.append((float(v[0]), float(d.qpos[duck.free_qadr + 2]), pilot.tilt, pilot.state))
            if name == "kick":
                bv = d.qvel[duck.ball_dadr:duck.ball_dadr + 2]
                ball_peak = max(ball_peak, float(bv[0] * math.cos(ball_yaw) + bv[1] * math.sin(ball_yaw)))
    L, m = np.array([r[:3] for r in log]), marks
    seg = lambda a, b=None, skip=1.0: L[m[a] + round(skip / duck_pilot.DT):(m[b] if b else len(L))]
    return dict(falls=pilot.falls, walk_v=float(seg("walk", "turn")[:, 0].mean()), run_v=float(seg("run", "stop2")[:, 0].mean()),
                sprint_v=float(seg("sprint", "stop3")[:, 0].mean()), seated_z=float(seg("sit", "up", 1.5)[:, 1].mean()),
                stand_z=float(seg("stop4", "jump", 0.5)[:, 1].mean()), jump_rise=float(seg("jump", "flip", 0)[:, 1].max() - L[m["jump"], 1]),
                flip_peak_tilt=float(seg("flip", "settle", 0)[:, 2].max()), flip_upright=bool(L[m["settle"] - 1, 2] < 20 and L[m["settle"] - 1, 1] > 0.08),
                kick_ball=ball_peak, kick_peak_tilt=float(seg("kick", None, 0)[:, 2].max()), kick_upright=bool(L[-1, 2] < 20 and L[-1, 1] > 0.08),
                end_state=log[-1][3]), L, marks


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    nets = policy_net.load(HERE / "policies.npz")
    rows, first = [], None
    for k in range(a.trials):
        r, L, marks = trial(nets, k)
        first = first or (L, marks)
        rows.append(r)
        print(k, {n: (round(v, 3) if isinstance(v, float) else v) for n, v in r.items()}, flush=True)
    stat = lambda n: dict(mean=round(float(np.mean([r[n] for r in rows])), 3), min=round(float(np.min([r[n] for r in rows])), 3),
                          max=round(float(np.max([r[n] for r in rows])), 3))
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), trials=a.trials, falls_total=int(sum(r["falls"] for r in rows)),
               clean_trials=int(sum(r["falls"] == 0 for r in rows)), flips_upright=int(sum(r["flip_upright"] for r in rows)),
               kicks_upright=int(sum(r["kick_upright"] for r in rows)),
               commanded=dict(walk_v=0.40, run_v=0.70, sprint_v=1.2),
               **{n: stat(n) for n in ("walk_v", "run_v", "sprint_v", "seated_z", "stand_z", "jump_rise", "flip_peak_tilt", "kick_ball", "kick_peak_tilt")})
    (HERE.parent / "docs/pilot_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, marks = first
    t = np.arange(len(L)) * duck_pilot.DT
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.2), sharex=True)
    for a_, col, lab in zip(ax, range(3), ("forward speed (m/s)", "trunk height (m)", "tilt (deg)")):
        a_.plot(t, L[:, col], color="#1a73e8", lw=1)
        a_.set_ylabel(lab, fontsize=8)
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=7)
        for n, i in marks.items():
            a_.axvline(i * duck_pilot.DT, color="#000", alpha=0.12, lw=0.6)
    for n, i in marks.items():
        ax[0].text(i * duck_pilot.DT + 0.05, ax[0].get_ylim()[1], n, fontsize=6, rotation=90, va="top")
    ax[2].set_xlabel("time (s), trial 0", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs/figures/pilot_sim.pdf")
