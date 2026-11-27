"""Do the duck's emotion clips move the whole body, as in Duck Live, once their legs are played (duck_pilot.ClipBody)?

    ~/.venvs/microduck_live/bin/python firmware/test_clip_body.py [--trials 4]

The duck's pilot (duck_pilot.py + the numpy policy pack) on the simulated duck (app/engine.py plant), standing; then
every clip of the duck's animation pack (anim_pack.pack(), what the duck plays) is played the way duck_agent.Player
plays it while the policies drive: the head mixed onto the policy's target with the same ease-in/out, and -- the
change of 2026-11-20 -- the clip's legs through ClipBody (its walk command and its sit / jump / flip, the timeline
held while a skill runs). Two modes per clip: "head only" (the duck until 2026-11-20) and "whole body" (now).
Per trial: the trunk's path length and net yaw change over the clip, its highest point, the states the pilot passed
through, falls, whether the clip ran to its end (not cut by a fall). Writes docs/clip_body_results.json and
docs/figures/clip_body.pdf.
"""
import argparse, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import anim_pack, duck_pilot, policy_net

DEFAULT_HEAD = np.array([0.3491, 0.3491, 0.0, 0.0])          # duck_agent.Player.DEFAULT (the four head joints)


def head_at(keys, ms):
    i = max(0, min(len(keys) - 2, next((n for n, k in enumerate(keys) if k[0] > ms), len(keys)) - 1))
    a, b = keys[i], keys[i + 1]
    w = min(1.0, max(0.0, (ms - a[0]) / max(1, b[0] - a[0])))
    return DEFAULT_HEAD + np.array([x + (y - x) * w for x, y in zip(a[1:5], b[1:5])]) / 1000


def trial(job):
    name, mode, seed = job
    import engine, mujoco
    clip = next(c for c in anim_pack.pack()[1] if c["name"] == name)
    rng = np.random.default_rng(seed)
    duck = engine.Duck()
    duck.servos.set_lag(int(rng.integers(engine.DELAY_RANGE[0], engine.DELAY_RANGE[1] + 1)))
    duck.reset(yaw=float(rng.uniform(-0.035, 0.035)))
    d, s, a = duck.data, duck.servos, duck.free_qadr
    d.qpos[s.qpos_idx] += rng.uniform(-0.0175, 0.0175, size=14)
    mujoco.mj_forward(duck.model, d)
    pilot = duck_pilot.Pilot(policy_net.load(HERE / "policies.npz"))
    body = duck_pilot.ClipBody(clip) if mode == "whole body" else None
    ease = lambda u: u * u * u * (u * (6 * u - 15) + 10)
    T, t, n, path, zmax, states, ended = clip["T"], -0.5 - 1.5, 0, 0.0, 0.0, [], "end"
    xy0 = xy = d.qpos[a:a + 2].copy()
    yaw0 = engine.yaw_of(d.qpos[a + 3:a + 7])
    falls0 = None
    while t < T + 0.4:
        playing = t >= -0.5                                   # 1.5 s of plain standing first
        if playing and falls0 is None:
            falls0, xy0, yaw0 = pilot.falls, d.qpos[a:a + 2].copy(), engine.yaw_of(d.qpos[a + 3:a + 7])
            xy = xy0
        if body is not None and playing:
            for key in body.step(t, duck_pilot.DT, pilot.state, pilot.still()):
                pilot.press(key)
            pilot.twist = body.twist
            if body.done:
                ended = "fell" if "fell" in body.note else body.note
                break
        q = d.qpos[a + 3:a + 7]
        target = pilot.tick(d.sensordata[duck.gyro_adr:duck.gyro_adr + 3].copy(), q.copy(),
                            d.qpos[s.qpos_idx] + d.qpos[duck.bl_qidx], d.qvel[s.dof_idx] + d.qvel[duck.bl_didx])
        if playing and pilot.state in ("stand", "walk", "stopping", "seated", "run", "sprint"):
            w = ease((t + 0.5) / 0.5) if t < 0 else 1.0 if t <= T else 1.0 - ease((t - T) / 0.4)
            if body is not None and body.waiting:
                w = 0.0
            target = target.copy()
            target[5:9] = (1 - w) * target[5:9] + w * head_at(clip["k"], min(T, max(0.0, t)) * 1000)
        for _ in range(duck.decimation):
            s.command(target)
            s.substep()
        if playing:
            p = d.qpos[a:a + 2].copy()
            path += float(np.linalg.norm(p - xy))
            xy = p
            zmax = max(zmax, float(d.qpos[a + 2]))
            if not states or states[-1] != pilot.state:
                states.append(pilot.state)
        if not (body is not None and body.waiting):
            t += duck_pilot.DT
        n += 1
        if n > 4000:
            ended = "timeout"
            break
    dyaw = math.degrees((engine.yaw_of(d.qpos[a + 3:a + 7]) - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return dict(clip=name, mode=mode, seed=seed, path_m=path, net_m=float(np.linalg.norm(d.qpos[a:a + 2] - xy0)), yaw_deg=dyaw,
                zmax_m=zmax, states=states, falls=pilot.falls - (falls0 or 0), ended=ended, end_state=pilot.state,
                tilt_end=pilot.tilt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="every clip, not only the ones with legs plus two head-only controls")
    a = ap.parse_args()
    clips = anim_pack.pack()[1]
    legs = [c["name"] for c in clips if c.get("tw") or c.get("sk")]
    names = [c["name"] for c in clips] if a.all else legs + ["hello", "yes"]
    jobs = [(c, m, k) for c in names for m in ("head only", "whole body") for k in range(a.trials)]
    with Pool() as pool:
        rows = pool.map(trial, jobs)
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), trials=a.trials, pack=sorted(policy_net.load(HERE / "policies.npz")), rows=rows, summary={})
    print(f"{'clip':12s} {'mode':11s} {'path m':>7s} {'yaw deg':>8s} {'z max':>6s} {'falls':>5s} {'ran to end':>10s}  states (trial 0)")
    for c in names:
        for m in ("head only", "whole body"):
            R = [r for r in rows if r["clip"] == c and r["mode"] == m]
            S = dict(path_m=float(np.mean([r["path_m"] for r in R])), yaw_abs_deg=float(np.mean([abs(r["yaw_deg"]) for r in R])),
                     zmax_m=float(np.max([r["zmax_m"] for r in R])), falls=int(sum(r["falls"] for r in R)),
                     ran_to_end=int(sum(r["ended"] == "end" for r in R)), n=len(R), states=R[0]["states"])
            out["summary"][f"{c} | {m}"] = S
            print(f"{c:12s} {m:11s} {S['path_m']:7.3f} {S['yaw_abs_deg']:8.1f} {S['zmax_m']:6.3f} {S['falls']:5d} {S['ran_to_end']:>5d}/{S['n']}"
                  f"  {' > '.join(S['states'])}")
    (HERE.parent / "docs/clip_body_results.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(len(names))
    for k, (m, col) in enumerate((("head only", "#999999"), ("whole body", "#1f77b4"))):
        P = [[r["path_m"] for r in rows if r["clip"] == c and r["mode"] == m] for c in names]
        Y = [[abs(r["yaw_deg"]) for r in rows if r["clip"] == c and r["mode"] == m] for c in names]
        axs[0].bar(x + (k - 0.5) * 0.4, [np.mean(p) for p in P], 0.4, yerr=[np.std(p) for p in P], color=col, label=m)
        axs[1].bar(x + (k - 0.5) * 0.4, [np.mean(y) for y in Y], 0.4, yerr=[np.std(y) for y in Y], color=col, label=m)
    for ax, lab in zip(axs, ("trunk path over the clip [m]", "|net yaw change| [deg]")):
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_ylabel(lab); ax.legend(fontsize=8)
    axs[0].set_title(f"emotion clips on the duck's pilot (sim, {a.trials} starts each)", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE.parent / "docs/figures/clip_body.pdf"); plt.close(fig)
    print("wrote docs/clip_body_results.json, docs/figures/clip_body.pdf")
