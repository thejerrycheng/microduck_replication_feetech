#!/usr/bin/env python3
"""Duck Live: the real-duck round-2 fine-tunes (4090 laptop, 2026-11-19/20) against the picks they replace, on the
app's own plant (MuJoCo + BAM HD1910 servos, the Mac app's physics; NOT the Isaac Lab grader of docs/real2 on the
laptop). A smoke test that every new default loads, runs every key and holds up -- plus one slippery-floor condition.

  ~/.venvs/microduck_live/bin/python app/bench_real2.py [--quick]

Writes docs/bench_real2_results.json and docs/figures/duck_app_real2_compare.pdf; prints a table. bench_mac's own
figures from the experiments it reuses go to a temp dir (docs/figures/duck_app_*.pdf are the Duck Live doc's).
Deterministic: fixed seeds per experiment (the same starts for both policy sets).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench_mac as B  # noqa: E402
import engine  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

B.FIG = tempfile.mkdtemp(prefix="bench_real2_")      # keep bench_mac's figures out of docs/figures

OLD = dict(walk="velstand_allcoll_hd1910_iter21495.onnx", run="run_v3_allcoll_hd1910_iter5999.onnx",
           sprint="sprint_max_allcoll_hd1910_iter7998.onnx", sitstand="sitstand_fast_hd1910_iter5499.onnx",
           jump="jump_hd1910_iter1500.onnx", roulade="roulade_head_hd1910_iter7000.onnx", recovery="standup_hd1910_iter9999.onnx")
NEW = dict(engine.DEFAULTS, walk="velstand_real2_hd1910_iter25249.onnx")      # every role at its round-2 pick
# the third set isolates the round-2 skills from the round-2 walk (which creeps at zero command, so the app never sees
# the still stand a trick waits for and every stand-up hands back to a creeping walk)
SETS = (("before (app defaults to 2026-11-19)", OLD), ("real-duck round 2", NEW),
        ("round 2, old walk 21495", dict(NEW, walk=OLD["walk"])))
SLIP_MU = 0.3          # sliding friction for the slippery condition, floor and soles alike (MuJoCo takes the max)


def set_friction(duck, mu):
    if not hasattr(duck, "_mu0"):
        duck._mu0 = duck.model.geom_friction[:, 0].copy()
    duck.model.geom_friction[:, 0] = duck._mu0 if mu is None else mu


def walk_sweep(duck, quick, mu=None):
    rng = np.random.default_rng(10)
    trials = 3 if quick else 6
    rows = []
    set_friction(duck, mu)
    for c in (0.15, 0.3, 0.4, 0.5):
        ach, fell = [], 0
        for _ in range(trials):
            B.jitter(duck, rng); B.set_level(duck, walk=c)
            r = B.rollout(duck, 6.0, {"up"}, settle=1.0)
            sel = r["t"] > r["t"][0] + 2.0
            ach.append(float(r["v"][sel, 0].mean())); fell += int(r["fell"])
        rows.append(dict(cmd=c, vx=float(np.mean(ach)), vx_min=float(np.min(ach)), vx_max=float(np.max(ach)), falls=fell, trials=trials))
    # standing still for 10 s (creep / re-fall at zero command)
    creep, fell = [], 0
    for _ in range(trials):
        B.jitter(duck, rng)
        r = B.rollout(duck, 10.0, settle=1.0)
        creep.append(float(np.abs(r["v"][:, 0]).mean())); fell += int(r["fell"])
    B.restore_levels(); set_friction(duck, None)
    return dict(walk=rows, stand=dict(abs_vx_mean=float(np.mean(creep)), falls=fell, trials=trials))


def sprint(duck, quick, mu=None):
    rng = np.random.default_rng(11)
    trials = 3 if quick else 6
    set_friction(duck, mu)
    ach, zmin, fell = [], [], 0
    for _ in range(trials):
        B.jitter(duck, rng); B.set_level(duck, run="sprint")
        r = B.rollout(duck, 6.0, {"up"}, meta=True, settle=1.0)
        sel = r["t"] > r["t"][0] + 2.0
        ach.append(float(r["v"][sel, 0].mean())); zmin.append(float(r["z"][sel].min())); fell += int(r["fell"])
    B.restore_levels(); set_friction(duck, None)
    return dict(vx=float(np.mean(ach)), z_min=float(np.min(zmin)), falls=fell, trials=trials)


def pushes(duck, quick, mu=None):
    """Standing duck, a horizontal velocity impulse in a random direction, 4 s to get back up (the walk policy at
    zero twist does it by itself)."""
    rng = np.random.default_rng(12)
    reps = 4 if quick else 8
    set_friction(duck, mu)
    rows = []
    for lvl in (1.0, 2.0, 3.0):
        fell = up = 0
        for _ in range(reps):
            B.jitter(duck, rng)
            B.rollout(duck, 1.0, settle=1.0)
            a = rng.uniform(0, 2 * math.pi)
            f0 = duck.falls
            duck.push(lvl * math.cos(a), lvl * math.sin(a))
            for _ in range(int(4.0 / duck.control_dt)):
                duck.step()
            fell += int(duck.falls > f0); up += int(duck.upright and duck.state != "fallen")
        rows.append(dict(level=lvl, fell=fell, upright_after_4s=up, repeats=reps))
    set_friction(duck, None)
    return rows


def getup(duck, quick):
    """Prone / supine / side starts; the app's recovery chain (stand-up policy) with 8 s, then 2 s standing."""
    rng = np.random.default_rng(13)
    reps = 4 if quick else 8
    out = {}
    for fam in ("prone", "supine", "side"):
        ok, ts = 0, []
        for _ in range(reps):
            B.place_fallen(duck, rng, fam)
            duck.start_behavior("recovery")
            tu = None
            for i in range(int(8.0 / duck.control_dt)):
                duck.step()
                if duck.state == "stand":
                    tu = (i + 1) * duck.control_dt; break
            r = B.rollout(duck, 2.0)
            good = tu is not None and duck.upright and not r["fell"]
            ok += int(good)
            if good:
                ts.append(tu)
        out[fam] = dict(success=ok, repeats=reps, t_mean=float(np.mean(ts)) if ts else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    duck = engine.Duck(auto_reset=False)
    res = dict(date=time.strftime("%Y-%m-%d %H:%M"), quick=a.quick, slip_mu=SLIP_MU, sets={})
    for label, pols in SETS:
        print(f"==== {label}")
        duck.load_policies(pols)
        names = dict((r, p.name) for r, p in duck.policies.items())
        assert all(names[r] == pols[r] for r in pols), names
        R = dict(policies=names)
        R["nominal"] = walk_sweep(duck, a.quick); R["slip"] = walk_sweep(duck, a.quick, SLIP_MU)
        R["sprint"] = sprint(duck, a.quick); R["sprint_slip"] = sprint(duck, a.quick, SLIP_MU)
        R["push"] = pushes(duck, a.quick); R["push_slip"] = pushes(duck, a.quick, SLIP_MU)
        R["getup"] = getup(duck, a.quick)
        o = {}
        B.exp_transitions(duck, o, True); B.exp_jump(duck, o, a.quick); B.exp_roulade(duck, o, a.quick)
        tr = o["transitions"]
        R["sitstand"] = dict(sit_settle_s=[s["t_settle"] for s in tr["sit"]], sit_fell=sum(bool(s["fell"]) for s in tr["sit"]),
                             stand_up_s=[s["t_up"] for s in tr["stand"]], standing_after=sum(s["standing"] for s in tr["stand"]), trials=len(tr["stand"]),
                             stop_falls=sum(s["falls"] for s in tr["stop"]), stop_trials=sum(s["trials"] for s in tr["stop"]),
                             handover=tr["handover"])
        R["jump"] = {k: o["jump"][k] for k in ("trials", "takeoff", "standing", "fell", "rise_mm")}
        R["roulade"] = {k: o["roulade"][k] for k in ("trials", "completed", "standing", "fell", "peak_deg")}
        res["sets"][label] = R
    path = os.path.join(REPO, "docs", "bench_real2_results.json")
    json.dump(res, open(path, "w"), indent=1)

    # ---- table ----
    labs = [l for l, _ in SETS]
    tex = []
    def row(name, f):
        cells = [f(res["sets"][l]) for l in labs]
        print(f"{name:42s} " + " | ".join(f"{c:>26s}" for c in cells))
        tex.append(name.replace("_slip", r", $\mu$\," + f"{SLIP_MU}").replace("_", " ") + " & " + " & ".join(cells) + r" \\")
    print("\n" + " " * 43 + " | ".join(f"{l[:26]:>26s}" for l in labs))
    for cond in ("nominal", "slip"):
        for i, c in enumerate((0.15, 0.3, 0.4, 0.5)):
            row(f"walk {c:.2f} m/s, {cond}: vx / falls", lambda R: f"{R[cond]['walk'][i]['vx']:+.3f} / {R[cond]['walk'][i]['falls']}/{R[cond]['walk'][i]['trials']}")
        row(f"stand 10 s, {cond}: |vx| / falls", lambda R: f"{R[cond]['stand']['abs_vx_mean']:.3f} / {R[cond]['stand']['falls']}/{R[cond]['stand']['trials']}")
    for k in ("sprint", "sprint_slip"):
        row(f"{k}: vx / min trunk z / falls", lambda R: f"{R[k]['vx']:.2f} / {R[k]['z_min']:.3f} / {R[k]['falls']}/{R[k]['trials']}")
    for k in ("push", "push_slip"):
        for i, lvl in enumerate((1.0, 2.0, 3.0)):
            row(f"{k} {lvl:.0f} m/s: fell / upright 4 s", lambda R: f"{R[k][i]['fell']}/{R[k][i]['repeats']} / {R[k][i]['upright_after_4s']}/{R[k][i]['repeats']}")
    for fam in ("prone", "supine", "side"):
        row(f"get-up {fam}: ok / mean s", lambda R: f"{R['getup'][fam]['success']}/{R['getup'][fam]['repeats']} / " + (f"{R['getup'][fam]['t_mean']:.2f}" if R['getup'][fam]['t_mean'] else "-"))
    row("sit-stand: standing after / trials", lambda R: f"{R['sitstand']['standing_after']}/{R['sitstand']['trials']}, sit falls {R['sitstand']['sit_fell']}")
    row("stops from every level: falls", lambda R: f"{R['sitstand']['stop_falls']}/{R['sitstand']['stop_trials']}")
    row("jump: take-off / standing / rise mm", lambda R: f"{R['jump']['takeoff']}/{R['jump']['trials']} / {R['jump']['standing']}/{R['jump']['trials']} / {R['jump']['rise_mm']['mean']:.0f}")
    row("flip: completed / standing / stand-up needed", lambda R: f"{R['roulade']['completed']}/{R['roulade']['trials']} / {R['roulade']['standing']}/{R['roulade']['trials']} / {R['roulade']['fell']}/{R['roulade']['trials']}")
    with open(os.path.join(REPO, "docs", "duck_app_real2_table.tex"), "w") as f:
        f.write("% generated by app/bench_real2.py\n\\begin{tabular}{l" + "r" * len(labs) + "}\\toprule\n")
        f.write(" & " + " & ".join(["before", "round 2", "round 2, walk 21495"]) + r" \\\midrule" + "\n")
        f.write("\n".join(t.replace("%", r"\%") for t in tex) + "\n\\bottomrule\\end{tabular}\n")

    # ---- figure ----
    short = dict(zip(labs, ("before", "round 2", "round 2 + walk 21495")))
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9))
    cols = (B.GREY, B.BLUE, B.ORANGE)
    for (l, _), col in zip(SETS[:2], cols):          # the third set's walk IS the first set's
        R = res["sets"][l]
        for cond, ls in (("nominal", "-"), ("slip", "--")):
            c = [w["cmd"] for w in R[cond]["walk"]]; v = [w["vx"] for w in R[cond]["walk"]]
            axs[0].plot(c, v, "o" + ls, color=col, ms=4, lw=1.2, label=f"{short[l]}, {cond if cond == 'nominal' else f'mu {SLIP_MU}'}")
    axs[0].plot([0, 0.55], [0, 0.55], color=B.GREY, lw=.8, ls=":")
    axs[0].set_xlabel("commanded $v_x$ [m/s]"); axs[0].set_ylabel("achieved $v_x$ [m/s]"); axs[0].set_title("walk (VelStand)", fontsize=9); axs[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    w = 0.8 / len(SETS)
    off = lambda k: (k - 0.5 * (len(SETS) - 1)) * w
    for k, (l, _) in enumerate(SETS):
        R = res["sets"][l]
        lv = np.arange(3) + off(k)
        axs[1].bar(lv, [p["upright_after_4s"] / p["repeats"] for p in R["push"]], w * 0.9, color=cols[k], label=f"{short[l]}, nominal")
        axs[1].plot(lv, [p["upright_after_4s"] / p["repeats"] for p in R["push_slip"]], "v", color=cols[k], mec="k", ms=6, label=f"mu {SLIP_MU}" if k == 0 else None)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["1 m/s", "2 m/s", "3 m/s"]); axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("upright 4 s after the push"); axs[1].set_title("pushes while standing", fontsize=9); axs[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    cats = ["get-up\nprone", "get-up\nsupine", "get-up\nside", "sit-stand", "jump\nstanding", "flip\nstanding"]
    for k, (l, _) in enumerate(SETS):
        R = res["sets"][l]
        vals = [R["getup"][f]["success"] / R["getup"][f]["repeats"] for f in ("prone", "supine", "side")]
        vals += [R["sitstand"]["standing_after"] / R["sitstand"]["trials"], R["jump"]["standing"] / R["jump"]["trials"], R["roulade"]["standing"] / R["roulade"]["trials"]]
        axs[2].bar(np.arange(len(cats)) + off(k), vals, w * 0.9, color=cols[k], label=short[l])
    axs[2].set_xticks(range(len(cats))); axs[2].set_xticklabels(cats, fontsize=7); axs[2].set_ylim(0, 1.05)
    axs[2].set_ylabel("success fraction"); axs[2].set_title("skills (app plant, nominal floor)", fontsize=9); axs[2].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "docs", "figures", "duck_app_real2_compare.pdf")); plt.close(fig)
    print("\nwrote docs/bench_real2_results.json, docs/figures/duck_app_real2_compare.pdf")


if __name__ == "__main__":
    main()
