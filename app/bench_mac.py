#!/usr/bin/env python3
"""Duck Live measurements (the doc's experiments): everything docs/duck_live_app.tex reports comes from here.

  ~/.venvs/microduck_live/bin/python app/bench_mac.py [--quick] [--only step,speed,turn,transitions,jump,roulade,recovery,sequence]

Writes docs/figures/duck_app_*.pdf (vector), docs/bench_results.json and the LaTeX macro / table fragments
docs/duck_app_*.tex; prints a summary table. Deterministic: fixed seeds, the plant's per-trial randomness is the
initial-pose jitter, the servo delay draw and (recovery) the push direction.

Experiments
  step         control-tick cost on this Mac (physics + servo model vs ONNX), 1500 ticks of walking
  speed        commanded vs achieved forward speed: walk (VelStand, -0.4..0.6 m/s), run (run v3, 0.2..1.1), sprint
  turn         commanded vs achieved yaw rate, walk policy (0.25..1.0 rad/s, both signs), turning while walking
  transitions  stand height at zero twist, sit-down / stand-up time (fast sit/stand), stops from each level, handovers
  jump         space: take-off, rise, landing, hand-back over N standing starts
  roulade      R: rotation reached, landed standing, time to upright over N standing starts
  recovery     pushes (velocity impulses, 0.5-3.0 m/s) while standing / walking / running: fell, back up within 2 s,
               with the walk getting up by itself vs the stand-up policy; get-up from prone / supine / side poses
  sequence     the keyboard scenario (walk, run, stop, sit, stand, turn, back, sidestep, jump, flip, sprint) over N starts
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import engine  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

FIG = os.path.join(REPO, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25})
BLUE, PURPLE, ORANGE, GREEN, RED, GREY, PINK = "#0a84ff", "#bf5af2", "#ff9f0a", "#30d158", "#ff453a", "#8e8e93", "#ff6482"


def jitter(duck: engine.Duck, rng: np.random.Generator, seated=False):
    """A fresh duck with a small random pose perturbation (+-1 deg servo angles, +-2 deg yaw) and a random servo lag."""
    duck.servos.set_lag(int(rng.integers(engine.DELAY_RANGE[0], engine.DELAY_RANGE[1] + 1)))
    duck.set_keys(set()); duck.set_perturb(-1)
    duck.reset(seated=seated, yaw=float(rng.uniform(-0.035, 0.035)))
    duck.falls = 0
    d = duck.data
    d.qpos[duck.servos.qpos_idx] += rng.uniform(-0.0175, 0.0175, size=14)
    mujoco.mj_forward(duck.model, d)


def rollout(duck: engine.Duck, secs: float, held=(), meta=False, alt=False, press=None, settle=0.0, stop_when=None):
    """Hold keys for ``secs`` (after ``settle`` s with nothing held); returns per-tick arrays."""
    if settle > 0:
        duck.set_keys(set())          # the previous trial's keys must not run through the settling second
    for _ in range(int(settle / duck.control_dt)):
        duck.step()
    duck.set_keys(set(held), meta, alt)
    if press:
        duck.press(press)
    n = int(secs / duck.control_dt)
    f0 = duck.falls
    T = np.zeros(n); V = np.zeros((n, 3)); W = np.zeros(n); Z = np.zeros(n); TILT = np.zeros(n); CMD = np.zeros((n, 3)); ST = []
    for i in range(n):
        duck.step()
        T[i] = (i + 1) * duck.control_dt; V[i] = duck.v_body; W[i] = duck.wz; Z[i] = duck.z; TILT[i] = duck.tilt_deg; CMD[i] = duck.cmd[:3]
        ST.append(duck.state)
        if stop_when is not None and stop_when(duck):
            n = i + 1
            break
    return dict(t=T[:n], v=V[:n], wz=W[:n], z=Z[:n], tilt=TILT[:n], cmd=CMD[:n], states=ST, fell=duck.falls > f0)


def set_level(duck, **kw):
    """Override the current speed level for a trial (the key layer quantises to levels)."""
    lv = dict(walk=0.4, run=0.7, yaw=1.0, label="bench"); lv.update(kw)
    engine.LEVELS[duck.level - 1] = lv


def restore_levels():
    engine.LEVELS[:] = [dict(lv) for lv in _LEVELS0]


# ---- experiments --------------------------------------------------------------------------------------------------------
def exp_step(duck, out, quick):
    n = 400 if quick else 1500
    duck.reset(); duck.set_keys({"up"})
    for _ in range(50):
        duck.step()
    phys, pol, tot = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        duck.operate(); obs = duck.observe()
        t1 = time.perf_counter()
        a = duck.policies[duck.policy_role](obs); duck.last_action = a.astype(np.float32)
        t2 = time.perf_counter()
        tgt = engine.DEFAULT_POSE + a
        for _ in range(duck.decimation):
            duck.servos.command(tgt); duck.servos.substep()
        duck.measure()
        t3 = time.perf_counter()
        pol.append((t2 - t1) * 1e3); phys.append((t3 - t2) * 1e3); tot.append((t3 - t0) * 1e3)
    pol, phys, tot = map(np.array, (pol, phys, tot))
    out["step"] = dict(n=n, total_ms=dict(median=float(np.median(tot)), p95=float(np.percentile(tot, 95)), max=float(tot.max())),
                       physics_ms=dict(median=float(np.median(phys)), p95=float(np.percentile(phys, 95))),
                       policy_ms=dict(median=float(np.median(pol)), p95=float(np.percentile(pol, 95))),
                       budget_ms=duck.control_dt * 1e3, realtime_capacity=float(duck.control_dt * 1e3 / np.median(tot)))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.linspace(0, max(2.0, np.percentile(tot, 99.5)), 60)
    ax.hist(phys, bins, alpha=.8, color=BLUE, label="physics: 4 × (BAM servo law + mj_step)")
    ax.hist(pol, bins, alpha=.8, color=PURPLE, label="policy: ONNX 61→14")
    ax.hist(tot, bins, histtype="step", color="k", lw=1.2, label="whole control tick")
    ax.axvline(duck.control_dt * 1e3, color=RED, ls="--", lw=1, label="20 ms budget (50 Hz)")
    ax.set_xlabel("time per 50 Hz control tick [ms]"); ax.set_ylabel("ticks"); ax.set_xlim(0, bins[-1])
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_title(f"Control-tick cost on this Mac (n = {n}, median {np.median(tot):.2f} ms)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_step_time.pdf")); plt.close(fig)


def exp_speed(duck, out, quick):
    rng = np.random.default_rng(0)
    trials = 2 if quick else 4
    secs = 6.0
    walk_cmds = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    run_cmds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    res = dict(walk=[], run=[], sprint=[])
    for role, cmds in (("walk", walk_cmds), ("run", run_cmds), ("sprint", [engine.SPRINT_VX])):
        if role == "sprint" and "sprint" not in duck.policies:
            continue
        for c in cmds:
            ach, zs, fell = [], [], 0
            for k in range(trials):
                jitter(duck, rng)
                if role == "walk":
                    set_level(duck, walk=abs(c))
                elif role == "run":
                    set_level(duck, run=c)
                else:
                    set_level(duck, run="sprint")
                r = rollout(duck, secs, {"up" if c > 0 else "down"}, meta=(role != "walk"), settle=1.0)
                sel = r["t"] > r["t"][0] + 2.0        # after the slew and the gait start
                ach.append(float(r["v"][sel, 0].mean())); zs.append(float(r["z"][sel].mean())); fell += int(r["fell"])
            res[role].append(dict(cmd=c, achieved_mean=float(np.mean(ach)), achieved_min=float(np.min(ach)), achieved_max=float(np.max(ach)),
                                  z_mean=float(np.mean(zs)), falls=fell, trials=trials))
            print(f"  {role:6s} vx {c:+.2f} -> {np.mean(ach):+.3f} [{np.min(ach):+.3f},{np.max(ach):+.3f}]  z {np.mean(zs):.3f}  falls {fell}/{trials}")
    restore_levels()
    out["speed"] = res
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([-0.5, 1.3], [-0.5, 1.3], color=GREY, lw=1, ls="--", label="perfect tracking")
    names = out.get("policies", {})
    for role, col, lab in (("walk", BLUE, "walk: " + names.get("walk", "")), ("run", PURPLE, "run: " + names.get("run", ""))):
        c = np.array([r["cmd"] for r in res[role]]); m = np.array([r["achieved_mean"] for r in res[role]])
        lo = np.array([r["achieved_min"] for r in res[role]]); hi = np.array([r["achieved_max"] for r in res[role]])
        f = np.array([r["falls"] for r in res[role]]) > 0
        ax.errorbar(c, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, lw=1.2, capsize=2, label=lab.replace("_hd1910", "").replace(".onnx", ""))
        if f.any():
            ax.plot(c[f], m[f], "x", color=RED, ms=9, mew=2, label="fell in ≥1 trial")
    for r in res["sprint"]:
        ax.errorbar([r["cmd"]], [r["achieved_mean"]], yerr=[[r["achieved_mean"] - r["achieved_min"]], [r["achieved_max"] - r["achieved_mean"]]], fmt="*", color=PINK, ms=11, capsize=2,
                    label="sprint: " + names.get("sprint", "").replace("_hd1910", "").replace(".onnx", "") + " (command fixed)")
    for lv in _LEVELS0:
        ax.axvline(lv["walk"], color=BLUE, alpha=.18, lw=4)
        ax.axvline(engine.SPRINT_VX if lv["run"] == "sprint" else lv["run"], color=PURPLE, alpha=.18, lw=4)
    ax.set_xlabel("commanded forward speed $v_x$ [m/s]"); ax.set_ylabel("achieved $v_x$, mean over 4 s [m/s]")
    ax.set_title("Speed tracking on the Mac plant (bands: the app's five speed levels)", fontsize=9); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_speed_tracking.pdf")); plt.close(fig)


def exp_turn(duck, out, quick):
    rng = np.random.default_rng(1)
    trials = 2 if quick else 4
    cmds = [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]
    res = []
    for c in cmds:
        ach, fell = [], 0
        for k in range(trials):
            jitter(duck, rng)
            set_level(duck, walk=0.35, yaw=abs(c))
            r = rollout(duck, 6.0, {"left" if c > 0 else "right"}, settle=1.0)
            sel = r["t"] > r["t"][0] + 2.0
            ach.append(float(r["wz"][sel].mean())); fell += int(r["fell"])
        res.append(dict(cmd=c, achieved_mean=float(np.mean(ach)), achieved_min=float(np.min(ach)), achieved_max=float(np.max(ach)), falls=fell, trials=trials))
        print(f"  yaw {c:+.2f} -> {np.mean(ach):+.3f} [{np.min(ach):+.3f},{np.max(ach):+.3f}] falls {fell}/{trials}")
    # turning while walking (the ▲+◀ combination)
    combo = []
    for c in (0.5, 0.85):
        ach_v, ach_w, fell = [], [], 0
        for k in range(trials):
            jitter(duck, rng)
            set_level(duck, walk=0.35, yaw=c)
            r = rollout(duck, 6.0, {"up", "left"}, settle=1.0)
            sel = r["t"] > r["t"][0] + 2.0
            ach_v.append(float(r["v"][sel, 0].mean())); ach_w.append(float(r["wz"][sel].mean())); fell += int(r["fell"])
        combo.append(dict(vx_cmd=0.35, yaw_cmd=c, vx=float(np.mean(ach_v)), wz=float(np.mean(ach_w)), falls=fell, trials=trials))
        print(f"  ▲+◀ vx 0.35 yaw {c:+.2f} -> vx {np.mean(ach_v):+.3f} wz {np.mean(ach_w):+.3f} falls {fell}/{trials}")
    restore_levels()
    out["turn"] = dict(in_place=res, with_forward=combo)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    c = np.array([r["cmd"] for r in res]); m = np.array([r["achieved_mean"] for r in res])
    lo = np.array([r["achieved_min"] for r in res]); hi = np.array([r["achieved_max"] for r in res])
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color=GREY, lw=1, ls="--", label="perfect tracking")
    ax.errorbar(c, m, yerr=[m - lo, hi - m], fmt="o-", color=BLUE, ms=4, lw=1.2, capsize=2, label="turn in place (walk policy)")
    for cb in combo:
        ax.plot(cb["yaw_cmd"], cb["wz"], "s", color=ORANGE, ms=6, label="turning while walking 0.35 m/s" if cb is combo[0] else None)
    ax.set_xlabel("commanded yaw rate [rad/s]"); ax.set_ylabel("achieved yaw rate, mean over 4 s [rad/s]")
    ax.set_title("Turn-rate tracking", fontsize=9); ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_turn_tracking.pdf")); plt.close(fig)


def exp_transitions(duck, out, quick):
    rng = np.random.default_rng(2)
    trials = 3 if quick else 6
    res = dict(stand_z=[], sit=[], stand=[], stop=[], handover=[])
    sit_traces, stand_traces, stop_traces = [], [], []
    for k in range(trials):
        jitter(duck, rng)
        r0 = rollout(duck, 3.0)
        res["stand_z"].append(float(r0["z"][-100:].mean()))        # the stand = walk policy at zero twist: no squat?
        r = rollout(duck, 3.0, press="s")
        t = r["t"] - r["t"][0]
        below = np.where(r["z"] < engine.SIT_SETTLED_Z)[0]
        res["sit"].append(dict(t_settle=float(t[below[0]]) if len(below) else None, z_final=float(r["z"][-50:].mean()), tilt_final=float(r["tilt"][-50:].mean()), fell=r["fell"]))
        sit_traces.append((t, r["z"].copy()))
        r2 = rollout(duck, 4.0, press="s")
        t2 = r2["t"] - r2["t"][0]
        above = np.where(r2["z"] > engine.STAND_UP_Z)[0]
        res["stand"].append(dict(t_up=float(t2[above[0]]) if len(above) else None, z_final=float(r2["z"][-50:].mean()), tilt_final=float(r2["tilt"][-50:].mean()),
                                 fell=r2["fell"], standing=bool(r2["states"][-1] == "stand")))
        stand_traces.append((t2, r2["z"].copy()))
        print(f"  trial {k}: stand z {res['stand_z'][-1]:.3f}; sit settles {res['sit'][-1]['t_settle']} s (z {res['sit'][-1]['z_final']:.3f}); stand-up z>0.10 at {res['stand'][-1]['t_up']} s, final {r2['states'][-1]}, z {res['stand'][-1]['z_final']:.3f}")
    for li, lv in enumerate(_LEVELS0):
        for role in ("walk", "run"):
            dist, tt, fell = [], [], 0
            for k in range(trials):
                jitter(duck, rng); duck.level = li + 1
                rollout(duck, 4.0, {"up"}, meta=(role == "run"), settle=1.0)
                p0 = duck.data.qpos[duck.free_qadr:duck.free_qadr + 2].copy()
                r = rollout(duck, 4.0, set(), stop_when=lambda d: (d.state == "stand" and abs(d.v_filt[0]) < 0.03) or d.state in ("fallen", "recover"))
                p1 = duck.data.qpos[duck.free_qadr:duck.free_qadr + 2]
                dist.append(float(np.linalg.norm(p1 - p0))); tt.append(float(r["t"][-1])); fell += int(r["fell"])
                if li == len(_LEVELS0) - 1 and role == "run":
                    stop_traces.append((r["t"] - r["t"][0], r["v"][:, 0].copy()))
            cmd = (engine.SPRINT_VX if lv["run"] == "sprint" else lv["run"]) if role == "run" else lv["walk"]
            gait = ("sprint" if lv["run"] == "sprint" else "run") if role == "run" else "walk"
            res["stop"].append(dict(level=li + 1, role=gait, cmd=cmd, stop_s=float(np.mean(tt)), stop_s_max=float(np.max(tt)), dist_m=float(np.mean(dist)), dist_m_max=float(np.max(dist)), falls=fell, trials=trials))
            print(f"  stop from {gait} level {li + 1} ({cmd:.2f} m/s): {np.mean(tt):.2f} s (max {np.max(tt):.2f}), {np.mean(dist):.2f} m (max {np.max(dist):.2f}), falls {fell}/{trials}")
    duck.level = engine.DEFAULT_LEVEL
    # walk -> run -> walk and walk -> sprint -> walk handovers, hot-swapping the network with the shared last action
    hand = {}
    for name, lvl in (("run", engine.DEFAULT_LEVEL), ("sprint", len(_LEVELS0))):
        fell = 0
        for k in range(trials):
            jitter(duck, rng); duck.level = lvl
            rollout(duck, 3.0, {"up"}, settle=1.0); rollout(duck, 3.0, {"up"}, meta=True); r = rollout(duck, 3.0, {"up"})
            fell += int(r["fell"])
        hand[name] = dict(falls=fell, trials=trials, level=lvl)
        print(f"  walk->{name}->walk handover: falls {fell}/{trials}")
    duck.level = engine.DEFAULT_LEVEL
    res["handover"] = hand
    out["transitions"] = res
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for t, z in sit_traces: axs[0].plot(t, z * 1000, color=ORANGE, alpha=.7, lw=1)
    axs[0].axhline(engine.SIT_SETTLED_Z * 1000, color=GREY, ls="--", lw=.8); axs[0].set_title("S: sit down (flag 1)", fontsize=9); axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("trunk height [mm]")
    for t, z in stand_traces: axs[1].plot(t, z * 1000, color=GREEN, alpha=.7, lw=1)
    axs[1].axhline(engine.STAND_UP_Z * 1000, color=GREY, ls="--", lw=.8); axs[1].set_title("S again: stand up (flag 0)", fontsize=9); axs[1].set_xlabel("time [s]")
    for t, v in stop_traces: axs[2].plot(t, v, color=PURPLE, alpha=.7, lw=1)
    top = _LEVELS0[-1]["run"]
    axs[2].axhline(0, color=GREY, lw=.8); axs[2].set_title("release ⌘▲ from the sprint" if top == "sprint" else f"release ⌘▲ from {top:.1f} m/s", fontsize=9); axs[2].set_xlabel("time [s]"); axs[2].set_ylabel("$v_x$ [m/s]")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_transitions.pdf")); plt.close(fig)


def exp_jump(duck, out, quick):
    if "jump" not in duck.policies:
        print("  no jump export"); return
    rng = np.random.default_rng(4)
    trials = 4 if quick else 12
    rows, traces = [], []
    for k in range(trials):
        jitter(duck, rng)
        rollout(duck, 1.5)
        z0 = duck.z
        duck.press("space")
        n = int(engine.JUMP_MAX_S / duck.control_dt) + 1
        T, Z, FLAG = [], [], []
        took_off = False; rise = 0.0; air = 0; t_land = None; t_back = None; st_end = None
        for i in range(n):
            duck.step()
            T.append((i + 1) * duck.control_dt); Z.append(duck.z)
            j = duck.jump
            FLAG.append(1 if (j and j["launch"]) else 0)
            if j:
                took_off = took_off or j["airborne"]; rise = max(rise, j["rise"]); air += int(j["airborne"] and not j["landed"])
                if j["landed"] and t_land is None:
                    t_land = T[-1]
            if duck.state != "jump" and t_back is None:
                t_back = T[-1]; st_end = duck.state
            if duck.state not in ("jump",) and i > 5 and t_back is not None and T[-1] - t_back > 0.3:
                break
        r = rollout(duck, 1.0)         # one more second under the walk policy: still on its feet?
        rows.append(dict(trial=k, took_off=bool(took_off), rise_mm=float(rise * 1000), air_ms=float(air * duck.control_dt * 1000), t_land=t_land,
                         t_back=t_back, state_after=st_end, standing=bool(r["states"][-1] == "stand" and duck.upright), fell=bool(r["fell"] or st_end == "recover")))
        traces.append((np.array(T), np.array(Z) - z0, np.array(FLAG)))
        print(f"  jump {k}: take-off {took_off}, rise {rise * 1000:.0f} mm, air {air * duck.control_dt * 1000:.0f} ms, landed {t_land}, back to {st_end} at {t_back}, standing after {rows[-1]['standing']}")
    out["jump"] = dict(trials=trials, rows=rows, takeoff=int(sum(r["took_off"] for r in rows)), standing=int(sum(r["standing"] for r in rows)),
                       fell=int(sum(r["fell"] for r in rows)), rise_mm=dict(mean=float(np.mean([r["rise_mm"] for r in rows if r["took_off"]] or [0])),
                       min=float(np.min([r["rise_mm"] for r in rows if r["took_off"]] or [0])), max=float(np.max([r["rise_mm"] for r in rows if r["took_off"]] or [0]))))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for t, z, f in traces:
        ax.plot(t, z * 1000, color=PINK, alpha=.6, lw=1)
    t, z, f = traces[0]
    ax.fill_between(t, -40, 50, where=f > 0, color=ORANGE, alpha=.12, label="launch flag = 1 (trial 1)")
    ax.axhline(0, color=GREY, lw=.8); ax.set_ylim(-40, 50)
    ax.set_xlabel("time since space [s]"); ax.set_ylabel("trunk rise [mm]")
    ax.set_title(f"Space: jump policy, {trials} standing starts — take-off {out['jump']['takeoff']}/{trials}, standing after {out['jump']['standing']}/{trials}", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_jump.pdf")); plt.close(fig)


def exp_roulade(duck, out, quick):
    if "roulade" not in duck.policies:
        print("  no roulade export"); return
    rng = np.random.default_rng(5)
    trials = 4 if quick else 12
    rows, traces = [], []
    for k in range(trials):
        jitter(duck, rng)
        rollout(duck, 1.5)
        duck.press("r")
        n = int((engine.ROULADE_MAX_S + 1.0) / duck.control_dt)
        T, ANG, Z = [], [], []
        peak = 0.0; t_back = None; st_end = None; t_up = None
        for i in range(n):
            duck.step()
            T.append((i + 1) * duck.control_dt); Z.append(duck.z)
            if duck.roll:
                peak = max(peak, duck.roll["peak"]); ANG.append(math.degrees(duck.roll["angle"]))
            else:
                ANG.append(ANG[-1] if ANG else 0.0)
            if duck.state != "roulade" and t_back is None:
                t_back = T[-1]; st_end = duck.state
            if t_back is not None and t_up is None and duck.upright:
                t_up = T[-1]
            if t_back is not None and T[-1] - t_back > 1.0:
                break
        rows.append(dict(trial=k, peak_deg=float(math.degrees(peak)), t_back=t_back, state_after=st_end, t_upright=t_up,
                         standing=bool(duck.state == "stand" and duck.upright), fell=bool(st_end == "recover")))
        traces.append((np.array(T), np.array(ANG), np.array(Z)))
        print(f"  flip {k}: rolled {math.degrees(peak):.0f} deg, back to {st_end} at {t_back}, upright at {t_up}, standing {rows[-1]['standing']}")
    comp = [r for r in rows if r["peak_deg"] > 300]
    out["roulade"] = dict(trials=trials, rows=rows, completed=len(comp), standing=int(sum(r["standing"] for r in rows)), fell=int(sum(r["fell"] for r in rows)),
                          peak_deg=dict(mean=float(np.mean([r["peak_deg"] for r in rows])), min=float(np.min([r["peak_deg"] for r in rows])), max=float(np.max([r["peak_deg"] for r in rows]))),
                          t_upright=dict(mean=float(np.mean([r["t_upright"] for r in rows if r["t_upright"]] or [0])), max=float(np.max([r["t_upright"] for r in rows if r["t_upright"]] or [0]))))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for t, a, z in traces:
        axs[0].plot(t, a, color=PINK, alpha=.6, lw=1); axs[1].plot(t, z * 1000, color=PINK, alpha=.6, lw=1)
    axs[0].axhline(360, color=GREY, ls="--", lw=.8); axs[0].set_xlabel("time since R [s]"); axs[0].set_ylabel("rotation about pitch [deg]"); axs[0].set_title("Front flip: rolled angle", fontsize=9)
    axs[1].axhline(engine.UPRIGHT_Z * 1000, color=GREY, ls="--", lw=.8); axs[1].set_xlabel("time since R [s]"); axs[1].set_ylabel("trunk height [mm]")
    axs[1].set_title(f"landed standing {out['roulade']['standing']}/{trials}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_roulade.pdf")); plt.close(fig)


def place_fallen(duck, rng, family):
    """Drop the duck into a fallen family (prone = face down, supine = face up, side) with the default pose."""
    duck.set_keys(set()); duck.set_perturb(-1)
    duck.reset()
    d = duck.data
    if family == "prone":
        ax, ang = (0, 1, 0), math.pi / 2
    elif family == "supine":
        ax, ang = (0, 1, 0), -math.pi / 2
    else:
        ax, ang = (1, 0, 0), math.pi / 2 * (1 if rng.random() < 0.5 else -1)
    ang += rng.uniform(-0.17, 0.17)
    q = np.array([math.cos(ang / 2), ax[0] * math.sin(ang / 2), ax[1] * math.sin(ang / 2), ax[2] * math.sin(ang / 2)])
    d.qpos[duck.free_qadr + 2] = 0.10
    d.qpos[duck.free_qadr + 3:duck.free_qadr + 7] = q
    d.qpos[duck.servos.qpos_idx] += rng.uniform(-0.1, 0.1, size=14)
    mujoco.mj_forward(duck.model, d)
    duck.state_to("fallen")          # servos hold the default pose while it drops and settles
    for _ in range(50):
        duck.step()
    duck.falls = 0


def exp_recovery(duck, out, quick):
    rng = np.random.default_rng(6)
    repeats = 2 if quick else 4
    levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    has_standup = "recovery" in duck.policies
    conds = [("stand · walk policy gets up", dict(held=(), meta=False, self_rec=True)),
             ("walk 0.3 · walk policy gets up", dict(held=("up",), meta=False, self_rec=True))]
    if has_standup:
        conds += [("walk 0.3 · stand-up policy", dict(held=("up",), meta=False, self_rec=False)),
                  ("run 0.7 · stand-up policy", dict(held=("up",), meta=True, self_rec=True))]
    push = {}
    for name, c in conds:
        rows = []
        for lvl in levels:
            fell = 0; up2 = 0; rec = 0; t_up = []
            for k in range(repeats):
                jitter(duck, rng)
                duck.walk_self_recovery = c["self_rec"]
                set_level(duck, walk=0.3, run=0.7)
                rollout(duck, 1.0, c["held"], meta=c["meta"], settle=1.0)
                ang = rng.uniform(0, 2 * math.pi)
                duck.push(lvl * math.cos(ang), lvl * math.sin(ang))
                f0 = duck.falls; tu = None
                for i in range(int(2.0 / duck.control_dt)):
                    duck.step()
                    if duck.falls > f0 and tu is None and duck.upright:     # first upright tick after the fall (scripts/push_test.py's test)
                        tu = (i + 1) * duck.control_dt
                f = duck.falls > f0
                u = duck.upright and duck.state != "fallen"
                fell += int(f); up2 += int(u); rec += int(f and u)
                if f and u and tu:
                    t_up.append(tu)
            rows.append(dict(level=lvl, fell_frac=fell / repeats, upright_after_2s_frac=up2 / repeats, recovered_after_fall_frac=(rec / fell if fell else None), fell=fell, repeats=repeats,
                             t_upright_mean=float(np.mean(t_up)) if t_up else None))
            print(f"  push {lvl:.1f} m/s, {name}: fell {fell}/{repeats}, upright after 2 s {up2}/{repeats}, recovered {rec}/{fell}" + (f", back up in {np.mean(t_up):.2f} s" if t_up else ""))
        push[name] = rows
    duck.walk_self_recovery = True
    restore_levels()
    # get-up battery: prone / supine / side, the stand-up policy vs the walk policy at zero twist
    getup = {}
    pols = [("walk policy (VelStand) at zero twist", "walk")] + ([("stand-up policy", "recovery")] if has_standup else [])
    saved = dict(duck.policies)
    for pname, role in pols:
        for fam in ("prone", "supine", "side"):
            ts, ok = [], 0
            for k in range(repeats):
                place_fallen(duck, rng, fam)
                duck.policies = dict(saved) if role == "recovery" else dict((r, p) for r, p in saved.items() if r != "recovery")
                duck.start_behavior("recovery")
                tu = None
                for i in range(int(8.0 / duck.control_dt)):
                    duck.step()
                    if duck.state == "stand":
                        tu = (i + 1) * duck.control_dt
                        break
                r = rollout(duck, 1.0)
                good = tu is not None and duck.upright and not r["fell"]
                ok += int(good)
                if good:
                    ts.append(tu - engine.RECOVERED_S)      # hand-back waits RECOVERED_S after upright
            getup[f"{pname} / {fam}"] = dict(policy=pname, family=fam, success=ok, repeats=repeats, t_up_mean=float(np.mean(ts)) if ts else None, t_up_max=float(np.max(ts)) if ts else None)
            print(f"  get-up {fam:6s} with {pname}: {ok}/{repeats}" + (f", upright in {np.mean(ts):.2f} s (max {np.max(ts):.2f})" if ts else ""))
    duck.policies = saved
    out["recovery"] = dict(levels=levels, repeats=repeats, push=push, getup=getup)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    cols = [BLUE, GREEN, ORANGE, PURPLE]
    for (name, rows), col in zip(push.items(), cols):
        axs[0].plot([r["level"] for r in rows], [r["fell_frac"] for r in rows], "o--", color=col, ms=4, lw=1, alpha=.6)
        axs[0].plot([r["level"] for r in rows], [r["upright_after_2s_frac"] for r in rows], "s-", color=col, ms=4, lw=1.3, label=name)
    axs[0].set_xlabel("push impulse [m/s]"); axs[0].set_ylabel("fraction of pushes"); axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_title("dashed: fell · solid: upright 2 s after the push", fontsize=9); axs[0].legend(fontsize=6.5, loc="lower left")
    fams = ("prone", "supine", "side")
    for pi, (pname, _) in enumerate(pols):
        x = np.arange(len(fams)) + (pi - 0.5 * (len(pols) - 1)) * 0.35
        y = [getup[f"{pname} / {f}"]["t_up_mean"] or 0 for f in fams]
        s = [getup[f"{pname} / {f}"]["success"] for f in fams]
        axs[1].bar(x, y, 0.33, color=[BLUE, ORANGE][pi], label=pname)
        for xi, yi, si in zip(x, y, s):
            axs[1].text(xi, yi + 0.05, f"{si}/{repeats}", ha="center", fontsize=7)
    axs[1].set_xticks(np.arange(len(fams))); axs[1].set_xticklabels(fams); axs[1].set_ylabel("time to upright [s]"); axs[1].set_title("get-up from the floor (label: successes)", fontsize=9)
    axs[1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_app_recovery.pdf")); plt.close(fig)


def exp_sequence(duck, out, quick):
    rng = np.random.default_rng(3)
    trials = 3 if quick else 8
    SEQ = [("stand", 1.5, (), False, False, None), ("walk ▲", 4.0, ("up",), False, False, None), ("run ⌘▲", 4.0, ("up",), True, False, None),
           ("release", 2.0, (), False, False, None), ("sit S", 2.0, (), False, False, "s"), ("stand S", 2.0, (), False, False, "s"),
           ("turn ◀", 4.0, ("left",), False, False, None), ("▲+▶", 4.0, ("up", "right"), False, False, None), ("back ▼", 3.0, ("down",), False, False, None),
           ("sidestep ⌥◀", 3.0, ("left",), False, True, None), ("release", 2.0, (), False, False, None),
           ("jump space", 3.0, (), False, False, "space"), ("flip R", 4.0, (), False, False, "r"),
           ("sprint ⌘▲ (D D)", 4.0, ("up",), True, False, None), ("release", 2.5, (), False, False, None)]
    rows = []
    for k in range(trials):
        jitter(duck, rng)
        segs = []
        for label, secs, held, meta, alt, press in SEQ:
            if label.startswith("sprint"):
                duck.level = len(_LEVELS0)
            r = rollout(duck, secs, held, meta, alt, press)
            segs.append(dict(label=label, vx=float(r["v"][len(r["v"]) // 2:, 0].mean()), wz=float(r["wz"][len(r["wz"]) // 2:].mean()),
                             z=float(r["z"][-25:].mean()), state=r["states"][-1], falls=duck.falls))
        duck.level = engine.DEFAULT_LEVEL
        rows.append(dict(trial=k, falls=duck.falls, dist=duck.odometer, segs=segs))
        print(f"  trial {k}: falls {duck.falls}, {duck.odometer:.2f} m, end state {segs[-1]['state']}; " + " · ".join(f"{s['label']} {s['state']}" for s in segs if s['state'] not in ('walk', 'stand', 'run', 'sprint', 'seated')))
    out["sequence"] = dict(trials=trials, falls_total=int(sum(r["falls"] for r in rows)), clean_trials=int(sum(r["falls"] == 0 for r in rows)),
                           segments=[s[0] for s in SEQ], rows=rows)


_LEVELS0 = [dict(lv) for lv in engine.LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="step,speed,turn,transitions,jump,roulade,recovery,sequence")
    ap.add_argument("--tex-only", action="store_true", help="regenerate the LaTeX fragments from docs/bench_results.json")
    a = ap.parse_args()
    path = os.path.join(REPO, "docs", "bench_results.json")
    out = json.load(open(path)) if os.path.exists(path) else {}      # a partial run keeps the other sections
    if a.tex_only:
        write_tex(out); print("wrote docs/duck_app_numbers.tex and the tables"); return
    duck = engine.Duck(auto_reset=False)      # a fall is handled by the policies; it must never be hidden by a reset
    out.update(date=time.strftime("%Y-%m-%d %H:%M"), policies=dict((r, p.name) for r, p in duck.policies.items()),
               plant=dict(physics_dt=float(duck.model.opt.timestep), decimation=duck.decimation, vin=duck.servos.vin0, sag_gain=duck.servos.sag_gain,
                          delay_steps=list(engine.DELAY_RANGE), kp_fw=engine.KP_FW, force_limit_Nm=duck.servos.force_limit, mass_kg=duck.total_mass),
               levels=_LEVELS0, quick=a.quick)
    for name in a.only.split(","):
        print(f"== {name}")
        t0 = time.time()
        globals()["exp_" + name](duck, out, a.quick)
        print(f"   ({time.time() - t0:.0f} s)")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    write_tex(out)
    print("\nwrote docs/bench_results.json, docs/duck_app_numbers.tex, docs/duck_app_*_table.tex and docs/figures/duck_app_*.pdf")


def write_tex(out):
    """The doc's numbers as LaTeX macros + tables, so docs/duck_live_app.tex always shows what was measured."""
    D = os.path.join(REPO, "docs")
    M = []
    def mac(name, val):
        M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    def short(n):
        return n.replace("_hd1910", "").replace(".onnx", "").replace("_", "\\_")
    mac("benchDate", out.get("date", ""))
    mac("plantMass", f"{out['plant']['mass_kg']:.3f}")
    for role in ("walk", "run", "sprint", "sitstand", "jump", "roulade", "recovery"):
        mac("pol" + role.capitalize(), short(out.get("policies", {}).get(role, "--")))
    if "step" in out:
        s = out["step"]
        mac("stepN", s["n"]); mac("stepMedian", f"{s['total_ms']['median']:.2f}"); mac("stepPninetyfive", f"{s['total_ms']['p95']:.2f}")
        mac("stepMax", f"{s['total_ms']['max']:.2f}"); mac("stepPhysics", f"{s['physics_ms']['median']:.2f}"); mac("stepPolicy", f"{s['policy_ms']['median']:.3f}")
        mac("stepCapacity", f"{s['realtime_capacity']:.0f}")
    if "speed" in out:
        rows = []
        for role in ("walk", "run", "sprint"):
            for r in out["speed"].get(role, []):
                rows.append(f"{role} & {r['cmd']:+.2f} & {r['achieved_mean']:+.3f} & [{r['achieved_min']:+.3f}, {r['achieved_max']:+.3f}] & {r['z_mean'] * 1000:.0f} & {r['falls']}/{r['trials']} \\\\")
        with open(os.path.join(D, "duck_app_speed_table.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrrr}\\toprule\npolicy & $v_x$ cmd [m/s] & achieved mean [m/s] & range over trials & trunk $z$ [mm] & falls \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
        w = [r for r in out["speed"]["walk"] if abs(r["cmd"]) >= 0.2]
        mac("walkMaxErr", f"{max(abs(r['achieved_mean'] - r['cmd']) for r in w):.3f}")
        run_ok = [r for r in out["speed"]["run"] if r["falls"] == 0 and r["achieved_mean"] > 0.25]
        mac("runTopCmd", f"{max(r['cmd'] for r in run_ok):.1f}"); mac("runTopAch", f"{max(r['achieved_mean'] for r in run_ok):.2f}")
        run_falls = [r for r in out["speed"]["run"] if r["falls"] > 0]
        mac("runFallCmds", ", ".join(f"{r['cmd']:.1f}" for r in run_falls) if run_falls else "none")
        sp = out["speed"].get("sprint", [])
        mac("sprintAch", f"{sp[0]['achieved_mean']:.2f}" if sp else "--"); mac("sprintMin", f"{sp[0]['achieved_min']:.2f}" if sp else "--"); mac("sprintMax", f"{sp[0]['achieved_max']:.2f}" if sp else "--")
        mac("sprintFalls", f"{sp[0]['falls']}/{sp[0]['trials']}" if sp else "--")
        trials = out["speed"]["walk"][0]["trials"]; mac("speedTrials", trials)
    if "turn" in out:
        t = out["turn"]["in_place"]
        mac("turnTrials", t[0]["trials"])
        good = [r for r in t if abs(r["achieved_mean"]) > 0.3]
        mac("turnMinCmd", f"{min(abs(r['cmd']) for r in good):.2f}" if good else "--")
        mac("turnMaxErr", f"{max(abs(r['achieved_mean'] - r['cmd']) for r in t):.2f}")
        rows = [f"{r['cmd']:+.2f} & {r['achieved_mean']:+.3f} & [{r['achieved_min']:+.3f}, {r['achieved_max']:+.3f}] & {r['falls']}/{r['trials']} \\\\" for r in t]
        rows += [f"\\multicolumn{{4}}{{l}}{{\\emph{{while walking {c['vx_cmd']:.2f} m/s}}: yaw {c['yaw_cmd']:+.2f} $\\to$ {c['wz']:+.3f} rad/s, $v_x$ {c['vx']:+.3f} m/s, falls {c['falls']}/{c['trials']}}} \\\\" for c in out["turn"]["with_forward"]]
        with open(os.path.join(D, "duck_app_turn_table.tex"), "w") as f:
            f.write("\\begin{tabular}{rrrr}\\toprule\nyaw cmd [rad/s] & achieved mean [rad/s] & range over trials & falls \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    if "transitions" in out:
        tr = out["transitions"]
        st = [r["t_settle"] for r in tr["sit"] if r["t_settle"] is not None]
        up = [r["t_up"] for r in tr["stand"] if r["t_up"] is not None]
        mac("transTrials", len(tr["sit"])); mac("sitSettleMean", f"{np.mean(st):.2f}"); mac("sitSettleMin", f"{np.min(st):.2f}"); mac("sitSettleMax", f"{np.max(st):.2f}")
        mac("sitZ", f"{np.mean([r['z_final'] for r in tr['sit']]) * 1000:.0f}")
        sz = tr.get("stand_z") or [float("nan")]
        mac("standZ", f"{np.mean(sz) * 1000:.0f}"); mac("standZmin", f"{np.min(sz) * 1000:.0f}")
        mac("standUpN", len(up)); mac("standUpMean", f"{np.mean(up):.2f}" if up else "--"); mac("standUpMin", f"{np.min(up):.2f}" if up else "--"); mac("standUpMax", f"{np.max(up):.2f}" if up else "--")
        mac("standUpStanding", sum(r["standing"] for r in tr["stand"])); mac("standUpZ", f"{np.mean([r['z_final'] for r in tr['stand']]) * 1000:.0f}")
        rows = [f"{r['level']} & {r['role']} & {r['cmd']:.2f} & {r['stop_s']:.2f} ({r['stop_s_max']:.2f}) & {r['dist_m']:.2f} ({r['dist_m_max']:.2f}) & {r['falls']}/{r['trials']} \\\\" for r in tr["stop"]]
        with open(os.path.join(D, "duck_app_stop_table.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrrr}\\toprule\nlevel & gait & $v_x$ cmd [m/s] & time to stand, mean (max) [s] & distance, mean (max) [m] & falls \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
        mac("stopMaxS", f"{max(r['stop_s_max'] for r in tr['stop']):.2f}"); mac("stopMaxM", f"{max(r['dist_m_max'] for r in tr['stop']):.2f}")
        mac("stopFalls", sum(r["falls"] for r in tr["stop"])); mac("stopTrials", sum(r["trials"] for r in tr["stop"]))
        h = tr["handover"]
        mac("handoverFalls", h["run"]["falls"]); mac("handoverTrials", h["run"]["trials"])
        mac("sprintHandoverFalls", h.get("sprint", {}).get("falls", "--")); mac("sprintHandoverTrials", h.get("sprint", {}).get("trials", "--"))
    if "jump" in out:
        j = out["jump"]
        mac("jumpN", j["trials"]); mac("jumpTakeoff", j["takeoff"]); mac("jumpStanding", j["standing"]); mac("jumpFell", j["fell"])
        mac("jumpRiseMean", f"{j['rise_mm']['mean']:.0f}"); mac("jumpRiseMin", f"{j['rise_mm']['min']:.0f}"); mac("jumpRiseMax", f"{j['rise_mm']['max']:.0f}")
        tb = [r["t_back"] for r in j["rows"] if r["t_back"]]
        mac("jumpBackMean", f"{np.mean(tb):.2f}" if tb else "--"); mac("jumpBackMax", f"{np.max(tb):.2f}" if tb else "--")
        air = [r["air_ms"] for r in j["rows"] if r["took_off"]]
        mac("jumpAirMean", f"{np.mean(air):.0f}" if air else "--")
    if "roulade" in out:
        r = out["roulade"]
        mac("rollN", r["trials"]); mac("rollCompleted", r["completed"]); mac("rollStanding", r["standing"]); mac("rollFell", r["fell"])
        mac("rollDegMean", f"{r['peak_deg']['mean']:.0f}"); mac("rollDegMin", f"{r['peak_deg']['min']:.0f}"); mac("rollDegMax", f"{r['peak_deg']['max']:.0f}")
        mac("rollUpMean", f"{r['t_upright']['mean']:.2f}"); mac("rollUpMax", f"{r['t_upright']['max']:.2f}")
        tb = [x["t_back"] for x in r["rows"] if x["t_back"]]
        mac("rollBackMean", f"{np.mean(tb):.2f}" if tb else "--")
    if "recovery" in out:
        rc = out["recovery"]
        mac("pushRepeats", rc["repeats"]); mac("pushLevels", ", ".join(f"{l:.1f}" for l in rc["levels"]))
        rows = []
        for name, rs in rc["push"].items():
            cells = " & ".join(f"{int(round(r['fell_frac'] * rc['repeats']))} / {int(round(r['upright_after_2s_frac'] * rc['repeats']))}" for r in rs)
            rows.append(f"{name.replace(' · ', ', ')} & {cells} \\\\")
        with open(os.path.join(D, "duck_app_push_table.tex"), "w") as f:
            f.write("\\begin{tabular}{l" + "c" * len(rc["levels"]) + "}\\toprule\ncondition & " + " & ".join(f"{l:.1f} m/s" for l in rc["levels"]) + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
        grows = [f"{g['policy']} & {g['family']} & {g['success']}/{g['repeats']} & {g['t_up_mean']:.2f} & {g['t_up_max']:.2f} \\\\" if g["t_up_mean"] is not None else f"{g['policy']} & {g['family']} & {g['success']}/{g['repeats']} & -- & -- \\\\" for g in rc["getup"].values()]
        with open(os.path.join(D, "duck_app_getup_table.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrr}\\toprule\nget-up policy & start pose & upright & time to upright, mean [s] & max [s] \\\\\\midrule\n" + "\n".join(grows) + "\n\\bottomrule\\end{tabular}\n")
        allg = list(rc["getup"].values())
        mac("getupSuccess", sum(g["success"] for g in allg)); mac("getupTrials", sum(g["repeats"] for g in allg))
        ts = [g["t_up_max"] for g in allg if g["t_up_max"] is not None]
        mac("getupMaxS", f"{max(ts):.2f}" if ts else "--")
        # worst push level at which every condition was upright again after 2 s
        full = [l for i, l in enumerate(rc["levels"]) if all(rs[i]["upright_after_2s_frac"] >= 0.999 for rs in rc["push"].values())]
        mac("pushAllUpright", f"{max(full):.1f}" if full else "--")
        rec = [r for rs in rc["push"].values() for r in rs if r["fell"] > 0]
        mac("pushFallsTotal", sum(r["fell"] for r in rec)); mac("pushRecovered", sum(int(round(r["recovered_after_fall_frac"] * r["fell"])) for r in rec))
    if "sequence" in out:
        s = out["sequence"]
        mac("seqTrials", s["trials"]); mac("seqClean", s["clean_trials"]); mac("seqFalls", s["falls_total"])
        mac("seqDist", f"{np.mean([r['dist'] for r in s['rows']]):.2f}")
        mac("seqSegments", len(s["segments"]))
    with open(os.path.join(D, "duck_app_numbers.tex"), "w") as f:
        f.write("% generated by app/bench_mac.py -- do not edit\n" + "\n".join(M) + "\n")


if __name__ == "__main__":
    main()
