#!/usr/bin/env python3
"""Duck Live animation measurements: everything the "Emotions and idle life" part of docs/duck_live_app.tex reports.

  ~/.venvs/microduck_live/bin/python app/bench_anim.py [--quick] [--only headtrack,clips,idle,walk,reactions,cost]

Writes docs/figures/duck_anim_*.pdf (vector), docs/bench_anim_results.json and the LaTeX fragments
docs/duck_anim_numbers.tex / docs/duck_anim_*_table.tex; prints a summary. Deterministic: fixed seeds; a trial's
randomness is the initial-pose jitter, the servo delay draw (bench_mac.jitter) and the animation seed.

Experiments
  headtrack  a constant head offset on one joint, sent two ways (the policy's head command / straight to the head
             servos), standing under VelStand and seated under sit-stand: achieved offset, trunk tilt, drift.
             This is what HEAD_MODE in animation.py is chosen from.
  clips      every emotion clip, N jittered starts (and seated, for the clips that allow it): falls, completion,
             peak tilt, drift, head tracking error, the skills fired (jump / front flip / sit) and how they ended;
             plus the tracking error of every clip under the other head mode
  idle       10 minutes of standing idle life per seed vs the bare policy: falls, drift, tilt, glances per minute,
             where the head points; 3 minutes seated (dozing)
  walk       a driving scenario (walk, curve, turn, stop) and pushes, idle life on vs off: speed error, falls, get-ups
  reactions  jump -> "yep", flip -> proud, knocked over -> shake-off, play dead -> wake -> get-up, an arrow key cutting
             a clip: does the reaction fire, how long until the duck is back
  cost       control-tick cost with the animation layer on vs off
  bodypose   why the clips' body pitch / height are not sent as the policy's body-pose command: each of the six
             slots at the two ends of its training range, standing: what the trunk actually does
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
import animation  # noqa: E402
import engine  # noqa: E402
from bench_mac import BLUE, GREEN, GREY, ORANGE, PINK, PURPLE, RED, jitter, plt  # noqa: E402  (also sets the plot style)

FIG = os.path.join(REPO, "docs", "figures")
DOCS = os.path.join(REPO, "docs")
HEAD = ("neck pitch", "head pitch", "head yaw", "head roll")


def head_meas(d):
    return d.data.qpos[d.servos.qpos_idx][5:9] - engine.DEFAULT_POSE[5:9]


def fresh(duck, rng, seed, seated=False, lively=False, react=False):
    duck.anim.seed, duck.anim.lively, duck.anim.react, duck.anim.pin, duck.anim.viewer_pos = int(seed), lively, react, None, None
    jitter(duck, rng)
    duck.walk_self_recovery = True
    for _ in range(50):
        duck.step()
    if seated:
        duck.press("s")
        for _ in range(100):
            duck.step()


# ---- head offsets: through the policy or straight to the servos ---------------------------------------------------------
def exp_headtrack(duck, out, quick):
    rng = np.random.default_rng(11)
    amps = [np.array([-0.45, -0.3, -0.15, 0.15, 0.3]), np.array([-0.6, -0.3, 0.3, 0.6, 0.9]), np.array([-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]),
            np.array([-0.4, -0.2, 0.2, 0.4])]
    reps = 2 if quick else 4
    res = {}
    mode0 = dict(animation.HEAD_MODE)
    for posture, role in (("standing", "walk"), ("seated", "sitstand")):
        for mode in ("cmd", "direct"):
            animation.HEAD_MODE[role] = mode
            rows = []
            for j in range(4):
                for a in (np.concatenate([[-0.8, -0.6], amps[j]]) if (j == 0 and posture == "seated") else amps[j]):
                    got, tilt, drift, fell = [], [], [], 0
                    for k in range(reps):
                        fresh(duck, rng, k, seated=posture == "seated")
                        p0 = duck.data.qpos[:2].copy()
                        pin = np.zeros(4); pin[j] = a
                        duck.anim.pin = pin
                        acc, mt = [], 0.0
                        for i in range(150):
                            duck.step()
                            mt = max(mt, duck.tilt_deg)
                            if i >= 75:
                                acc.append(head_meas(duck)[j])
                        got.append(float(np.mean(acc))); tilt.append(mt); drift.append(float(np.linalg.norm(duck.data.qpos[:2] - p0)))
                        fell += int(duck.falls > 0 or duck.tilt_deg > 45)
                    rows.append(dict(joint=j, want=float(a), got=float(np.mean(got)), tilt=float(np.max(tilt)), drift=float(np.mean(drift)), fell=fell, n=reps))
            e = np.array([r["got"] - r["want"] for r in rows])
            res["%s/%s" % (posture, mode)] = dict(rows=rows, rmse=float(np.sqrt(np.mean(e ** 2))), max_tilt=float(max(r["tilt"] for r in rows)),
                                                 fell=int(sum(r["fell"] for r in rows)), trials=reps * len(rows), drift=float(np.mean([r["drift"] for r in rows])))
            print("   %-8s %-6s rmse %.3f rad  max tilt %5.1f deg  mean drift %.3f m  fell %d/%d" % (posture, mode, res["%s/%s" % (posture, mode)]["rmse"],
                  res["%s/%s" % (posture, mode)]["max_tilt"], res["%s/%s" % (posture, mode)]["drift"], res["%s/%s" % (posture, mode)]["fell"], reps * len(rows)))
    animation.HEAD_MODE.update(mode0)
    duck.anim.pin = None
    out["headtrack"] = dict(res=res, reps=reps, chosen=mode0)
    fig, axs = plt.subplots(2, 4, figsize=(9.2, 4.4), sharey=False)
    for r, posture in enumerate(("standing", "seated")):
        for j in range(4):
            ax = axs[r, j]
            lim = [(-0.8 if (j == 0 and r == 1) else min(amps[j])) * 1.1, max(amps[j]) * 1.1]
            ax.plot(lim, lim, color=GREY, lw=0.8, ls="--")
            for mode, col, lab in (("cmd", ORANGE, "policy head command"), ("direct", BLUE, "head servos directly")):
                rows = [x for x in res["%s/%s" % (posture, mode)]["rows"] if x["joint"] == j]
                ax.plot([x["want"] for x in rows], [x["got"] for x in rows], "o-", color=col, ms=3.5, lw=1.2, label=lab)
                for x in rows:
                    if x["fell"]:
                        ax.plot(x["want"], x["got"], "x", color=RED, ms=8, mew=2)
            ax.set_title("%s · %s" % (HEAD[j], posture), fontsize=8.5)
            if r == 1:
                ax.set_xlabel("commanded offset [rad]")
            if j == 0:
                ax.set_ylabel("achieved [rad]")
    axs[0, 0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_anim_head_tracking.pdf")); plt.close(fig)


# ---- every clip ----------------------------------------------------------------------------------------------------------
def play_clip(duck, rng, name, seed, seated=False, lively=True):
    """One jittered start, the clip, 2.5 s after it. Returns the trial's metrics."""
    fresh(duck, rng, seed, seated=seated, lively=lively)
    c = duck.anim.lib[name]
    p0 = duck.data.qpos[:2].copy()
    duck.emote(name)
    err, mt, states, started, t_end, skills = [], 0.0, [], False, None, []
    for i in range(int((c.T + 9.0) * 50)):
        duck.step()
        a = duck.anim
        started = started or a.clip is not None
        if not states or states[-1] != duck.state:
            states.append(duck.state)
        mt = max(mt, duck.tilt_deg) if duck.state not in ("roulade", "dead", "recover") else mt
        if a.clip is not None and a.wait is None and a.w > 0.95 and duck.state in ("stand", "seated"):
            err.append(head_meas(duck) - a.head)
        if started and a.clip is None and t_end is None:
            t_end = i
        if t_end is not None and i - t_end > 125:
            break
    e = np.sqrt(np.mean(np.square(err), 0)) if err else np.full(4, np.nan)
    done = bool(duck.anim.last and duck.anim.last["name"] == name and duck.anim.last["completed"])
    return dict(fell=int(duck.falls > 0), done=int(done), tilt=mt, drift=float(np.linalg.norm(duck.data.qpos[:2] - p0)), rmse=e.tolist(),
                states=states, end=duck.state)


def exp_clips(duck, out, quick):
    rng = np.random.default_rng(12)
    n = 4 if quick else 20
    n_alt = 2 if quick else 6
    rows = []
    for name, c in duck.anim.lib.items():
        for seated in ((False, True) if c.seated_ok else (False,)):
            tr = [play_clip(duck, rng, name, k, seated=seated) for k in range(n)]
            jumped = sum(1 for t in tr if "jump" in t["states"])
            flipped = sum(1 for t in tr if "roulade" in t["states"])
            sat = sum(1 for t in tr if "seated" in t["states"] or "dead" in t["states"]) if not seated else 0
            raw = c.head
            lo, hi = (animation.HEAD_LO_SEATED, animation.HEAD_HI_SEATED) if seated else (animation.HEAD_LO_STAND, animation.HEAD_HI_STAND)
            row = dict(name=name, group=c.meta.get("group", ""), T=c.T, seated=seated, n=n, falls=sum(t["fell"] for t in tr), done=sum(t["done"] for t in tr),
                       tilt_mean=float(np.mean([t["tilt"] for t in tr])), tilt_max=float(np.max([t["tilt"] for t in tr])),
                       drift=float(np.mean([t["drift"] for t in tr])), rmse=np.nanmean([t["rmse"] for t in tr], 0).tolist(),
                       jumped=jumped, flipped=flipped, sat=sat, uses=c.meta.get("uses", []),
                       limited=float(np.mean(np.any((raw < lo - 1e-9) | (raw > hi + 1e-9), axis=1))),
                       ends=sorted(set(t["end"] for t in tr)))
            rows.append(row)
            print("   %-11s %-6s falls %2d/%d done %2d tilt %4.1f (max %4.1f) drift %.2f m rmse %s%s" % (
                name, "seated" if seated else "stand", row["falls"], n, row["done"], row["tilt_mean"], row["tilt_max"], row["drift"],
                np.round(row["rmse"], 3), (" jump %d flip %d sat %d" % (jumped, flipped, sat)) if (jumped or flipped or sat) else ""))
    # the other head mode, standing: how far the head ends up from the clip
    mode0 = dict(animation.HEAD_MODE)
    alt = {}
    animation.HEAD_MODE["walk"] = "cmd"
    for name, c in duck.anim.lib.items():
        if c.needs or c.stages or c.has_twist:
            continue
        tr = [play_clip(duck, rng, name, 100 + k) for k in range(n_alt)]
        alt[name] = dict(rmse=np.nanmean([t["rmse"] for t in tr], 0).tolist(), falls=sum(t["fell"] for t in tr), n=n_alt)
    animation.HEAD_MODE.update(mode0)
    st = [r for r in rows if not r["seated"]]
    tot = dict(clips=len(duck.anim.lib), trials=sum(r["n"] for r in rows), falls=sum(r["falls"] for r in rows), done=sum(r["done"] for r in rows),
               stand_trials=sum(r["n"] for r in st), stand_falls=sum(r["falls"] for r in st),
               seated_trials=sum(r["n"] for r in rows if r["seated"]), seated_falls=sum(r["falls"] for r in rows if r["seated"]),
               rmse_direct=float(np.mean([np.mean(r["rmse"]) for r in st if r["name"] in alt])),
               rmse_cmd=float(np.mean([np.mean(v["rmse"]) for v in alt.values()])),
               tilt_max=float(max(r["tilt_max"] for r in rows)))
    out["clips"] = dict(rows=rows, alt=alt, total=tot, n=n, n_alt=n_alt)
    print("   total: %d trials, %d falls, %d completed; head rmse direct %.3f vs policy command %.3f rad" % (tot["trials"], tot["falls"], tot["done"], tot["rmse_direct"], tot["rmse_cmd"]))
    fig, axs = plt.subplots(2, 1, figsize=(9.2, 5.0), sharex=True)
    names = [r["name"] for r in st]
    x = np.arange(len(names))
    axs[0].bar(x, [r["tilt_mean"] for r in st], color=[BLUE if r["group"] == "live" else PURPLE for r in st], width=0.6)
    axs[0].plot(x, [r["tilt_max"] for r in st], "_", color="k", ms=10, mew=1.5)
    for i, r in enumerate(st):
        if r["falls"]:
            axs[0].text(i, r["tilt_max"] + 1, "%d fell" % r["falls"], ha="center", fontsize=7, color=RED)
    axs[0].set_ylabel("peak trunk tilt [deg]\n(bar: mean, tick: worst)")
    axs[0].set_title("Every clip, %d jittered standing starts (blue: Duck Live originals, purple: Pollen's set)" % n, fontsize=9)
    w = 0.38
    axs[1].bar(x - w / 2, [np.mean(r["rmse"]) for r in st], w, color=BLUE, label="head servos directly (used)")
    axs[1].bar([i + w / 2 for i, r in enumerate(st) if r["name"] in alt], [np.mean(alt[r["name"]]["rmse"]) for r in st if r["name"] in alt], w, color=ORANGE,
               label="through the policy's head command")
    axs[1].set_ylabel("head tracking RMSE [rad]\n(mean of the 4 joints)")
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_anim_clips.pdf")); plt.close(fig)
    # one clip in detail: commanded vs measured head, beak, trunk tilt
    fresh(duck, rng, 7, lively=True)
    name = "excited" if "excited" in duck.anim.lib else next(iter(duck.anim.lib))
    duck.emote(name)
    T, H, M, B, TL = [], [], [], [], []
    for i in range(int((duck.anim.lib[name].T + 1.5) * 50)):
        duck.step()
        T.append(i * duck.control_dt); H.append(duck.anim.head.copy()); M.append(head_meas(duck)); B.append(duck.anim.mouth); TL.append(duck.tilt_deg)
    H, M = np.array(H), np.array(M)
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 4.8), sharex=True)
    for j, col in ((1, ORANGE), (2, BLUE)):
        axs[0].plot(T, H[:, j], color=col, lw=1.0, ls="--"); axs[0].plot(T, M[:, j], color=col, lw=1.3, label=HEAD[j])
    axs[0].set_ylabel("head offset [rad]\n(dashed: commanded)"); axs[0].legend(fontsize=7, frameon=False, ncol=2)
    axs[0].set_title("'%s' (Pollen) on the walk policy" % name, fontsize=9)
    axs[1].fill_between(T, B, color=PINK, alpha=0.6); axs[1].set_ylabel("beak opening")
    axs[2].plot(T, TL, color=GREY); axs[2].set_ylabel("trunk tilt [deg]"); axs[2].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_anim_clip_trace.pdf")); plt.close(fig)


# ---- idle life -----------------------------------------------------------------------------------------------------------
def exp_idle(duck, out, quick):
    rng = np.random.default_rng(13)
    seeds = 3 if quick else 8
    mins = 2.0 if quick else 10.0
    res = {}
    trace = None
    for cond, lively in (("idle life", True), ("bare policy", False)):
        falls, drift, tilt, yaw_all, glances, viewer, beak = 0, [], [], [], [], [], []
        for k in range(seeds):
            fresh(duck, rng, k, lively=lively)
            az = 2 * math.pi * k / seeds                       # the viewer somewhere on a 0.9 m circle around the duck, 0.4 m up
            duck.anim.viewer_pos = [0.9 * math.cos(az), 0.9 * math.sin(az), 0.40] if lively else None
            p0 = duck.data.qpos[:2].copy()
            n = int(mins * 60 * 50)
            Y = np.zeros(n); P = np.zeros(n); R = np.zeros(n); Bk = np.zeros(n); TL = np.zeros(n); g = 0; g_t0 = duck.anim.idle.g_t0; vw = 0
            for i in range(n):
                duck.step()
                h = head_meas(duck)
                Y[i], P[i], R[i], Bk[i], TL[i] = h[2], h[1], h[3], duck.anim.mouth, duck.tilt_deg
                if duck.anim.idle.g_t0 != g_t0:
                    g_t0 = duck.anim.idle.g_t0; g += 1
                vw += int(duck.anim.idle.g_viewer)
            falls += duck.falls; drift.append(float(np.linalg.norm(duck.data.qpos[:2] - p0))); tilt.append(float(TL.max()))
            yaw_all.append(Y); glances.append(g / mins); viewer.append(vw / n); beak.append(float(np.mean(Bk > 0.03)))
            if lively and k == 0:
                trace = dict(t=(np.arange(3000) / 50).tolist(), yaw=Y[:3000].tolist(), pitch=P[:3000].tolist(), roll=R[:3000].tolist(), beak=Bk[:3000].tolist())
        Yc = np.concatenate(yaw_all)
        res[cond] = dict(seeds=seeds, minutes=mins, falls=int(falls), drift_mean=float(np.mean(drift)), drift_max=float(np.max(drift)), tilt_max=float(np.max(tilt)),
                         yaw_std=float(np.std(Yc)), yaw_p95=float(np.percentile(np.abs(Yc), 95)), glances_per_min=float(np.mean(glances)),
                         viewer_frac=float(np.mean(viewer)), beak_frac=float(np.mean(beak)), hist=np.histogram(Yc, bins=40, range=(-0.9, 0.9), density=True)[0].tolist())
        print("   %-11s %d x %.0f min: falls %d, drift %.3f m (max %.3f), max tilt %.1f deg, head yaw std %.3f rad, %.1f glances/min, viewer %.0f %%" % (
            cond, seeds, mins, falls, res[cond]["drift_mean"], res[cond]["drift_max"], res[cond]["tilt_max"], res[cond]["yaw_std"], res[cond]["glances_per_min"], 100 * res[cond]["viewer_frac"]))
    # seated: dozing
    dz = []
    falls = 0
    for k in range(seeds):
        fresh(duck, rng, 50 + k, seated=True, lively=True)
        t_doze = None
        for i in range(int(180 * 50)):
            duck.step()
            if t_doze is None and duck.anim.idle.doze > 0.9:
                t_doze = i / 50
        falls += duck.falls
        dz.append(dict(t_doze=t_doze, pitch=float(head_meas(duck)[1]), state=duck.state, z=duck.z))
    res["seated"] = dict(seeds=seeds, falls=int(falls), t_doze=float(np.mean([d["t_doze"] for d in dz if d["t_doze"] is not None])),
                         dozed=sum(1 for d in dz if d["t_doze"] is not None), pitch=float(np.mean([d["pitch"] for d in dz])),
                         still_seated=sum(1 for d in dz if d["state"] == "seated"))
    print("   seated 3 min: falls %d, dozed %d/%d after %.0f s, head pitch %.2f rad, still seated %d" % (falls, res["seated"]["dozed"], seeds, res["seated"]["t_doze"], res["seated"]["pitch"], res["seated"]["still_seated"]))
    out["idle"] = dict(res=res, trace=trace)
    fig = plt.figure(figsize=(9.2, 4.2))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2]); ax.plot(trace["t"], trace["yaw"], color=BLUE, lw=1.0, label="head yaw"); ax.plot(trace["t"], trace["pitch"], color=ORANGE, lw=1.0, label="head pitch")
    ax.plot(trace["t"], trace["roll"], color=PURPLE, lw=0.9, label="head roll"); ax.set_ylabel("measured offset [rad]"); ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title("Idle life, the first minute of one seed (measured joint angles)", fontsize=9)
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax); ax2.fill_between(trace["t"], trace["beak"], color=PINK, alpha=0.7); ax2.set_ylabel("beak"); ax2.set_xlabel("time [s]")
    ax3 = fig.add_subplot(gs[:, 2]); c = np.linspace(-0.9, 0.9, 41); c = 0.5 * (c[1:] + c[:-1])
    ax3.plot(c, res["idle life"]["hist"], color=BLUE, label="idle life"); ax3.plot(c, np.minimum(res["bare policy"]["hist"], max(res["idle life"]["hist"]) * 1.6), color=GREY, label="bare policy (clipped)")
    ax3.set_xlabel("head yaw [rad]"); ax3.set_ylabel("density"); ax3.set_title("Where the head points", fontsize=9); ax3.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "duck_anim_idle.pdf")); plt.close(fig)


# ---- does a lively head cost anything while driving? ------------------------------------------------------------------------
def exp_walk(duck, out, quick):
    n = 4 if quick else 16
    script = [(2.5, ("up",)), (2.5, ("up", "left")), (2.0, ("up", "right")), (2.0, ("left",)), (1.5, ("down",)), (1.5, ())]
    res = {}
    for cond, lively in (("idle life", True), ("bare policy", False)):
        rng = np.random.default_rng(14)                 # the same starts for both conditions
        falls, ev, ew, tilt = 0, [], [], []
        for k in range(n):
            fresh(duck, rng, k, lively=lively)
            for secs, held in script:
                duck.set_keys(set(held))
                for i in range(int(secs * 50)):
                    duck.step()
                    if i > 50 and held:
                        ev.append(duck.v_filt[0] - duck.cmd[0]); ew.append(duck.wz_filt - duck.cmd[2])
                    tilt.append(duck.tilt_deg)
            falls += duck.falls
        pushes = dict(fell=0, up2=0, n=0, t_up=[])
        for lvl in (1.0, 1.5, 2.0, 2.5):
            for k in range(n // 2):
                fresh(duck, rng, 200 + k, lively=lively)
                ang = rng.uniform(0, 2 * math.pi)
                duck.push(lvl * math.cos(ang), lvl * math.sin(ang))
                tu = None
                for i in range(150):
                    duck.step()
                    if duck.falls > 0 and tu is None and duck.upright:
                        tu = (i + 1) * duck.control_dt
                pushes["n"] += 1; pushes["fell"] += int(duck.falls > 0)
                pushes["up2"] += int(duck.falls == 0 or (tu is not None and tu <= 2.0))
                if tu is not None:
                    pushes["t_up"].append(tu)
        res[cond] = dict(n=n, falls=int(falls), vx_rmse=float(np.sqrt(np.mean(np.square(ev)))), wz_rmse=float(np.sqrt(np.mean(np.square(ew)))), tilt_max=float(np.max(tilt)),
                         tilt_mean=float(np.mean(tilt)), push_n=pushes["n"], push_fell=pushes["fell"], push_up2=pushes["up2"],
                         push_t_up=float(np.mean(pushes["t_up"])) if pushes["t_up"] else float("nan"))
        print("   %-11s scenario x%d: falls %d, vx rmse %.3f m/s, wz rmse %.3f rad/s, tilt mean %.1f max %.1f | pushes %d: fell %d, standing after 2 s %d, up in %.2f s" % (
            cond, n, falls, res[cond]["vx_rmse"], res[cond]["wz_rmse"], res[cond]["tilt_mean"], res[cond]["tilt_max"], pushes["n"], pushes["fell"], pushes["up2"], res[cond]["push_t_up"]))
    duck.set_keys(set())
    out["walk"] = dict(res=res, script=[[s, list(h)] for s, h in script], push_levels=[1.0, 1.5, 2.0, 2.5])


# ---- reactions -------------------------------------------------------------------------------------------------------------
def exp_reactions(duck, out, quick):
    rng = np.random.default_rng(15)
    n = 4 if quick else 20
    res = {}

    def run_until(pred, secs):
        for i in range(int(secs * 50)):
            duck.step()
            if pred():
                return (i + 1) / 50
        return None

    def reacted(name, secs=6.0):
        return run_until(lambda: any(p[1] == name and p[2] for p in duck.anim.played), secs)

    for key, press, expect in (("jump", "space", ("yes_fast", "mmh")), ("flip", "r", ("proud",))):
        ok = fired = 0; lat = []
        for k in range(n):
            fresh(duck, rng, k, lively=True, react=True)
            duck.press(press)
            run_until(lambda: duck.state not in ("jump", "roulade") and duck.state_time > 0.1 and duck.ticks > 60, 6.0)
            landed = duck.state == "stand" and duck.falls == 0
            t = run_until(lambda: any(p[1] in expect and p[2] for p in duck.anim.played), 4.0) if landed else None
            ok += int(landed); fired += int(t is not None)
            if t is not None:
                lat.append(t)
            run_until(lambda: False, 3.0)
        res[key] = dict(n=n, landed=ok, reacted=fired, latency=float(np.mean(lat)) if lat else float("nan"), falls_after=int(duck.falls))
        print("   %-5s landed %d/%d, reaction %d, %.2f s after the hand-back" % (key, ok, n, fired, res[key]["latency"]))
    # knocked over -> gets up by itself -> shakes it off
    fell = up = shook = 0; lat = []
    for k in range(n):
        fresh(duck, rng, k, lively=True, react=True)
        ang = rng.uniform(0, 2 * math.pi)
        duck.push(3.0 * math.cos(ang), 3.0 * math.sin(ang))
        run_until(lambda: duck.falls > 0, 2.0)
        if duck.falls:
            fell += 1
            t = run_until(lambda: not duck.down and duck.upright, 6.0)
            up += int(t is not None)
            t2 = reacted("dizzy", 4.0)
            shook += int(t2 is not None)
            if t2 is not None:
                lat.append(t2)
            run_until(lambda: False, 3.0)
    res["knocked"] = dict(n=n, fell=fell, up=up, reacted=shook, latency=float(np.mean(lat)) if lat else float("nan"))
    print("   push 3 m/s: fell %d/%d, got up %d, shook it off %d" % (fell, n, up, shook))
    # play dead -> hold -> wake -> stand-up policy
    dead = up = 0; t_up = []
    for k in range(n):
        fresh(duck, rng, k, lively=True, react=False)
        duck.emote("play_dead")
        run_until(lambda: duck.anim.last is not None and duck.anim.last["name"] == "play_dead", 12.0)
        run_until(lambda: False, 2.0)
        if duck.state == "dead":
            dead += 1
            duck.press("s")
            t = run_until(lambda: duck.state == "stand" and duck.upright, 10.0)
            up += int(t is not None)
            if t is not None:
                t_up.append(t)
    res["play_dead"] = dict(n=n, dead=dead, up=up, t_up=float(np.mean(t_up)) if t_up else float("nan"), t_up_max=float(np.max(t_up)) if t_up else float("nan"))
    print("   play dead: on its back %d/%d, got up after the wake key %d, in %.2f s (max %.2f)" % (dead, n, up, res["play_dead"]["t_up"], res["play_dead"]["t_up_max"]))
    # an arrow key cuts a clip: time until the duck walks at 80 % of the commanded speed, vs no clip playing
    t80 = dict(clip=[], none=[])
    for cond in ("clip", "none"):
        r2 = np.random.default_rng(16)
        for k in range(n):
            fresh(duck, r2, k, lively=True)
            if cond == "clip":
                duck.emote("sad")
                run_until(lambda: False, 2.0)
            duck.set_keys({"up"})
            t = run_until(lambda: duck.v_filt[0] > 0.8 * 0.4, 4.0)
            cut = duck.anim.clip is None
            if t is not None and cut:
                t80[cond].append(t)
            duck.set_keys(set())
    # "hello" with the viewer behind the duck (the app's default camera): it turns round first (walk policy), then greets
    turn = []
    for k in range(n):
        fresh(duck, rng, k, lively=True)
        duck.anim.viewer_pos = [-0.55, -0.75, 0.42]
        v0 = duck.anim.direction_to(duck.anim.viewer_pos)[0]
        duck.emote("hello")
        t = run_until(lambda: duck.anim.clip is not None, 8.0)
        v1 = duck.anim.direction_to(duck.anim.viewer_pos)[0]
        run_until(lambda: duck.anim.clip is None, 6.0)
        turn.append(dict(t=t, v0=abs(v0), v1=abs(v1), fell=duck.falls))
    res["face"] = dict(n=n, started=sum(1 for x in turn if x["t"] is not None), t=float(np.mean([x["t"] for x in turn if x["t"] is not None])),
                       yaw0=float(np.mean([x["v0"] for x in turn])), yaw1=float(np.mean([x["v1"] for x in turn])), yaw1_max=float(np.max([x["v1"] for x in turn])),
                       falls=int(sum(x["fell"] for x in turn)))
    print("   hello, viewer %.2f rad round: turned to %.2f rad (max %.2f) in %.2f s, then greeted %d/%d, falls %d" % (
        res["face"]["yaw0"], res["face"]["yaw1"], res["face"]["yaw1_max"], res["face"]["t"], res["face"]["started"], n, res["face"]["falls"]))
    res["interrupt"] = dict(n=n, t80_clip=float(np.mean(t80["clip"])), t80_none=float(np.mean(t80["none"])), ok=len(t80["clip"]))
    print("   arrow key during 'sad': walking at 80 %% of 0.4 m/s after %.2f s (no clip: %.2f s), %d/%d" % (res["interrupt"]["t80_clip"], res["interrupt"]["t80_none"], len(t80["clip"]), n))
    out["reactions"] = res


def exp_cost(duck, out, quick):
    rng = np.random.default_rng(17)
    res = {}
    for cond, lively in (("bare policy", False), ("idle life", True), ("clip playing", True)):
        fresh(duck, rng, 0, lively=lively)
        if cond == "clip playing":
            duck.emote("sing")
        ms = []
        for i in range(200):
            t0 = time.perf_counter(); duck.step(); ms.append((time.perf_counter() - t0) * 1e3)
        res[cond] = dict(median=float(np.median(ms)), p99=float(np.percentile(ms, 99)))
        print("   %-13s step %.3f ms (p99 %.3f)" % (cond, res[cond]["median"], res[cond]["p99"]))
    out["cost"] = res


def exp_bodypose(duck, out, quick):
    """VelStand's body_pose command (obs slots 7..12; trained range +-5 mm on the first three, +-0.05 rad on the last
    three): hold one slot at one end of its range for 3 s and measure the trunk against the zero-command stand."""
    rng = np.random.default_rng(18)
    reps = 2 if quick else 6

    def hold(idx, val, k):
        r = np.random.default_rng(1000 + k)                 # the same start for every slot
        fresh(duck, r, k)
        p0 = duck.data.qpos[:2].copy()
        acc = []
        for i in range(150):
            duck.operate()
            duck.cmd[7 + idx] = val
            a = duck.policies[duck.policy_role](duck.observe())
            duck.last_action = a.astype(np.float32)
            for _ in range(duck.decimation):
                duck.servos.command(engine.DEFAULT_POSE + a); duck.servos.substep()
            duck.measure()
            if i >= 75:
                q = duck.data.qpos[duck.free_qadr + 3:duck.free_qadr + 7]
                g = engine.quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
                acc.append([duck.z * 1000, math.degrees(math.asin(g[0])), math.degrees(math.asin(-g[1])), math.degrees(engine.yaw_of(q))])
        return np.append(np.mean(acc, 0), 1000 * float(np.linalg.norm(duck.data.qpos[:2] - p0)))

    base = np.mean([hold(0, 0.0, k) for k in range(reps)], 0)
    rows = []
    for idx in range(6):
        for val in ((-0.005, 0.005) if idx < 3 else (-0.05, 0.05)):
            m = np.mean([hold(idx, val, k) for k in range(reps)], 0)
            rows.append(dict(slot=idx, value=val, dz=float(m[0] - base[0]), dpitch=float(m[1] - base[1]), droll=float(m[2] - base[2]), dyaw=float(m[3] - base[3]), walked=float(m[4])))
            print("   slot %d = %+.3f: dz %+.1f mm, nose-down %+.2f deg, roll %+.2f deg, yaw %+.1f deg, walked %.0f mm" % (idx, val, rows[-1]["dz"], rows[-1]["dpitch"], rows[-1]["droll"], rows[-1]["dyaw"], rows[-1]["walked"]))
    out["bodypose"] = dict(rows=rows, reps=reps, base=dict(z=float(base[0]), pitch=float(base[1]), walked=float(base[4])))


# ---- LaTeX fragments -------------------------------------------------------------------------------------------------------
def write_tex(out):
    L = []
    def mac(name, val, fmt="%s"):
        L.append("\\newcommand{\\%s}{%s}" % (name, fmt % val))
    if "clips" in out:
        t = out["clips"]["total"]
        mac("animClips", t["clips"], "%d"); mac("animTrials", t["trials"], "%d"); mac("animFalls", t["falls"], "%d"); mac("animDone", t["done"], "%d")
        mac("animStandTrials", t["stand_trials"], "%d"); mac("animStandFalls", t["stand_falls"], "%d")
        mac("animSeatedTrials", t["seated_trials"], "%d"); mac("animSeatedFalls", t["seated_falls"], "%d")
        mac("animN", out["clips"]["n"], "%d"); mac("animNalt", out["clips"]["n_alt"], "%d")
        mac("animRmseDirect", t["rmse_direct"], "%.3f"); mac("animRmseCmd", t["rmse_cmd"], "%.3f"); mac("animTiltMax", t["tilt_max"], "%.1f")
        rows = []
        for r in out["clips"]["rows"]:
            extra = []
            if r["jumped"]: extra.append("jump %d" % r["jumped"])
            if r["flipped"]: extra.append("flip %d" % r["flipped"])
            if r["sat"]: extra.append("sat %d" % r["sat"])
            rows.append("%s & %s & %s & %.1f & %d/%d & %d/%d & %.1f / %.1f & %.2f & %.3f & %s \\\\" % (
                r["name"].replace("_", "\\_"), "DL" if r["group"] == "live" else "P", "seated" if r["seated"] else "stand", r["T"], r["falls"], r["n"], r["done"], r["n"],
                r["tilt_mean"], r["tilt_max"], r["drift"], float(np.mean(r["rmse"])), ", ".join(extra) or "--"))
        with open(os.path.join(DOCS, "duck_anim_clip_table.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
    if "headtrack" in out:
        rows = []
        for k, lab in (("standing/cmd", "standing (VelStand), policy head command"), ("standing/direct", "standing (VelStand), head servos directly"),
                       ("seated/cmd", "seated (sit-stand), policy head command"), ("seated/direct", "seated (sit-stand), head servos directly")):
            r = out["headtrack"]["res"][k]
            rows.append("%s & %.3f & %.1f & %.3f & %d/%d \\\\" % (lab, r["rmse"], r["max_tilt"], r["drift"], r["fell"], r["trials"]))
            mac("headRmse" + "".join(w.capitalize() for w in k.split("/")), r["rmse"], "%.3f")
        with open(os.path.join(DOCS, "duck_anim_head_table.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
    if "idle" in out:
        a, b, s = out["idle"]["res"]["idle life"], out["idle"]["res"]["bare policy"], out["idle"]["res"]["seated"]
        mac("idleSeeds", a["seeds"], "%d"); mac("idleMinutes", a["minutes"], "%.0f"); mac("idleFalls", a["falls"], "%d"); mac("idleFallsBare", b["falls"], "%d")
        mac("idleDrift", a["drift_mean"] * 100, "%.1f"); mac("idleDriftMax", a["drift_max"] * 100, "%.1f"); mac("idleDriftBare", b["drift_mean"] * 100, "%.1f")
        mac("idleTilt", a["tilt_max"], "%.1f"); mac("idleTiltBare", b["tilt_max"], "%.1f"); mac("idleGlances", a["glances_per_min"], "%.1f")
        mac("idleYawStd", a["yaw_std"], "%.2f"); mac("idleYawStdBare", b["yaw_std"], "%.3f"); mac("idleYawP", a["yaw_p95"], "%.2f"); mac("idleViewer", 100 * a["viewer_frac"], "%.0f")
        mac("idleBeak", 100 * a["beak_frac"], "%.0f"); mac("dozeAfter", s["t_doze"], "%.0f"); mac("dozeN", s["dozed"], "%d"); mac("dozeFalls", s["falls"], "%d"); mac("dozePitch", s["pitch"], "%.2f")
    if "walk" in out:
        a, b = out["walk"]["res"]["idle life"], out["walk"]["res"]["bare policy"]
        for tag, r in (("Live", a), ("Bare", b)):
            mac("walk%sFalls" % tag, r["falls"], "%d"); mac("walk%sVx" % tag, r["vx_rmse"], "%.3f"); mac("walk%sWz" % tag, r["wz_rmse"], "%.3f")
            mac("walk%sTilt" % tag, r["tilt_mean"], "%.1f"); mac("walk%sTiltMax" % tag, r["tilt_max"], "%.1f")
            mac("push%sFell" % tag, r["push_fell"], "%d"); mac("push%sUp" % tag, r["push_up2"], "%d"); mac("push%sT" % tag, r["push_t_up"], "%.2f")
        mac("walkN", a["n"], "%d"); mac("pushN", a["push_n"], "%d")
    if "reactions" in out:
        r = out["reactions"]
        mac("reactN", r["jump"]["n"], "%d")
        mac("reactJumpLanded", r["jump"]["landed"], "%d"); mac("reactJump", r["jump"]["reacted"], "%d"); mac("reactJumpT", r["jump"]["latency"], "%.2f")
        mac("reactFlipLanded", r["flip"]["landed"], "%d"); mac("reactFlip", r["flip"]["reacted"], "%d"); mac("reactFlipT", r["flip"]["latency"], "%.2f")
        mac("reactFell", r["knocked"]["fell"], "%d"); mac("reactUp", r["knocked"]["up"], "%d"); mac("reactShook", r["knocked"]["reacted"], "%d")
        mac("deadN", r["play_dead"]["dead"], "%d"); mac("deadUp", r["play_dead"]["up"], "%d"); mac("deadT", r["play_dead"]["t_up"], "%.2f"); mac("deadTmax", r["play_dead"]["t_up_max"], "%.2f")
        mac("faceYawA", r["face"]["yaw0"], "%.2f"); mac("faceYawB", r["face"]["yaw1"], "%.2f"); mac("faceYawBmax", r["face"]["yaw1_max"], "%.2f")
        mac("faceT", r["face"]["t"], "%.2f"); mac("faceOk", r["face"]["started"], "%d"); mac("faceFalls", r["face"]["falls"], "%d")
        mac("cutClip", r["interrupt"]["t80_clip"], "%.2f"); mac("cutNone", r["interrupt"]["t80_none"], "%.2f"); mac("cutOk", r["interrupt"]["ok"], "%d")
    if "bodypose" in out:
        b = out["bodypose"]
        rows = ["%d & %+.3f & %+.1f & %+.2f & %+.2f & %+.1f & %.0f \\\\" % (r["slot"], r["value"], r["dz"], r["dpitch"], r["droll"], r["dyaw"], r["walked"]) for r in b["rows"]]
        with open(os.path.join(DOCS, "duck_anim_bodypose_table.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
        mac("bodyDzMax", max(abs(r["dz"]) for r in b["rows"]), "%.1f"); mac("bodyPitchMax", max(abs(r["dpitch"]) for r in b["rows"]), "%.1f")
        mac("bodyWalkMax", max(r["walked"] for r in b["rows"]), "%.0f"); mac("bodyReps", b["reps"], "%d")
    if "cost" in out:
        c = out["cost"]
        mac("costBare", c["bare policy"]["median"], "%.2f"); mac("costLive", c["idle life"]["median"], "%.2f"); mac("costClip", c["clip playing"]["median"], "%.2f")
    with open(os.path.join(DOCS, "duck_anim_numbers.tex"), "w") as f:
        f.write("% generated by app/bench_anim.py -- do not edit\n" + "\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="headtrack,clips,idle,walk,reactions,cost,bodypose")
    ap.add_argument("--tex-only", action="store_true")
    a = ap.parse_args()
    path = os.path.join(DOCS, "bench_anim_results.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    if a.tex_only:
        write_tex(out); return
    duck = engine.Duck(auto_reset=False)
    out.update(date=time.strftime("%Y-%m-%d %H:%M"), policies=dict((r, p.name) for r, p in duck.policies.items()), quick=a.quick,
               head_mode=dict(animation.HEAD_MODE), limits=dict(stand=[animation.HEAD_LO_STAND.tolist(), animation.HEAD_HI_STAND.tolist()],
                                                               seated=[animation.HEAD_LO_SEATED.tolist(), animation.HEAD_HI_SEATED.tolist()]))
    for name in a.only.split(","):
        print("== " + name)
        t0 = time.time()
        globals()["exp_" + name](duck, out, a.quick)
        print("   (%.0f s)" % (time.time() - t0))
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    write_tex(out)
    print("\nwrote docs/bench_anim_results.json, docs/duck_anim_*.tex and docs/figures/duck_anim_*.pdf")


if __name__ == "__main__":
    main()
