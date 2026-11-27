#!/usr/bin/env python3
"""The kick (B) on the Duck Live plant: when to hand the stand back to the walk policy.

  ~/.venvs/microduck_live/bin/python app/bench_kick.py [--quick] [--policy kick_....onnx]

The kick export (round 2, 7998) gets the ball away within ~0.1 s and then leans back (~50 deg by 0.5 s), so the
hand-back time KICK_S decides whether the walk policy can take the stand over or the stand-up policy has to. Sweeps
KICK_S over seeded, jittered standing starts (bench_mac.jitter: +-1 deg joints, +-2 deg yaw, a random servo lag); per
trial: the ball's peak speed along the heading and its heading error, the trunk tilt at the hand-back, whether the
stand-up policy was needed (= the duck was not on its feet at the hand-back, or fell under the walk afterwards) and
whether it is standing still 3 s after B. Writes docs/bench_kick_results.json and docs/figures/duck_app_kick.pdf.
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

B.FIG = tempfile.mkdtemp(prefix="bench_kick_")
SWEEP = (0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.6, 1.0)


def trial(duck, rng, kick_s):
    engine.KICK_S = kick_s
    B.jitter(duck, rng)
    B.rollout(duck, 1.5)
    duck.press("b")
    if duck.state != "kick":
        return None
    yaw = duck.kick["yaw"]
    f0 = duck.falls
    peak, peak_dir, tilt_handback, recovered, TILT = 0.0, 0.0, None, False, []
    for i in range(int(3.0 / duck.control_dt)):
        was = duck.state
        duck.step()
        v = duck.data.qvel[duck.ball_dadr:duck.ball_dadr + 2]
        fwd = float(v[0] * math.cos(yaw) + v[1] * math.sin(yaw))
        if fwd > peak:
            peak = fwd
            peak_dir = math.degrees(math.atan2(v[1], v[0]) - yaw)
        if was == "kick" and duck.state != "kick":
            tilt_handback = duck.tilt_deg
        recovered |= duck.state == "recover"
        TILT.append(duck.tilt_deg)
    T = np.array(TILT)
    return dict(kick_s=kick_s, ball_peak=peak, ball_dir_err_deg=abs((peak_dir + 180) % 360 - 180), tilt_handback=tilt_handback,
                tilt_max=float(T.max()), over60_s=float((T > engine.FALL_TILT_DEG).sum() * duck.control_dt),
                needed_standup=bool(recovered or duck.falls > f0), standing=bool(duck.state == "stand" and duck.upright),
                tilt=TILT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--policy", default=engine.DEFAULTS["kick"])
    a = ap.parse_args()
    n = 4 if a.quick else 12
    duck = engine.Duck(auto_reset=False)
    duck.load_policies(dict(kick=a.policy))
    kick_s0 = engine.KICK_S
    rows, traces = [], {}
    for ks in SWEEP:
        rng = np.random.default_rng(21)              # the same starts for every hand-back time
        R = [r for r in (trial(duck, rng, ks) for _ in range(n)) if r is not None]
        traces[ks] = [r.pop("tilt") for r in R]
        row = dict(kick_s=ks, trials=len(R),
                   ball_peak_mean=float(np.mean([r["ball_peak"] for r in R])), ball_peak_min=float(np.min([r["ball_peak"] for r in R])),
                   ball_peak_max=float(np.max([r["ball_peak"] for r in R])),
                   kicked=int(sum(r["ball_peak"] >= 0.5 for r in R)),
                   dir_err_deg_median=float(np.median([r["ball_dir_err_deg"] for r in R])),
                   tilt_handback_mean=float(np.mean([r["tilt_handback"] for r in R if r["tilt_handback"] is not None])),
                   needed_standup=int(sum(r["needed_standup"] for r in R)), standing_3s=int(sum(r["standing"] for r in R)),
                   tilt_max_mean=float(np.mean([r["tilt_max"] for r in R])), tilt_max_min=float(np.min([r["tilt_max"] for r in R])),
                   went_past_60=int(sum(r["tilt_max"] > engine.FALL_TILT_DEG for r in R)),
                   over60_s_mean=float(np.mean([r["over60_s"] for r in R])))
        rows.append(row)
        print(f"KICK_S {ks:.2f}: ball {row['ball_peak_mean']:.2f} m/s [{row['ball_peak_min']:.2f},{row['ball_peak_max']:.2f}], kicked {row['kicked']}/{row['trials']},"
              f" dir err {row['dir_err_deg_median']:.1f} deg, tilt at hand-back {row['tilt_handback_mean']:.0f} deg,"
              f" peak tilt {row['tilt_max_mean']:.0f} deg (min {row['tilt_max_min']:.0f}), past 60 deg {row['went_past_60']}/{row['trials']} for {row['over60_s_mean']:.2f} s,"
              f" stand-up needed {row['needed_standup']}/{row['trials']}, standing at 3 s {row['standing_3s']}/{row['trials']}")
    engine.KICK_S = kick_s0
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), policy=a.policy, trials=n, walk=duck.policies["walk"].name,
               recovery=duck.policies["recovery"].name, chosen_kick_s=kick_s0, sweep=rows)
    json.dump(out, open(os.path.join(REPO, "docs", "bench_kick_results.json"), "w"), indent=1)

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2))
    t = np.arange(1, len(next(iter(traces.values()))[0]) + 1) * duck.control_dt
    cm = plt.get_cmap("viridis")
    for i, ks in enumerate(SWEEP):
        tr = np.array(traces[ks])
        axs[0].plot(t, np.median(tr, 0), color=cm(i / (len(SWEEP) - 1)), lw=1.3, label=f"hand-back {ks:.2f} s")
    axs[0].axhline(engine.FALL_TILT_DEG, color=B.RED, lw=.8, ls="--")
    axs[0].set_xlabel("time since B [s]"); axs[0].set_ylabel("trunk tilt, median [deg]"); axs[0].set_title("tilt after the kick", fontsize=9)
    axs[0].legend(fontsize=6.5)
    ks = [r["kick_s"] for r in rows]
    axs[1].plot(ks, [r["standing_3s"] / r["trials"] for r in rows], "o-", color=B.BLUE, label="standing 3 s after B")
    axs[1].plot(ks, [1 - r["needed_standup"] / r["trials"] for r in rows], "s--", color=B.GREEN, label="no stand-up needed")
    axs[1].plot(ks, [r["kicked"] / r["trials"] for r in rows], "^:", color=B.ORANGE, label="ball >= 0.5 m/s")
    axs[1].plot(ks, [r["went_past_60"] / r["trials"] for r in rows], "v-", color=B.RED, label="tipped past 60 deg (any time)")
    axs[1].axvline(kick_s0, color=B.GREY, lw=4, alpha=.25)
    axs[1].set_xlabel("hand-back time KICK_S [s]"); axs[1].set_ylabel(f"fraction of {n} starts"); axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_title("kick outcome vs hand-back (band: the app's choice)", fontsize=9); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "docs", "figures", "duck_app_kick.pdf")); plt.close(fig)
    print("wrote docs/bench_kick_results.json, docs/figures/duck_app_kick.pdf")


if __name__ == "__main__":
    main()
