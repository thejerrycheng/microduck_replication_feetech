#!/usr/bin/env python3
"""Duck Live: the 2026-11-20 picks from the 4090 (flip round 6 @14000, level-head sit/stand @8497) against the
policies they replace (flip round 2 @13249, sit/stand 5499), on the app's own plant (MuJoCo + BAM HD1910 servos).

  ~/.venvs/microduck_live/bin/python app/bench_real3.py [--quick]

Per sit/stand set: bench_mac's transitions experiment (sit settle, stand-up, stop-falls) plus the head's world tilt
off level while seated (body jaw_soft, zero head command, 1-3 s after S) -- the user's complaint on the real duck was
the head dropping while seated. Per flip set: bench_mac's roulade experiment. Fixed seeds (the same starts for both
sets). Writes docs/bench_real3_results.json.
"""
from __future__ import annotations

import argparse, json, math, os, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench_mac as B  # noqa: E402
import engine  # noqa: E402

B.FIG = tempfile.mkdtemp(prefix="bench_real3_")
SIT = (("sitstand 5499", "sitstand_fast_hd1910_iter5499.onnx"), ("sitstand 8497 level head", "sitstand_fast_levelhead_hd1910_iter8497.onnx"))
FLIP = (("flip r2 13249", "roulade_head_real2_hd1910_iter13249.onnx"), ("flip r6 14000", "roulade_head_real3_hd1910_iter14000.onnx"))


def head_tilt_seated(duck, trials):
    """Head off level while seated, deg (median over 1-3 s after S): the angle between jaw_soft's x axis and world up,
    as docs/real2/sit/probe_head.py on the 4090 measures it (0 = level head)."""
    out = []
    rng = np.random.default_rng(5)
    for _ in range(trials):
        B.jitter(duck, rng)
        B.rollout(duck, 1.0)
        duck.press("s")
        A = []
        for i in range(int(3.0 / duck.control_dt)):
            duck.step()
            if i * duck.control_dt >= 1.0:
                R = duck.data.xmat[duck.head_body].reshape(3, 3)
                A.append(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 0])))))   # the Isaac probe's angle
        out.append(float(np.median(A)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    duck = engine.Duck(auto_reset=False)
    res = dict(date=time.strftime("%Y-%m-%d %H:%M"), quick=a.quick, sitstand={}, flip={})
    for label, f in SIT:
        duck.load_policies(dict(sitstand=f))
        o = {}
        B.exp_transitions(duck, o, a.quick)
        tr = o["transitions"]
        ht = head_tilt_seated(duck, 4 if a.quick else 10)
        res["sitstand"][label] = dict(policy=f, sit_fell=sum(bool(s["fell"]) for s in tr["sit"]), sit_trials=len(tr["sit"]),
                                      standing_after=sum(s["standing"] for s in tr["stand"]), stand_trials=len(tr["stand"]),
                                      stop_falls=sum(s["falls"] for s in tr["stop"]), stop_trials=sum(s["trials"] for s in tr["stop"]),
                                      head_tilt_seated_deg=ht)
        r = res["sitstand"][label]
        print(f"{label:28s} sit fell {r['sit_fell']}/{r['sit_trials']}, standing after S {r['standing_after']}/{r['stand_trials']},"
              f" stop falls {r['stop_falls']}/{r['stop_trials']}, head off level seated median {np.median(ht):.1f} deg"
              f" [{min(ht):.1f}, {max(ht):.1f}]")
    for label, f in FLIP:
        duck.load_policies(dict(roulade=f))
        o = {}
        B.exp_roulade(duck, o, a.quick)
        r = {k: o["roulade"][k] for k in ("trials", "completed", "standing", "fell", "peak_deg")}
        res["flip"][label] = dict(policy=f, **r)
        print(f"{label:28s} completed {r['completed']}/{r['trials']}, standing {r['standing']}/{r['trials']}, fell {r['fell']}/{r['trials']}")
    json.dump(res, open(os.path.join(REPO, "docs", "bench_real3_results.json"), "w"), indent=1)
    print("wrote docs/bench_real3_results.json")


if __name__ == "__main__":
    main()
