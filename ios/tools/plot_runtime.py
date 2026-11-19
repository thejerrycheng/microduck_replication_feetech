#!/usr/bin/env python3
"""Play with the real duck selected, measured on the stand-in runtime: what the app sent and what was applied.

Reads logs/ios/runtime_mock.jsonl, written by ios/tools/mock_runtime.py during the UI test `testRunPlay`
(ios/tools/test.sh), and the METRIC lines of logs/ios/test.log.

    ios/tools/test.sh && ~/.venvs/microduck_live/bin/python ios/tools/plot_runtime.py
    -> docs/figures/duck_ios_runtime.pdf, docs/duck_ios_runtime_table.tex, a table on stdout
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / "logs/ios/runtime_mock.jsonl").read_text().splitlines() if l.strip()]
st = [r for r in rows if r["method"] == "state"]
mv = [r for r in rows if r["method"] == "robot.move"]
ev = [r for r in rows if r["method"] not in ("state", "robot.move", "robot.health", "hello", "robot.subscribe")]
t = np.array([r["t"] for r in st]); req = np.array([r["req"] for r in st]); app = np.array([r["applied"] for r in st])
tm = np.array([r["t"] for r in mv]); vm = np.array([[r["params"][k] for k in ("vx", "vy", "vyaw")] for r in mv])
moving = np.abs(vm).sum(1) > 0
# gaps between consecutive non-zero commands of one push (a gap over 0.5 s is the pause between two pushes)
gap = np.diff(tm)[moving[1:] & moving[:-1]]
gap = gap[gap < 0.5]
# every push: from its first non-zero command to the zero that ends it; the release must be the last message
pushes, start = [], None
for i, on in enumerate(moving):
    if on and start is None: start = i
    if not on and start is not None: pushes.append((tm[start], tm[i])); start = None
# time for the applied speed to fall under 0.01 m/s (or rad/s) after each release
stop = []
for _, t1 in pushes:
    k = np.searchsorted(t, t1)
    below = np.nonzero(np.abs(app[k:]).max(1) < 0.01)[0]
    if len(below): stop.append(t[k + below[0]] - t1)
metric = {}
log = ROOT / "logs/ios/test.log"
if log.exists():
    for name, v in re.findall(r"METRIC (rt_\S+) (\S+)", log.read_text()): metric.setdefault(name, []).append(float(v))
s = lambda v, d=2: f"{np.mean(v):.{d}f} ({min(v):.{d}f}..{max(v):.{d}f})"
out = [("pushes of a stick", f"{len(pushes)}", "each ended by a zero command"),
       ("gap between commands while held [ms]", s(gap * 1000, 0), f"100 (n = {len(gap)})"),
       ("longest gap while held [ms]", f"{gap.max() * 1000:.0f}", "under the 500 deadman"),
       ("applied speed under 0.01 after release [s]", s(stop), f"n = {len(stop)}")]
names = dict(rt_ready_s="Run tapped to runtime answering [s]", rt_stand_s="power tapped to Standing [s]",
             rt_fwd_m="forward, level 3, 3 s push [m]", rt_right_m="to the right, level 3, 2 s push [m]",
             rt_cw_deg="clockwise, level 3, 1 s push [deg]", rt_m_level1="forward, level 1, 2 s push [m]",
             rt_m_level5="forward, level 5, 2 s push [m]")
expect = dict(rt_fwd_m="0.54 = 0.18 m/s x 3 s", rt_right_m="0.36 = 0.18 m/s x 2 s", rt_cw_deg="52 = 0.9 rad/s x 1 s",
              rt_m_level1="0.12 = 0.06 m/s x 2 s", rt_m_level5="0.60 = 0.30 m/s x 2 s", rt_stand_s="2.0 ramp")
for k, label in names.items():
    if k in metric: out.append((label, s(metric[k]) if len(metric[k]) > 1 else f"{metric[k][0]:.2f}", expect.get(k, "") + f" (n = {len(metric[k])})"))
for a, b, c in out: print(f"{a:46} {b:>22}   {c}")
(ROOT / "docs/duck_ios_runtime_table.tex").write_text("\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in out).replace(" x ", r" $\times$ ") + "\n\\bottomrule\n")

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(6.4, 6.2), gridspec_kw=dict(height_ratios=[1.3, 1.3, 1]))
for ax, i, unit, name in ((a1, 0, "m/s", "forward"), (a2, 2, "rad/s", "turn")):
    if i == 0: ax.plot(t, app[:, 1], color="#34c759", lw=1.2, label="applied, to the left")
    ax.step(tm, vm[:, i], where="post", color="#8e8e93", lw=0.9, label="sent by the app")
    ax.plot(t, app[:, i], color="#007aff", lw=1.4, label=f"applied, {name}")
    for r in ev:
        ax.axvline(r["t"], color="#ff9500", lw=0.6, alpha=0.6)
        if i == 0: ax.text(r["t"], ax.get_ylim()[1], " " + (r["params"].get("skill") or r["method"].split(".")[1] + ("" if "on" not in r["params"] else " on" if r["params"]["on"] else " off")),
                           rotation=90, va="top", ha="left", fontsize=6, color="#ff9500")
    ax.set_ylabel(f"{name} [{unit}]"); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.set_xlim(0, t[-1])
a2.set_xlabel("time since the stand-in runtime started [s]")
a3.hist(gap * 1000, bins=np.arange(0, 260, 10), color="#007aff", alpha=.8)
a3.axvline(100, color="#8e8e93", lw=0.8, ls="--"); a3.set_xlabel("gap between velocity commands while a stick is held [ms]"); a3.set_ylabel("count"); a3.grid(alpha=.25)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/duck_ios_runtime.pdf")
