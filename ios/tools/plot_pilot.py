#!/usr/bin/env python3
"""Play with the real duck in Setup (the duck's own policies, its pilot), measured on the stand-in duck: what the app
sent and what the pilot commanded.

Reads logs/ios/pilot_mock.jsonl, written by ios/tools/mock_studio.py during the UI test `testPilot`
(ios/tools/test.sh), and the METRIC lines of logs/ios/test.log. The stand-in runs the real pilot
(firmware/duck_pilot.py) and policy pack with a still, upright IMU: commands, states and timings are the duck's own
code, balance is not modelled.

    ios/tools/test.sh && ~/.venvs/microduck_live/bin/python ios/tools/plot_pilot.py
    -> docs/figures/duck_ios_pilot.pdf, docs/duck_ios_pilot_table.tex, a table on stdout
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / "logs/ios/pilot_mock.jsonl").read_text().splitlines() if l.strip()]
# the pilot's runs: from each live start to its stop
runs, cur = [], None
for r in rows:
    if r["what"] == "start":
        cur = [] if r["live"] else None
    elif cur is not None:
        if r["what"] == "stop":
            runs.append(cur); cur = None
        else:
            cur.append(r)
gap, stop, dur, hz, pushes, presses = [], [], {}, [], 0, 0
for run in runs:
    st = [r for r in run if r["what"] == "state"]
    dv = [r for r in run if r["what"] == "drive"]
    presses += sum(r["what"] == "press" for r in run)
    if len(st) < 100:
        continue
    t = np.array([r["t"] for r in st]); cmd = np.array([r["cmd"] for r in st])
    hz.append((len(st) - 1) / (t[-1] - t[0]))
    td = np.array([r["t"] for r in dv]); on = np.array([any(r["stick"][:3]) for r in dv])
    g = np.diff(td)[on[1:] & on[:-1]]
    gap += list(g[g < 0.5])                                  # a longer one is the pause between two pushes
    for i in range(1, len(dv)):                              # each release: until the pilot's command is zero again
        if on[i - 1] and not on[i]:
            pushes += 1
            k = np.searchsorted(t, td[i])
            z = np.nonzero(np.abs(cmd[k:]).max(1) < 0.01)[0]
            if len(z): stop.append(t[k + z[0]] - td[i])
    names = [r["state"] for r in st]                         # how long each visit to a state lasted
    a = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[a]:
            dur.setdefault(names[a], []).append(t[i - 1] - t[a] + 0.02); a = i
gap = np.array(gap)
metric = {}
log = ROOT / "logs/ios/test.log"
if log.exists():
    for name, v in re.findall(r"METRIC ((?:pilot|clip)_\S+) (\S+)", log.read_text()): metric.setdefault(name, []).append(float(v))
s = lambda v, d=2: f"{np.mean(v):.{d}f} ({min(v):.{d}f}..{max(v):.{d}f})"
out = [("pilot runs, power on to let go", f"{sum(len(r) > 100 for r in runs)}", f"control loop {np.mean(hz):.1f} Hz"),
       ("pushes of a stick", f"{pushes}", "each ended by a zero command"),
       ("gap between commands while held [ms]", s(gap * 1000, 0), f"100 (n = {len(gap)})"),
       ("longest gap while held [ms]", f"{gap.max() * 1000:.0f}", "under the pilot's 500"),
       ("command zero after release [s]", s(stop), f"n = {len(stop)}")]
for k, label in (("jump", "a jump lasts [s]"), ("flip", "a flip lasts [s]"), ("sit_down", "sitting down [s]"), ("stand_up", "standing up [s]")):
    if k in dur: out.append((label, s(dur[k]), f"n = {len(dur[k])}"))
names = dict(clip_start_s="emotion tapped to playing [s]", clip_s="``Hello'', tap to over [s]",
             pilot_start_s="Start tapped to Standing shown [s]", pilot_fwd_m="forward, level 3, 3 s push [m]",
             pilot_right_m="to the right, level 3, 2 s push [m]", pilot_cw_deg="clockwise, level 3, 1 s push [deg]",
             pilot_m_level1="forward, level 1, 2 s push [m]", pilot_m_level5="forward, level 5, 2 s push [m]",
             pilot_run_m="running, level 3, 2 s push [m]")
expect = dict(clip_s="3.5 + status lag", pilot_fwd_m="1.20 = 0.40 m/s x 3 s", pilot_right_m="0.56 = 0.28 m/s x 2 s",
              pilot_cw_deg="40 = 0.70 rad/s x 1 s", pilot_m_level1="0.40 = 0.20 m/s x 2 s", pilot_m_level5="1.20 = 0.60 m/s x 2 s",
              pilot_run_m="1.40 = 0.70 m/s x 2 s")
for k, label in names.items():
    if k in metric: out.append((label, s(metric[k]) if len(metric[k]) > 1 else f"{metric[k][0]:.2f}", (expect.get(k, "") + f" (n = {len(metric[k])})").strip()))
for a, b, c in out: print(f"{a:44} {b:>22}   {c}")
(ROOT / "docs/duck_ios_pilot_table.tex").write_text("\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in out).replace(" x ", r" $\times$ ") + "\n\\bottomrule\n")

# the figure: the longest run. Sticks sent (as fractions) and the velocity the pilot commanded; its state; the gaps
run = max(runs, key=len)
st = [r for r in run if r["what"] == "state"]; t0 = st[0]["t"]
t = np.array([r["t"] for r in st]) - t0; cmd = np.array([r["cmd"] for r in st]); stick = np.array([r["stick"] for r in st])
fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(6.4, 7.6), gridspec_kw=dict(height_ratios=[1.3, 1.1, 0.9, 0.9]))
a1.plot(t, cmd[:, 0], color="#007aff", lw=1.4, label="commanded, forward [m/s]")
a1.plot(t, -cmd[:, 1], color="#34c759", lw=1.2, label="commanded, to the right [m/s]")
a1.step(t, stick[:, 1], where="post", color="#8e8e93", lw=0.8, label="left stick, forward (fraction)")
a1.step(t, stick[:, 0], where="post", color="#c7c7cc", lw=0.8, label="left stick, right (fraction)")
a2.plot(t, -cmd[:, 2], color="#ff9500", lw=1.4, label="commanded, clockwise [rad/s]")
a2.step(t, stick[:, 2], where="post", color="#8e8e93", lw=0.8, label="right stick (fraction)")
states = sorted(set(r["state"] for r in st), key=[r["state"] for r in st].index)
a3.step(t, [states.index(r["state"]) for r in st], where="post", color="#af52de", lw=1.2)
a3.plot(t, [r["level"] - 1 for r in st], color="#8e8e93", lw=0.8, ls=":")
a3.set_yticks(range(len(states))); a3.set_yticklabels(states, fontsize=6); a3.set_xlabel("time since power on [s]  (dotted: speed level - 1)")
for ax in (a1, a2, a3): ax.grid(alpha=.25); ax.set_xlim(0, t[-1])
for ax in (a1, a2): ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
a4.hist(gap * 1000, bins=np.arange(0, 260, 10), color="#007aff", alpha=.8)
a4.axvline(100, color="#8e8e93", lw=0.8, ls="--"); a4.set_xlabel("gap between stick commands while a stick is held [ms]"); a4.set_ylabel("count"); a4.grid(alpha=.25)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/duck_ios_pilot.pdf")
