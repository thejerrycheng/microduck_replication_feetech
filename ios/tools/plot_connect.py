#!/usr/bin/env python3
"""The connection test's numbers and figure, from logs/ios/connect.log (ios/tools/test_connect.sh, real duck).

    ios/tools/test_connect.sh && ~/.venvs/microduck_live/bin/python ios/tools/plot_connect.py
    -> docs/figures/duck_ios_connect.pdf, docs/duck_ios_connect_table.tex, a table on stdout
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
m, mode, res = defaultdict(lambda: defaultdict(list)), None, defaultdict(lambda: [0, 0])
for line in (ROOT / "logs/ios/connect.log").read_text().splitlines():
    if t := re.match(r"TRIAL find=(\S+)", line): mode = t[1]
    elif t := re.match(r"METRIC (\S+) (\S+)", line): m[mode][t[1]].append(float(t[2]))
    elif "Test Case" in line: res[mode]["passed" in line] += 1
names = {"all": "all three", "bonjour": "Bonjour alone", "name": "name alone", "sweep": "sweep alone", "drop": "(drop runs)"}
rows = []
print(f"{'finding by':16} {'runs ok':>8} {'found [s] mean (min..max)':>28} {'connected [s] mean (min..max)':>30}")
for k in ("all", "bonjour", "name", "sweep"):
    f, r = m[k]["found_s"], m[k]["ready_s"]
    if not f: continue
    s = lambda v: f"{np.mean(v):.2f} ({min(v):.2f}..{max(v):.2f})" if v else "--"
    print(f"{names[k]:16} {res[k][1]:>4}/{sum(res[k]):<3} {s(f):>28} {s(r):>30}")
    rows.append(f"{names[k]} & {res[k][1]}/{sum(res[k])} & {s(f)} & {s(r)} \\\\")
lost, back = m["drop"]["lost_s"], m["drop"]["back_s"]
if lost:
    print(f"server stopped for 8 s: noticed after {np.mean(lost):.1f} s ({min(lost):.1f}..{max(lost):.1f}); "
          f"found and connected again {np.mean(back):.1f} s ({min(back):.1f}..{max(back):.1f}) after that, n={len(back)}")
    rows.append(f"\\midrule outage: noticed, back & {len(back)}/{sum(res['drop'])} & {np.mean(lost):.1f} ({min(lost):.1f}..{max(lost):.1f}) & {np.mean(back):.1f} ({min(back):.1f}..{max(back):.1f}) \\\\")
(ROOT / "docs/duck_ios_connect_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")       # the rule is in the file: \\input cannot be followed by one
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ks = [k for k in ("all", "bonjour", "name", "sweep") if m[k]["found_s"]]
for i, k in enumerate(ks):
    ax.plot([i - .12] * len(m[k]["found_s"]), m[k]["found_s"], "o", color="#007aff", alpha=.7, label="duck found" if i == 0 else None)
    ax.plot([i + .12] * len(m[k]["ready_s"]), m[k]["ready_s"], "s", color="#34c759", alpha=.7, label="computer and bus connected" if i == 0 else None)
ax.set_xticks(range(len(ks))); ax.set_xticklabels([names[k] for k in ks]); ax.set_ylabel("time from cold launch [s]"); ax.set_ylim(0)
ax.grid(alpha=.25, axis="y"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/duck_ios_connect.pdf")
