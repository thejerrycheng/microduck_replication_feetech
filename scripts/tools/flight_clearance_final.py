#!/usr/bin/env python3
"""Blocked-pose score of the frame as built: the collision primitives of docs/flight_cad_results.json plus the four fans.

    python scripts/tools/flight_clearance_final.py     # after cad/flight/build_parts.py; takes ~10 min

Writes docs/flight_clearance.json (also scores the v3 spar frame for comparison, from flight_clearance_candidates.json).
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_clearance as FC

cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
pack = json.load(open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json")))
boxes = []
for c in cad["collision"]:
    if c["type"] == "box":
        p, s = np.array(c["pos"]), np.array(c["size"])
        if p[1] + s[1] < 0:            # right-hand copies are added by the mirror in FC.hits
            continue
        boxes.append([p[0] - s[0], p[0] + s[0], max(p[1] - s[1], 0.0), p[1] + s[1], p[2] - s[2], p[2] + s[2]])
    elif c["type"] == "capsule" and c["fromto"][1] > 0:
        a, b, r = np.array(c["fromto"][:3]), np.array(c["fromto"][3:]), c["radius"]
        for t in np.linspace(0, 1, 9):
            q = a + t * (b - a)
            boxes.append([q[0] - r, q[0] + r, q[1] - r, q[1] + r, q[2] - r * 1.6, q[2] + r * 1.6])
e = cad["edf_collision"]
for f in pack["fans"]:
    if f["pos"][1] > 0:
        p, r = np.array(f["pos"]), e["radius_mm"] * 1e-3
        boxes.append([p[0] - r, p[0] + r, p[1] - r, p[1] + r, p[2] + e["z0_mm"] * 1e-3, p[2] + e["z1_mm"] * 1e-3])
r = FC.blocked(boxes)
old = json.load(open(os.path.join(ROOT, "docs", "flight_clearance_candidates.json")))["v3 roof spars (current)"]
json.dump(dict(v4=r, v3=old, n_boxes=len(boxes), margin_mm=FC.MARGIN * 1e3), open(os.path.join(ROOT, "docs", "flight_clearance.json"), "w"), indent=1)
fr = lambda d, j: "blocked at stand" if d[j] is None else f"{d[j][0]:+.0f}..{d[j][1]:+.0f} of {d[j][2]:+.0f}..{d[j][3]:+.0f}"
for lab, d in (("v4 as built", r), ("v3 spars", old)):
    print(f"{lab}: head/neck poses blocked {d['head']*100:.1f} %, leg poses blocked {d['leg']*100:.1f} %")
    for j in ("neck_pitch", "head_pitch", "head_yaw", "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee"):
        print(f"    {j:15s} free from stand: {fr(d, j)} deg")
