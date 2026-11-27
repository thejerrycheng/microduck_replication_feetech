#!/usr/bin/env python3
"""Tool access to every screw of the flying-duck frame.

    .venv-cad/bin/python scripts/tools/flight_tool_clearance.py

For each screw, a driver shank (cylinder, radius R_TOOL) is swept out of the head along the screw axis and along tilted
axes (ball-end hex key, up to TILT_MAX from the axis). The free length is the distance to the first point of any part
present at that assembly step (surface points sampled from the STLs in cad/flight/out, CAD frame, mm). A screw passes
if some direction within TILT_MAX gives at least L_NEED of free length (screw length + short arm of a hex key).
Writes docs/flight_tool_clearance.json and docs/figures/flight_tool_clearance.pdf.
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_config as C
R_TOOL, R_HEAD, TILT_MAX, L_NEED, L_MAX = 2.0, 2.9, 25.0, 8.0 + 20.0, 120.0   # mm, deg: 2.5 mm hex key shank r, head r, ball-end limit
rng = np.random.default_rng(0)
cadres = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
G = cadres["screws"]


def stl_pts(name, n=400000):
    v = np.fromfile(os.path.join(ROOT, "cad", "flight", "out", name + ".stl"), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), offset=84)["v"].astype(float)
    ar = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    i = rng.choice(len(v), n, p=ar / ar.sum()); u = rng.random((n, 2)); f = u.sum(1) > 1; u[f] = 1 - u[f]
    return v[i, 0] + u[:, :1] * (v[i, 1] - v[i, 0]) + u[:, 1:] * (v[i, 2] - v[i, 0])


P = {k: stl_pts(k) for k in ["backpack", "top_lid", "bottom_cap", "wing_left", "nacelle_left"]}
# trunk of the duck as a box (CAD frame): present for the screws driven with the frame on the robot
tr = np.array([[C.TRUNK_X0_MM, C.TRUNK_X0_MM + 110.0], [-C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2], [C.TRUNK_BOTTOM_Z_MM, C.ROOF_Z_MM]])
P["trunk"] = tr[:, 0] + rng.random((200000, 3)) * (tr[:, 1] - tr[:, 0])


F = cadres["fans_left"]; _n = 150000; _a = rng.random(_n) * 2 * math.pi
P["fans"] = np.vstack([np.c_[x + F["r"] * np.cos(_a), F["y"] + F["r"] * np.sin(_a), F["z0"] + rng.random(_n) * (F["z1"] - F["z0"])] for x in F["x"]])


def free_len(p0, d, pts, start):
    """Free length of a shank of radius R_TOOL leaving p0 along d; the first `start` mm (inside the counterbore) are ignored."""
    r = pts - p0; s = r @ d; rad = np.linalg.norm(r - np.outer(s, d), axis=1)
    hit = s[(s > start) & (rad < R_TOOL)]
    return min(hit.min(), L_MAX) if hit.size else L_MAX


def best(p0, axis, pts, start=3.0):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = np.cross(axis, [1.0, 0.3, 0.2]); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    out = [(0.0, 0.0, free_len(np.asarray(p0, float), axis, pts, start))]
    for tilt in (5, 10, 15, 20, 25):
        for az in np.arange(0, 360, 15):
            d = math.cos(math.radians(tilt)) * axis + math.sin(math.radians(tilt)) * (math.cos(math.radians(az)) * e1 + math.sin(math.radians(az)) * e2)
            out.append((tilt, az, free_len(np.asarray(p0, float), d, pts, start + 2.0 * math.tan(math.radians(tilt)) * R_HEAD + 1.0)))
    return out


rows = []
for s in G:
    pts = np.vstack([P[k] for k in s["present"]])
    o = best(s["head"], s["out"], pts)
    ok = [x for x in o if x[2] >= L_NEED]
    first = min(ok, key=lambda x: x[0]) if ok else max(o, key=lambda x: x[2])
    rows.append(dict(name=s["name"], straight_mm=round(o[0][2], 1), tilt_deg=first[0], free_mm=round(first[2], 1), ok=bool(ok), curve=[max(x[2] for x in o if x[0] == t) for t in (0, 5, 10, 15, 20, 25)]))
print(f"shank r {R_TOOL} mm, need {L_NEED:.0f} mm free, ball-end tilt <= {TILT_MAX:.0f} deg, cut-off {L_MAX:.0f} mm")
print(f"{'screw':28s} {'straight mm':>11s} {'min tilt deg':>12s} {'free mm':>8s}  ok")
for r in rows:
    print(f"{r['name']:28s} {r['straight_mm']:11.1f} {r['tilt_deg']:12.0f} {r['free_mm']:8.1f}  {'yes' if r['ok'] else 'NO'}")
json.dump(dict(r_tool_mm=R_TOOL, need_mm=L_NEED, tilt_max_deg=TILT_MAX, screws=[{k: v for k, v in r.items() if k != "curve"} for r in rows]), open(os.path.join(ROOT, "docs", "flight_tool_clearance.json"), "w"), indent=1)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for r in rows:
    ax.plot([0, 5, 10, 15, 20, 25], r["curve"], marker="o", ms=3, lw=1, label=r["name"])
ax.axhline(L_NEED, color="k", ls="--", lw=0.8); ax.text(0.3, L_NEED + 2, f"needed: {L_NEED:.0f} mm", fontsize=7)
ax.set_xlabel("driver tilt from the screw axis [deg]"); ax.set_ylabel("free length for the driver [mm]"); ax.legend(fontsize=6, ncol=2, frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_tool_clearance.pdf"))
