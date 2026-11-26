#!/usr/bin/env python3
"""Printed parts of the flying-duck frame (v3, spar frame), generated with CadQuery from scripts/tools/flight_config.py.

    .venv-cad/bin/python cad/flight/build_parts.py      # after scripts/tools/flight_design.py

Parts (duck standing, world frame, millimetres, x forward, y left, z up):
  frame_left / frame_right  twin-ring fan pod + two spars that cross the trunk roof and lap with the other half's
                            spars (4 x M3 through the laps) + a rail beside the shell. The right half is the left
                            one turned 180 deg about the thrust centroid (only the hook screw pilot differs).
  lift_hook (x2, mirrored)  C-clip from the rail top to the underside of the shell: carries the lift
  pack_tray                 flight-pack cradle on the back, held by the two rear lap bolts
The FC + 4-in-1 ESC stack screws into four heat-set inserts in the spar tops (30.5 mm pattern).
Writes STEP + STL to cad/flight/out/, masses to docs/flight_cad_results.json, a drawing to docs/figures/flight_frame.pdf.
"""
import json
import math
import os
import sys

import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools"))
import flight_config as C

OUT = os.path.join(ROOT, "cad", "flight", "out")
os.makedirs(OUT, exist_ok=True)
XC = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))["layout"]["x_c"] * 1e3

H = C.FRAME_H_MM
Z0 = C.ROOF_Z_MM + 1.0
ZM = Z0 + H / 2
assert abs(ZM - C.FAN_Z * 1e3) < 1e-6, "FAN_Z must equal ROOF_Z_MM + 1 + FRAME_H_MM / 2"
YI = C.TRUNK_WID_MM / 2 + C.COLLAR_FIT_MM
R_BORE = C.DUCT_OD_MM / 2 + 0.15
R_RING = R_BORE + C.RING_WALL_MM
LX, LY = C.FAN_LX * 1e3, C.FAN_LY * 1e3
A, W, LAP = C.SPAR_OFFSET_MM, C.SPAR_W_MM, C.LAP_Y_MM
RY0, RY1 = C.RAIL_Y_MM
LAP_BOLT_Y, LAP_BOLT_Z = 22.0, ZM - 3.0
M3_CLEAR, M2_PILOT, INSERT_D, INSERT_L = 3.4, 1.7, 4.0, 5.5
FANS_L = [(XC + LX, LY, +1, +1), (XC - LX, LY, -1, -1)]  # x, y, yaw-cant sign, pinch-ear side (+x / -x)


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl(r, p0, d, length):
    d = cq.Vector(*d).normalized()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, length, cq.Vector(*p0) - d * (length / 2), d))


def half_common():
    """The y > 0 half without trunk-specific features; the other half is this turned 180 deg about (XC, 0)."""
    sf = (XC + A, XC + A + W)        # front spar
    sr = (XC - A, XC - A + W)        # rear spar
    s = box(sf[0], sf[1], -LAP, LY - 12, Z0, Z0 + H).union(box(sr[0], sr[1], -LAP, LY - 12, Z0, Z0 + H))
    s = s.union(box(sr[0], sf[1], RY0, RY1, Z0, Z0 + H))                                  # rail
    # pod block: the cusp between the two rings is filled, so each ring wall is held over the half facing its
    # neighbour instead of hanging off thin struts (first FE pass: the bare rings nodded at 23 Hz)
    s = s.union(box(XC - LX, XC + LX, LY - R_RING, LY + R_RING, Z0, Z0 + H))
    for xf, yf, _, e in FANS_L:
        s = s.union(cq.Workplane("XY").workplane(offset=Z0).center(xf, yf).circle(R_RING).extrude(H))
        x0, x1 = sorted((xf + e * (R_RING - 2), xf + e * (R_RING + 9)))
        s = s.union(box(x0, x1, yf - 7, yf + 7, Z0, Z0 + H))                              # pinch ears, fore / aft
    s = s.union(box(XC - 2.0, XC + 6.0, RY0, RY0 + 2.6, Z0 - 18, Z0))                     # skirt finger on the shell side
    for x0, x1 in (sf, sr):                                                              # gussets under the spars, outboard of the shell
        tri = [(RY1 + 3.5, Z0 + 0.5), (RY1 + 3.5, Z0 - 18), (LY - R_RING + 2, Z0 + 0.5)]
        s = s.union(cq.Workplane("YZ").workplane(offset=x0).polyline(tri).close().extrude(x1 - x0))
    for xf, yf, sgn, e in FANS_L:
        rad = np.array([xf - XC, yf, 0.0]); rad /= np.linalg.norm(rad)
        g = math.radians(C.FAN_CANT_DEG)
        axis = np.array([0, 0, 1]) * math.cos(g) + sgn * np.cross([0, 0, 1], rad) * math.sin(g)
        s = s.cut(cyl(R_BORE, (xf, yf, ZM), axis, 80))
        x0, x1 = sorted((xf + e * (R_BORE - 1), xf + e * (R_RING + 10)))
        s = s.cut(box(x0, x1, yf - 1, yf + 1, Z0 - 1, Z0 + H + 1))                        # pinch slit
        s = s.cut(cyl(M3_CLEAR / 2, (xf + e * (R_RING + 4.5), yf, ZM), (0, 1, 0), 30))    # pinch bolt
    for y in (-LAP_BOLT_Y, LAP_BOLT_Y):                                                   # lap bolts through both spars
        s = s.cut(cyl(M3_CLEAR / 2, (XC, y, LAP_BOLT_Z), (1, 0, 0), 4 * (A + W)))
    for y in (-15.25, 15.25):                                                             # stack inserts, rear spar top
        s = s.cut(cyl(INSERT_D / 2, (XC - 15.25, y, Z0 + H), (0, 0, 1), 2 * INSERT_L))
    return s


def lift_hook():
    ys0 = RY1 + 0.2
    ys1 = ys0 + C.HOOK_STRAP_MM
    zt, zb = Z0 + H + 0.2, C.SHELL_BOTTOM_Z_MM - 0.2
    prof = [(RY0 + 0.3, zt), (RY0 + 0.3, zt + 2.4), (ys1, zt + 2.4), (ys1, zb - 2.6), (C.SHELL_LEDGE_Y_MM, zb - 2.6),
            (C.SHELL_LEDGE_Y_MM, zb), (ys0 - 4, zb), (ys0, zb + 4), (ys0, zt)]
    s = cq.Workplane("YZ").workplane(offset=C.HOOK_X_MM[0]).polyline(prof).close().extrude(C.HOOK_X_MM[1] - C.HOOK_X_MM[0])
    s = s.cut(cyl(2.2 / 2, (sum(C.HOOK_X_MM) / 2, ys0, ZM), (0, 1, 0), 20))
    lever = ys0 - (C.SHELL_LEDGE_Y_MM + YI) / 2
    return s, dict(lever_mm=lever, Z_mm3=(C.HOOK_X_MM[1] - C.HOOK_X_MM[0]) * C.HOOK_STRAP_MM ** 2 / 6)


def pack_tray():
    """C-shaped cradle: lug bolted to the rear spar pair, roof plate, back plate down the trunk's back, shelf."""
    bx, by, bz = (v * 1e3 for v in C.BATT_SIZE)
    w = 26.0
    xl = XC - A - W                         # rear face of the rear spar pair
    xb = C.TRUNK_X0_MM - 1.0                # front face of the back plate, 1 mm behind the shell
    zs = C.BATT_POS[2] * 1e3 - bz / 2       # pack underside
    s = box(xl - 3.0, xl, -w, w, Z0, Z0 + H)                           # lug
    s = s.union(box(xb - 2.0, xl, -w, w, Z0 + H - 2.4, Z0 + H))        # roof plate
    s = s.union(box(xb - 2.0, xb, -w, w, zs - 2.4, Z0 + H))            # back plate
    s = s.union(box(xb - 2.0 - bx - 3.4, xb, -w, w, zs - 2.4, zs))     # shelf
    s = s.union(box(xb - 2.0 - bx - 3.4, xb - 2.0 - bx - 1.0, -w, w, zs, zs + 8))  # end lip
    gus = [(xb, Z0 + H - 2.4), (xb + 8, Z0 + H - 2.4), (xb, Z0 + H - 10.4)]       # roof-plate gusset
    s = s.union(cq.Workplane("XZ").workplane(offset=-w).polyline(gus).close().extrude(2 * w))
    for y in (-LAP_BOLT_Y, LAP_BOLT_Y):
        s = s.cut(cyl(M3_CLEAR / 2, (xl - 1.5, y, LAP_BOLT_Z), (1, 0, 0), 12))
    for z in (zs + 14, zs + 52):                                       # strap slots
        s = s.cut(box(xb - 3, xb + 0.5, -16, 16, z, z + 3.5))
    for z in (zs + 27, zs + 41, Z0 - 12):                              # lightening
        for y in (-11, 0, 11):
            s = s.cut(cyl(4.6, (xb - 1, y, z), (1, 0, 0), 8))
    for y in (-11, 11):
        s = s.cut(cyl(4.6, ((xb + xl) / 2 - 1, y, Z0 + H - 1), (0, 0, 1), 8))
    return s


def print_mass(solid, rho):
    v, a = solid.val().Volume(), solid.val().Area()   # mm^3, mm^2
    skin = min(a * C.PRINT_SHELL_MM, v)
    return rho * 1e-6 * (skin + C.PRINT_INFILL * (v - skin))  # g


common = half_common()
HOOK_XM = sum(C.HOOK_X_MM) / 2
right = common.rotate((XC, 0, 0), (XC, 0, 1), 180).cut(cyl(M2_PILOT / 2, (HOOK_XM, -RY1, ZM), (0, 1, 0), 8))   # hook retention screw
left = common.cut(cyl(M2_PILOT / 2, (HOOK_XM, RY1, ZM), (0, 1, 0), 8))
hook, hook_info = lift_hook()
hook_r = hook.mirror("XZ")
tray = pack_tray()
parts = {"frame_left": left, "frame_right": right, "lift_hook_left": hook, "lift_hook_right": hook_r, "pack_tray": tray}

res = {"parts": {}, "frame_mass_g": {}, "version": "v3 spar frame"}
print(f"{'part':18s} {'bbox mm':>22s} {'vol cm3':>8s} " + " ".join(f"{m+' g':>7s}" for m in C.MATERIALS) + "  solid ABS g")
for name, p in parts.items():
    cq.exporters.export(p, os.path.join(OUT, name + ".step"))
    cq.exporters.export(p, os.path.join(OUT, name + ".stl"), tolerance=0.05, angularTolerance=0.2)
    bb = p.val().BoundingBox()
    dims = (bb.xlen, bb.ylen, bb.zlen)
    masses = {m: print_mass(p, d["rho"]) for m, d in C.MATERIALS.items()}
    cm = p.val().Center()
    inertia = cq.Shape.matrixOfInertia(p.val())          # unit density, about the centre of mass [mm^5]
    res["parts"][name] = dict(bbox_mm=dims, bbox_min=[bb.xmin, bb.ymin, bb.zmin], volume_cm3=p.val().Volume() / 1e3, mass_g=masses,
                              com_mm=[cm.x, cm.y, cm.z], inertia_unit_mm5=[[inertia.Value(i, j) if hasattr(inertia, "Value") else inertia[i - 1][j - 1] for j in (1, 2, 3)] for i in (1, 2, 3)],
                              solid_mass_abs_g=p.val().Volume() * 1e-6 * C.MATERIALS["ABS"]["rho"])
    print(f"{name:18s} {dims[0]:6.0f} x{dims[1]:5.0f} x{dims[2]:5.0f} {p.val().Volume()/1e3:8.1f} "
          + " ".join(f"{masses[m]:7.1f}" for m in C.MATERIALS) + f"  {res['parts'][name]['solid_mass_abs_g']:7.1f}")
asm = cq.Assembly()
for name, p in parts.items():
    asm.add(p, name=name)
asm.export(os.path.join(OUT, "flight_frame_assembly.step"))
FAST_G = 14.0  # 4 lap bolts M3x25 + nuts, 4 pinch bolts + nuts, 4 inserts, 2 M2 self-tappers, pack strap (estimate)
for m in C.MATERIALS:
    res["frame_mass_g"][m] = sum(v["mass_g"][m] for v in res["parts"].values()) + FAST_G
res["frame_mass_solid_abs_g"] = sum(v["solid_mass_abs_g"] for v in res["parts"].values()) + FAST_G
print(f"frame incl. {FAST_G:.0f} g fasteners/strap: " + ", ".join(f"{m} {v:.0f} g" for m, v in res["frame_mass_g"].items())
      + f"  (100 % infill ABS: {res['frame_mass_solid_abs_g']:.0f} g)")

T_all = 4 * C.FANS[C.PRIMARY]["thrust_g"] * 1e-3 * C.G
sig = (T_all / 2) * hook_info["lever_mm"] / hook_info["Z_mm3"]
res["hook"] = dict(load_n=T_all / 2, lever_mm=hook_info["lever_mm"], sigma_mpa=sig,
                   sf={m: d["yield_pa"] / 1e6 / sig for m, d in C.MATERIALS.items()})
print(f"lift hook: {T_all/2:.1f} N each at {hook_info['lever_mm']:.1f} mm lever -> strap bending {sig:.1f} MPa "
      f"(SF {res['hook']['sf']['ABS']:.1f} ABS, {res['hook']['sf']['PLA']:.1f} PLA on yield)")
res["ring_wall_min_mm"] = C.RING_WALL_MM - H / 2 * math.tan(math.radians(C.FAN_CANT_DEG))
res["span_y_mm"] = 2 * (LY + R_RING)
res["length_x_mm"] = 2 * (LX + R_RING + 9)
res["hook_fix_box_mm"] = [C.HOOK_X_MM[0], C.HOOK_X_MM[1], RY0, RY1, Z0 + H]
res["neck_gap_mm"] = 16.0 - (XC + A + W)
print(f"assembled envelope: {res['span_y_mm']:.0f} mm wide x {res['length_x_mm']:.0f} mm long; min ring wall {res['ring_wall_min_mm']:.2f} mm")
json.dump(res, open(os.path.join(ROOT, "docs", "flight_cad_results.json"), "w"), indent=1)

colors = {"frame_left": "#d1603d", "frame_right": "#e9a03b", "lift_hook_left": "#2a6f97", "lift_hook_right": "#2a6f97", "pack_tray": "#3a7d44"}
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
views = [((1, 0), "top: y, x forward [mm]"), ((0, 2), "side: x, z [mm]"), ((1, 2), "rear: y, z [mm]")]
for ax, ((i, j), lab) in zip(axs, views):
    for name, p in parts.items():
        v, tri = p.val().tessellate(0.3)
        v = np.array([[q.x, q.y, q.z] for q in v])
        ax.add_collection(PolyCollection([v[list(t)][:, [i, j]] for t in tri], fc=colors[name], ec="none", alpha=0.35))
    lim = {0: [C.TRUNK_X0_MM, C.TRUNK_X0_MM + C.TRUNK_LEN_MM], 1: [-C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2], 2: [C.SHELL_BOTTOM_Z_MM, C.ROOF_Z_MM]}
    ax.add_patch(plt.Rectangle((lim[i][0], lim[j][0]), lim[i][1] - lim[i][0], lim[j][1] - lim[j][0], fc="none", ec="k", ls="--", lw=0.8))
    bp, bs = np.array(C.BATT_POS) * 1e3, np.array(C.BATT_SIZE) * 1e3
    ax.add_patch(plt.Rectangle((bp[i] - bs[i] / 2, bp[j] - bs[j] / 2), bs[i], bs[j], fc="none", ec="#555", ls=":", lw=0.8))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(lab, fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Spar frame: left half (red), right half (orange), lift hooks (blue), pack tray (green); dashed = trunk shell, dotted = pack", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_frame.pdf"))
print("wrote cad/flight/out/*.step|stl, docs/flight_cad_results.json, docs/figures/flight_frame.pdf")
