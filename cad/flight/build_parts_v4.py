#!/usr/bin/env python3
"""Printed parts of the flying-duck frame (v4: hoop, arms and outboard pods), CadQuery, from scripts/tools/flight_config.py.

    .venv-cad/bin/python cad/flight/build_parts.py      # after scripts/tools/flight_design.py and build_edf.py

Why v4: the v3 spars (build_parts_v3.py) crossed the trunk roof 1.4 mm behind the neck and blocked the head even in the
standing pose (62 % of head/neck poses). Here nothing crosses the roof except one 4 mm hoop strap at the very rear, and
the fans sit outside the swept volumes of the head and of the legs (scripts/tools/flight_clearance.py).

Parts (duck standing, world frame, millimetres, x forward, y left, z up):
  hoop_arm_left / _right   half of a hoop around the trunk at the rear station + the arm that rises from its upper
                           corner outward (and slightly forward) to a saddle under the pod. Mirror pair, bolted together
                           at the top and under the belly: the hoop passes under the trunk, so it carries the lift.
  pod_left / _right        two clamp rings with canted bores in a filled block, bolted on the saddle (2 x M3). Mirror pair.
  pack_tray                cradle for the flight pack lying across the back, low; screwed to the hoop's rear face.
Writes STEP + STL to cad/flight/out/, docs/flight_cad_results.json (masses, inertia, collision primitives)
and docs/figures/flight_frame.pdf.
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

YI = C.TRUNK_WID_MM / 2 + C.COLLAR_FIT_MM            # hoop inner half-width
ZB = C.TRUNK_BOTTOM_Z_MM - C.COLLAR_FIT_MM           # hoop inner bottom
ZT = C.ROOF_Z_MM + C.COLLAR_FIT_MM                   # hoop inner top
HX0, HX1 = C.HOOP_X_MM
HT = C.HOOP_T_MM
H = C.POD_H_MM
ZM = C.FAN_Z * 1e3                                    # clamp-ring mid-plane
Z0 = ZM - H / 2
R_BORE = C.DUCT_OD_MM / 2 + 0.15
R_RING = R_BORE + C.RING_WALL_MM
LX, LY = C.FAN_LX * 1e3, C.FAN_LY * 1e3
M3_CLEAR, M3_PILOT, INSERT_D, INSERT_L = 3.4, 2.6, 4.0, 5.5
SAD_X = (C.ARM_TOP_X_MM - 15.0, C.ARM_TOP_X_MM + 15.0)   # saddle under the pod: 30 mm along x, two bolts 20 mm apart
SAD_Y = (LY - R_RING - 3.0, LY - R_RING + 15.0)
SAD_Z = (Z0 - 12.0, Z0)
FANS_L = [(XC + LX, LY, +1, +1), (XC - LX, LY, -1, -1)]  # x, y, yaw-cant sign, pinch-ear side


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl(r, p0, d, length):
    d = cq.Vector(*d).normalized()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, length, cq.Vector(*p0) - d * (length / 2), d))


def bar(a, b, thick, depth0, depth1):
    """Tapered bar from a to b: `thick` along the horizontal normal of the vertical plane through a-b, depth in that plane."""
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u)       # horizontal, across the bar
    w = np.cross(u, d)                                           # in the vertical plane, across the bar
    pts = [[p + su * u * thick / 2 + sw * w * dep / 2 for su, sw in ((1, 1), (-1, 1), (-1, -1), (1, -1))] for p, dep in ((a, depth0), (b, depth1))]
    mk = lambda q: cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*v) for v in q], close=True))
    faces = [mk(pts[0][::-1]), mk(pts[1])] + [mk([pts[0][i], pts[0][(i + 1) % 4], pts[1][(i + 1) % 4], pts[1][i]]) for i in range(4)]
    return cq.Workplane("XY").add(cq.Solid.makeSolid(cq.Shell.makeShell(faces)))


def hoop_arm_left():
    # half hoop (y >= 0): roof strap, side wall, belly strap, with bolt lugs on the centre plane
    s = box(HX0, HX1, 0, YI + HT, ZT, ZT + HT)                                     # roof strap
    s = s.union(box(HX0, HX1, YI, YI + HT, ZB - HT, ZT + HT))                      # side wall
    s = s.union(box(HX0, HX1, 0, YI + HT, ZB - HT, ZB))                            # belly strap: carries the lift
    s = s.union(box(HX0, HX1, 0, 5.0, ZT + HT, ZT + HT + 9.0))                     # top lug
    s = s.union(box(HX0, HX1, 0, 5.0, ZB - HT - 9.0, ZB - HT))                     # bottom lug
    # arm: from the hoop's side, high, outward and up (and forward) to the saddle under the pod
    a = ((HX0 + HX1) / 2, YI + HT - 1.0, C.ARM_ROOT_Z_MM)
    b = (C.ARM_TOP_X_MM, SAD_Y[0] + 3.0, SAD_Z[0] + 6.0)
    s = s.union(bar(a, b, C.ARM_T_MM, C.ARM_D_MM, C.ARM_D_MM - 6.0))
    s = s.union(box(HX0, HX1, YI, YI + HT + 9.0, C.ARM_ROOT_Z_MM - 16.0, C.ARM_ROOT_Z_MM + 12.0))   # root block on the side wall
    s = s.union(box(SAD_X[0], SAD_X[1], SAD_Y[0], SAD_Y[1], SAD_Z[0], SAD_Z[1]))                     # saddle
    s = s.cut(box(HX0 - 30, HX1 + 30, -1, YI, ZB, ZT))                                               # keep the trunk opening clean
    for z in (ZT + HT + 4.5, ZB - HT - 4.5):                                                         # hoop clamp bolts (along y)
        s = s.cut(cyl(M3_CLEAR / 2, ((HX0 + HX1) / 2, 0, z), (0, 1, 0), 30))
    for x in (C.ARM_TOP_X_MM - 10.0, C.ARM_TOP_X_MM + 10.0):                                         # pod bolts: inserts in the saddle
        s = s.cut(cyl(INSERT_D / 2, (x, SAD_Y[0] + 9.0, SAD_Z[1]), (0, 0, 1), 2 * INSERT_L))
    s = s.cut(cyl(M3_PILOT / 2, (HX0, YI + HT / 2 + 1.0, C.TRUNK_BOTTOM_Z_MM + 26.0), (1, 0, 0), 14))   # tray screw pilot, rear face
    return s


def pod_left():
    s = box(XC - LX, XC + LX, LY - R_RING, LY + R_RING, Z0, Z0 + H)                 # filled block between the rings
    for xf, yf, _, e in FANS_L:
        s = s.union(cq.Workplane("XY").workplane(offset=Z0).center(xf, yf).circle(R_RING).extrude(H))
        x0, x1 = sorted((xf + e * (R_RING - 2), xf + e * (R_RING + 9)))
        s = s.union(box(x0, x1, yf - 7, yf + 7, Z0, Z0 + H))                         # pinch ears, fore / aft
    s = s.union(box(SAD_X[0], SAD_X[1], SAD_Y[0], LY - R_RING + 1.0, Z0, Z0 + H))    # flange over the saddle
    for xf, yf, sgn, e in FANS_L:
        rad = np.array([xf - XC, yf, 0.0]); rad /= np.linalg.norm(rad)
        g = math.radians(C.FAN_CANT_DEG)
        axis = np.array([0, 0, 1]) * math.cos(g) + sgn * np.cross([0, 0, 1], rad) * math.sin(g)
        s = s.cut(cyl(R_BORE, (xf, yf, ZM), axis, 80))
        x0, x1 = sorted((xf + e * (R_BORE - 1), xf + e * (R_RING + 10)))
        s = s.cut(box(x0, x1, yf - 1, yf + 1, Z0 - 1, Z0 + H + 1))                   # pinch slit
        s = s.cut(cyl(M3_CLEAR / 2, (xf + e * (R_RING + 4.5), yf, ZM), (0, 1, 0), 30))
    for x in (C.ARM_TOP_X_MM - 10.0, C.ARM_TOP_X_MM + 10.0):
        s = s.cut(cyl(M3_CLEAR / 2, (x, SAD_Y[0] + 9.0, ZM), (0, 0, 1), 40))
    for x in (XC - 14.0, XC + 14.0):                                                 # lightening in the block, clear of the bolts
        s = s.cut(cyl(7.0, (x, LY + 6.0, ZM), (0, 0, 1), 40))
    return s


def pack_tray():
    bx, by, bz = (v * 1e3 for v in C.BATT_SIZE)
    cx, cz = C.BATT_POS[0] * 1e3, C.BATT_POS[2] * 1e3
    xr = HX0                                       # the hoop's rear face
    s = box(xr - 2.0, xr, -(YI + HT + 2), YI + HT + 2, cz - bz / 2 - 2.4, cz + bz / 2 + 14)       # back plate against the hoop
    s = s.union(box(cx - bx / 2 - 3.0, xr, -by / 2 - 2.4, by / 2 + 2.4, cz - bz / 2 - 2.4, cz - bz / 2))   # shelf
    s = s.union(box(cx - bx / 2 - 3.0, cx - bx / 2 - 1.0, -by / 2 - 2.4, by / 2 + 2.4, cz - bz / 2, cz - bz / 2 + 9))  # rear lip
    for sy in (1, -1):
        y0, y1 = sorted((sy * (by / 2 + 0.4), sy * (by / 2 + 2.4)))
        s = s.union(box(cx - bx / 2 - 3.0, xr, y0, y1, cz - bz / 2, cz - bz / 2 + 12))            # end cheeks
        s = s.cut(cyl(M3_CLEAR / 2, (xr - 1, sy * (YI + HT / 2 + 1.0), C.TRUNK_BOTTOM_Z_MM + 26.0), (1, 0, 0), 12))
    for y in (-22, 0, 22):
        s = s.cut(cyl(6.0, (xr - 1, y, cz + 4), (1, 0, 0), 8))
    for y in (-26, 26):                                                                          # strap slots in the shelf
        s = s.cut(box(cx - 12, cx + 12, y - 1.8, y + 1.8, cz - bz / 2 - 3, cz - bz / 2 + 0.5))
    return s


def print_mass(solid, rho):
    v, a = solid.val().Volume(), solid.val().Area()
    skin = min(a * C.PRINT_SHELL_MM, v)
    return rho * 1e-6 * (skin + C.PRINT_INFILL * (v - skin))


hl = hoop_arm_left(); hr = hl.mirror("XZ")
pl = pod_left(); pr = pl.mirror("XZ")
tray = pack_tray()
parts = {"hoop_arm_left": hl, "hoop_arm_right": hr, "pod_left": pl, "pod_right": pr, "pack_tray": tray}

res = {"parts": {}, "frame_mass_g": {}, "version": "v4 hoop + arms + outboard pods"}
print(f"{'part':16s} {'bbox mm':>22s} {'vol cm3':>8s} " + " ".join(f"{m+' g':>7s}" for m in C.MATERIALS))
for name, p in parts.items():
    cq.exporters.export(p, os.path.join(OUT, name + ".step"))
    cq.exporters.export(p, os.path.join(OUT, name + ".stl"), tolerance=0.05, angularTolerance=0.2)
    bb = p.val().BoundingBox()
    cm = p.val().Center()
    inertia = cq.Shape.matrixOfInertia(p.val())
    masses = {m: print_mass(p, d["rho"]) for m, d in C.MATERIALS.items()}
    res["parts"][name] = dict(bbox_mm=[bb.xlen, bb.ylen, bb.zlen], bbox_min=[bb.xmin, bb.ymin, bb.zmin], volume_cm3=p.val().Volume() / 1e3, mass_g=masses,
                              com_mm=[cm.x, cm.y, cm.z], inertia_unit_mm5=[[inertia[i][j] for j in range(3)] for i in range(3)],
                              solid_mass_abs_g=p.val().Volume() * 1e-6 * C.MATERIALS["ABS"]["rho"])
    print(f"{name:16s} {bb.xlen:6.0f} x{bb.ylen:5.0f} x{bb.zlen:5.0f} {p.val().Volume()/1e3:8.1f} " + " ".join(f"{masses[m]:7.1f}" for m in C.MATERIALS))
asm = cq.Assembly()
for name, p in parts.items():
    asm.add(p, name=name)
asm.export(os.path.join(OUT, "flight_frame_assembly.step"))
FAST_G = 14.0   # 2 hoop bolts M3x16 + nuts, 4 pod bolts M3x25, 4 pinch bolts + nuts, 4 inserts, 2 tray screws, pack strap (estimate)
for m in C.MATERIALS:
    res["frame_mass_g"][m] = sum(v["mass_g"][m] for v in res["parts"].values()) + FAST_G
res["frame_mass_solid_abs_g"] = sum(v["solid_mass_abs_g"] for v in res["parts"].values()) + FAST_G
print(f"frame incl. {FAST_G:.0f} g fasteners/strap: " + ", ".join(f"{m} {v:.0f} g" for m, v in res["frame_mass_g"].items()))
res["ring_wall_min_mm"] = C.RING_WALL_MM - H / 2 * math.tan(math.radians(C.FAN_CANT_DEG))
res["span_y_mm"] = 2 * (LY + R_RING)
res["length_x_mm"] = 2 * (LX + R_RING + 9)
res["fix_box_mm"] = [HX0, HX1, -YI - HT, YI + HT, ZB - HT, ZB]          # belly strap: where the trunk bears on the hoop (FE support)
print(f"assembled envelope: {res['span_y_mm']:.0f} mm wide x {res['length_x_mm']:.0f} mm long; pods {ZM-120:.0f} mm above the trunk origin")

# ---- collision primitives (trunk frame, metres): boxes for the frame and pack, a capsule per arm; fans are cylinders ----
T0 = np.array([0, 0, 120.0])
edf = json.load(open(os.path.join(ROOT, "docs", "flight_edf.json")))
col = []
bxs = lambda x0, x1, y0, y1, z0, z1, n: dict(name=n, type="box", pos=((np.array([x0 + x1, y0 + y1, z0 + z1]) / 2 - T0) * 1e-3).tolist(),
                                              size=(np.array([x1 - x0, y1 - y0, z1 - z0]) / 2 * 1e-3).tolist())
for sy, side in ((1, "left"), (-1, "right")):
    yy = lambda y0, y1: sorted((sy * y0, sy * y1))
    col.append(bxs(HX0, HX1, *yy(YI, YI + HT), ZB - HT, ZT + HT, f"hoop_side_{side}"))
    a = np.array([(HX0 + HX1) / 2, YI + HT - 1.0, C.ARM_ROOT_Z_MM]); b = np.array([C.ARM_TOP_X_MM, SAD_Y[0] + 3.0, SAD_Z[0] + 6.0])
    a[1] *= sy; b[1] *= sy
    col.append(dict(name=f"arm_{side}", type="capsule", fromto=((a - T0) * 1e-3).tolist() + ((b - T0) * 1e-3).tolist(), radius=C.ARM_T_MM / 2 * 1e-3))
    col.append(bxs(XC - LX, XC + LX, *yy(LY - R_RING, LY + R_RING), Z0, Z0 + H, f"pod_block_{side}"))
    col.append(bxs(SAD_X[0], SAD_X[1], *yy(SAD_Y[0], SAD_Y[1]), SAD_Z[0], SAD_Z[1], f"saddle_{side}"))
col.append(bxs(HX0, HX1, -YI - HT, YI + HT, ZT, ZT + HT, "hoop_roof"))
col.append(bxs(HX0, HX1, -YI - HT, YI + HT, ZB - HT, ZB, "hoop_belly"))
bx, by, bz = (v * 1e3 for v in C.BATT_SIZE); bp = np.array(C.BATT_POS) * 1e3
col.append(bxs(bp[0] - bx / 2 - 3, HX0, -by / 2 - 2.4, by / 2 + 2.4, bp[2] - bz / 2 - 2.4, bp[2] + bz / 2, "pack_and_tray"))
fp = np.array(C.FC_POS) * 1e3
col.append(bxs(fp[0] - 18, fp[0] + 18, -18, 18, fp[2] - 8, fp[2] + 8, "fc_esc_stack"))
res["collision"] = col
res["edf_collision"] = edf["collision"]
json.dump(res, open(os.path.join(ROOT, "docs", "flight_cad_results.json"), "w"), indent=1)

colors = {"hoop_arm_left": "#d1603d", "hoop_arm_right": "#e9a03b", "pod_left": "#2a6f97", "pod_right": "#2a6f97", "pack_tray": "#3a7d44"}
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
views = [((1, 0), "top: y, x forward [mm]"), ((0, 2), "side: x, z [mm]"), ((1, 2), "rear: y, z [mm]")]
for ax, ((i, j), lab) in zip(axs, views):
    for name, p in parts.items():
        v, tri = p.val().tessellate(0.3)
        v = np.array([[q.x, q.y, q.z] for q in v])
        ax.add_collection(PolyCollection([v[list(t)][:, [i, j]] for t in tri], fc=colors[name], ec="none", alpha=0.35))
    lim = {0: [C.TRUNK_X0_MM, C.TRUNK_X0_MM + C.TRUNK_LEN_MM], 1: [-C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2], 2: [C.SHELL_BOTTOM_Z_MM, C.ROOF_Z_MM]}
    ax.add_patch(plt.Rectangle((lim[i][0], lim[j][0]), lim[i][1] - lim[i][0], lim[j][1] - lim[j][0], fc="none", ec="k", ls="--", lw=0.8))
    bp_, bs = np.array(C.BATT_POS) * 1e3, np.array(C.BATT_SIZE) * 1e3
    ax.add_patch(plt.Rectangle((bp_[i] - bs[i] / 2, bp_[j] - bs[j] / 2), bs[i], bs[j], fc="none", ec="#555", ls=":", lw=0.8))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(lab, fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Frame v4: hoop + arm (red, orange), pods (blue), pack tray (green); dashed = trunk shell, dotted = pack", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_frame.pdf"))
print("wrote cad/flight/out/*.step|stl, docs/flight_cad_results.json, docs/figures/flight_frame.pdf")
