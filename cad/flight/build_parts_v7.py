#!/usr/bin/env python3
"""Printed parts of the flying-duck frame, v6: backpack, four flat wing trusses and two nacelles (CadQuery).

    .venv-cad/bin/python cad/flight/build_parts.py      # after scripts/tools/flight_design.py and build_edf.py

History: v3 (build_parts_v3.py) crossed the roof behind the neck and blocked the head at stand; v4 (build_parts_v4.py)
put a hoop in the gap between trunk and thighs (47 % of leg poses blocked, 5 Hz first mode); v5 carried each nacelle on
one swept arm: no triangulation, and it needed supports to print. v6 keeps v5's backpack and nacelles and replaces the
arms with trusses.

Parts (duck standing, world frame, millimetres, x forward, y left, z up):
  backpack            rounded box on the trunk's rear face: flight pack upright, FC/ESC stack above it, hook over the
                      roof's rear edge. The four wing roots bolt onto its side walls from inside the bay.
  bottom_cap          closes the bay from below and carries the hook under the trunk; two screws from underneath.
  wing_rear_l/r       flat triangulated truss, root post on the backpack's side, tip lug under the nacelle's rear fan.
  wing_front_l/r      the same, swept forward to the front fan: with the rear wing it makes a V in plan, so the
                      nacelle is held at two points and the structure is triangulated in the lift plane and in plan.
                      All four print flat (9 mm thick), no supports, layers in the plane of the loads.
  nacelle_left/right  stadium-shaped pod around two fans with two inboard lugs. Each fan slides in from below against
                      an internal shoulder (thrust in compression); the nacelle's own lip is the bell-mouth inlet.
  fan_collar (x4)     split collar on the shroud under the nacelle: keeps the fan from dropping out.
No fastener is visible from above or from the side. Writes STEP + STL to cad/flight/out/, docs/flight_cad_results.json
(masses, inertia, collision primitives) and docs/figures/flight_frame.pdf.
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
LX, LY, ZM = C.FAN_LX * 1e3, C.FAN_LY * 1e3, C.FAN_Z * 1e3
R_BORE = C.DUCT_OD_MM / 2 + 0.15
R_OUT = R_BORE + C.POD_WALL6_MM
HP = C.POD_H6_MM
POD_TOP = ZM + C.DUCT_LEN_MM / 2 + C.BELL_MM          # the shroud's top sits BELL_MM under the nacelle's top face
POD_BOT = POD_TOP - HP
XR = C.TRUNK_X0_MM                                     # trunk rear face
W = C.PACK_WALL_MM
bx, by, bz = (v * 1e3 for v in C.BATT_SIZE)
BAY_X = (XR - W - 38.0, XR - W)                        # inner bay: 38 deep (pack 34, stack 36)
BAY_Y = 19.5                                           # inner half-width (stack 36 wide)
BAY_Z = (C.BATT_POS[2] * 1e3 - bz / 2 - 0.5, C.FC_POS[2] * 1e3 + 10.0)
BOX = (BAY_X[0] - W, XR, -(BAY_Y + W), BAY_Y + W, BAY_Z[0], BAY_Z[1] + W)   # outer backpack box (open at the bottom)
M3_CLEAR, INSERT_D, INSERT_L = 3.4, 4.0, 5.5
YB = BAY_Y + W                                                               # backpack outer half-width
WT, WB = C.WING_T_MM, C.WING_BAR_MM
FANS_L = [(XC + LX, LY, +1), (XC - LX, LY, -1)]                              # x, y, yaw-cant sign


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl(r, p0, d, length):
    d = cq.Vector(*d).normalized()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, length, cq.Vector(*p0) - d * (length / 2), d))


TAB_X = (XC - LX - R_BORE - C.TAB_R_MM + 5.0, XC + LX + R_BORE + C.TAB_R_MM - 5.0)       # tail / nose tab centres, on the nacelle's centre line
X_REAR = BOX[0] - WT / 2 + 0.3                                                             # rear wings lie on the backpack's rear face


def wing(kind, side=1):
    """A flat triangulated truss from a root post on the backpack to a seat under one end of the nacelle.

    Built in its own vertical plane (s outward along the plan direction, z up) from rounded bars: top chord, bottom
    chord, root post, two web members (two triangles) and a horizontal seat that passes under the nacelle's tab on the
    thrust line, with a disc at every node as a gusset. "rear": the plane is the backpack's rear face (the root post
    laps onto it). "front": the root is on the side wall and the plane sweeps forward to the nose tab. Prints flat.
    """
    zs = POD_BOT - WB / 2                                                                  # seat centre line: its top is the nacelle's underside
    if kind == "rear":
        spec = C.WING_REAR7
        o = np.array([X_REAR, spec["root_y"]]); tgt = np.array([X_REAR, LY])
    else:
        spec = C.WING_FRONT7
        o = np.array([spec["root_x"], YB - 0.2]); tgt = np.array([TAB_X[1], LY])
    zb, zt = spec["root_z"]
    d = tgt - o; Lc = float(np.linalg.norm(d)); h = d / Lc
    S = Lc - C.SEAT_MM / 2                                                                 # tip node: where the seat begins
    N = dict(rt=(0.0, zt - WB / 2), rb=(0.0, zb + WB / 2), tip=(S, zs), end=(S + C.SEAT_MM, zs), mt=(S * 0.5, (zt - WB / 2 + zs) / 2), mb=(S * 0.52, (zb + WB / 2 + zs) / 2 - 2.0),
             kn=(S - 14.0, zs - 15.0))
    bars = [("rt", "tip"), ("rb", "kn"), ("kn", "end"), ("rt", "rb"), ("rb", "mt"), ("mt", "mb"), ("kn", "tip"), ("tip", "end")]
    sk = None
    for a_, b_ in bars:
        p, q = np.array(N[a_]), np.array(N[b_]); L = float(np.linalg.norm(q - p)); ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        bar = cq.Workplane("XY").center(*((p + q) / 2)).transformed(rotate=(0, 0, ang)).slot2D(L + WB, WB).extrude(WT)
        sk = bar if sk is None else sk.union(bar)
    for k_, r_ in (("rt", WB * 0.95), ("rb", WB * 0.95), ("tip", WB * 0.85), ("kn", WB * 0.85), ("mt", WB * 0.8), ("mb", WB * 0.7)):
        sk = sk.union(cq.Workplane("XY").center(*N[k_]).circle(r_).extrude(WT))
    post_w = 2 * (YB - spec["root_y"]) if kind == "rear" else WB + 1.0                     # the rear root is a plate lapping the rear face
    sk = sk.union(cq.Workplane("XY").center(post_w / 2 - WB / 2 if kind == "rear" else 0.5, (N["rt"][1] + N["rb"][1]) / 2).rect(post_w, zt - zb).extrude(WT))
    sk = sk.intersect(cq.Workplane("XY").center(S / 2 + 10, 0).rect(S + C.SEAT_MM + 200, 2 * (POD_BOT)).extrude(WT).cut(
        cq.Workplane("XY").center(S / 2 + 10, POD_BOT + 50).rect(S + C.SEAT_MM + 200, 100).extrude(WT)))            # flat top at the nacelle's underside
    sk = sk.translate((0, 0, -WT / 2))
    yaw = math.degrees(math.atan2(h[1], h[0]))
    sk = sk.rotate((0, 0, 0), (1, 0, 0), 90).rotate((0, 0, 0), (0, 0, 1), yaw).translate((float(o[0]), float(o[1]), 0))
    if kind == "front":
        sk = sk.cut(box(-200, 200, -200, YB, 0, 400))                                      # flat root face on the side wall
    seat_c = o + h * (S + C.SEAT_MM / 2)
    screws = [seat_c + h * dv for dv in (-9.0, 9.0)]
    for q in screws:                                                                       # seat screws, from below, counterbored
        sk = sk.cut(cyl(M3_CLEAR / 2, (q[0], q[1], zs), (0, 0, 1), 30)).cut(cyl(3.0, (q[0], q[1], zs - WB / 2), (0, 0, 1), 5.0))
    root_z = (zb + 10.0, zt - 10.0) if zt - zb > 34 else ((zb + zt) / 2,)
    for z in root_z:                                                                       # root inserts: the screws come from inside the bay
        if kind == "rear":
            sk = sk.cut(cyl(INSERT_D / 2, (X_REAR + WT / 2, spec["root_y"] + 5.0, z), (1, 0, 0), 2 * INSERT_L))
        else:
            sk = sk.cut(cyl(INSERT_D / 2, (o[0] + h[0] / max(h[1], 0.3) * 3.0, YB, z), (0, 1, 0), 2 * INSERT_L))
    geo = dict(kind=kind, yaw_deg=yaw, span=S, seat=[float(seat_c[0]), float(seat_c[1])], screws=[[float(v) for v in q] for q in screws], root_z=list(root_z),
               nodes={k_: [float(o[0] + v[0] * h[0]), float(o[1] + v[0] * h[1]), float(v[1])] for k_, v in N.items()}, bars=bars)
    return (sk if side == 1 else sk.mirror("XZ")), geo


def backpack():
    x0, x1, y0, y1, z0, z1 = BOX
    s = box(x0, x1, y0, y1, z0, z1).edges("|X or |Y or |Z").edges(cq.selectors.BoxSelector((x0 - 1, y0 - 1, z0 + 5), (x1 - 3, y1 + 1, z1 + 1))).fillet(C.PACK_FILLET_MM)
    # hook over the roof's rear edge and two thin fingers on the rear corners (location only)
    s = s.union(box(XR - 0.1, XR + 12.0, -24.0, 24.0, C.ROOF_Z_MM + 0.4, C.ROOF_Z_MM + 2.9).edges("|Z").fillet(4.0))
    for side in (1, -1):
        yy = sorted((side * (C.TRUNK_WID_MM / 2 + 0.4), side * (C.TRUNK_WID_MM / 2 + 2.4)))
        s = s.union(box(XR - 0.1, XR + 9.0, yy[0], yy[1], 132.0, 158.0))
        s = s.union(box(XR - 5.0, XR, min(side * YB, side * (C.TRUNK_WID_MM / 2 + 2.4)), max(side * YB, side * (C.TRUNK_WID_MM / 2 + 2.4)), 128.0, 162.0))
    s = s.cut(box(BAY_X[0], BAY_X[1], -BAY_Y, BAY_Y, BAY_Z[0] - 1.0, BAY_Z[1]))                # the bay, open at the bottom
    s = s.cut(box(XR, XR + 200, -C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2, C.TRUNK_BOTTOM_Z_MM, C.ROOF_Z_MM + 0.4))   # trunk volume
    for side in (1, -1):
        for z in GEO["rear"]["root_z"]:                                                        # wing-root screws, heads inside the bay
            s = s.cut(cyl(M3_CLEAR / 2, (BAY_X[0] - W / 2, side * (C.WING_REAR7["root_y"] + 5.0), z), (1, 0, 0), 3 * W))
        for z in GEO["front"]["root_z"]:
            s = s.cut(cyl(M3_CLEAR / 2, (C.WING_FRONT7["root_x"] + 2.5, side * (BAY_Y + W / 2), z), (0, 1, 0), 3 * W))
        s = s.cut(cyl(INSERT_D / 2, (BAY_X[0] - W / 2 + 4.5, side * (BAY_Y - 4.0), BAY_Z[0]), (0, 0, 1), 2 * INSERT_L + 2))     # bottom-cap inserts
        s = s.cut(cyl(3.0, (BAY_X[0] + 12.0, side * (BAY_Y + W / 2), C.WING_REAR7["root_z"][1] - 24.0), (0, 1, 0), 3 * W))  # motor-wire grommet
    return s


def bottom_cap():
    x0, x1, y0, y1, z0, _ = BOX
    s = box(x0, x1, y0, y1, z0 - 3.0, z0).edges("|Z").fillet(C.PACK_FILLET_MM).faces("<Z").edges().fillet(1.2)
    s = s.union(box(BAY_X[0] + 0.3, BAY_X[1] - 0.3, -BAY_Y + 0.3, BAY_Y - 0.3, z0, z0 + 4.0))                # plug into the bay
    s = s.union(box(XR - 0.1, XR + 10.0, -20.0, 20.0, C.TRUNK_BOTTOM_Z_MM - 3.0, C.TRUNK_BOTTOM_Z_MM - 0.4).edges("|Z").fillet(3.0))  # hook under the trunk
    s = s.union(box(XR - 8.0, XR + 0.5, -20.0, 20.0, z0 - 3.0, C.TRUNK_BOTTOM_Z_MM - 0.4)) if C.TRUNK_BOTTOM_Z_MM - 0.4 > z0 else s
    for side in (1, -1):
        p = (BAY_X[0] - W / 2 + 4.5, side * (BAY_Y - 4.0))
        s = s.cut(cyl(M3_CLEAR / 2, (p[0], p[1], z0), (0, 0, 1), 30)).cut(cyl(3.0, (p[0], p[1], z0 - 3.0), (0, 0, 1), 3.6))
    return s


def nacelle_left():
    c0, c1 = (XC - LX, LY), (XC + LX, LY)
    outline = cq.Workplane("XY").workplane(offset=POD_BOT).center(XC, LY).slot2D(2 * LX + 2 * R_OUT, 2 * R_OUT)
    s = outline.extrude(HP).edges("|Z").fillet(R_OUT - 0.5) if False else outline.extrude(HP)
    s = s.faces(">Z or <Z").edges().fillet(C.POD_FILLET_MM)
    for xf, yf, sgn in FANS_L:
        rad = np.array([xf - XC, yf, 0.0]); rad /= np.linalg.norm(rad)
        g = math.radians(C.FAN_CANT_DEG)
        ax = np.array([0, 0, 1]) * math.cos(g) + sgn * np.cross([0, 0, 1], rad) * math.sin(g)
        top = np.array([xf, yf, ZM]) + ax * (C.DUCT_LEN_MM / 2)                     # shoulder: the shroud's top face
        s = s.cut(cyl(R_BORE, tuple(top - ax * 30.0), ax, 60.0))                     # shroud bore, from below up to the shoulder
        s = s.cut(cyl(25.3, tuple(top + ax * 5.0), ax, 12.0))                        # air path through the shoulder
        # bell-mouth: a rounded lip from the air path out to the nacelle's top face
        prof = cq.Workplane("XZ").moveTo(25.3, -0.2).lineTo(25.3 + C.BELL_MM + 2.0, C.BELL_MM + 2.5).lineTo(25.3 + C.BELL_MM + 2.0 + 20, C.BELL_MM + 2.5).lineTo(25.3 + 20, -0.2).close()
        lip = cq.Workplane("XZ").moveTo(25.3, 0.3).threePointArc((25.3 + 1.3, C.BELL_MM * 0.75), (25.3 + C.BELL_MM + 2.2, C.BELL_MM + 0.3)).lineTo(25.3 + C.BELL_MM + 2.2, C.BELL_MM + 3.0).lineTo(25.3, C.BELL_MM + 3.0).close().revolve(360, (0, 0, 0), (0, 1, 0))
        zax = cq.Vector(0, 0, 1); a3 = cq.Vector(*ax)
        rot_axis = zax.cross(a3); ang = math.degrees(math.acos(max(-1.0, min(1.0, zax.dot(a3)))))
        lip = lip.rotate((0, 0, 0), tuple(rot_axis.toTuple()) if rot_axis.Length > 1e-9 else (1, 0, 0), ang).translate(tuple(top))
        s = s.cut(lip)
    return s


def nacelle_with_lugs():
    """The nacelle with a tab under its tail and its nose, on the centre line: the wings' seats screw up into them."""
    s = nacelle_left()
    for tx, g_ in ((TAB_X[0], GEO["rear"]), (TAB_X[1], GEO["front"])):
        tab = cq.Workplane("XY").workplane(offset=POD_BOT).center(tx, LY).circle(C.TAB_R_MM).extrude(9.0).faces(">Z").edges().fillet(2.0)
        s = s.union(tab)
    for xf, yf, sgn in FANS_L:                                                         # re-open the bores the tabs touched
        rad = np.array([xf - XC, yf, 0.0]); rad /= np.linalg.norm(rad)
        g = math.radians(C.FAN_CANT_DEG)
        ax = np.array([0, 0, 1]) * math.cos(g) + sgn * np.cross([0, 0, 1], rad) * math.sin(g)
        top = np.array([xf, yf, ZM]) + ax * (C.DUCT_LEN_MM / 2)
        s = s.cut(cyl(R_BORE, tuple(top - ax * 30.0), ax, 60.0))
    for g_ in (GEO["rear"], GEO["front"]):
        for q in g_["screws"]:
            s = s.cut(cyl(INSERT_D / 2, (q[0], q[1], POD_BOT), (0, 0, 1), 2 * INSERT_L))  # inserts for the seat screws, from below
    return s


def fan_collar():
    """Split collar on the shroud under the nacelle; one M2.5 pinch screw, placed on the inboard side at assembly."""
    r0, r1, h = C.DUCT_OD_MM / 2 + 0.1, C.DUCT_OD_MM / 2 + 3.2, 6.0
    s = cq.Workplane("XY").circle(r1).circle(r0).extrude(h).faces(">Z or <Z").edges("%CIRCLE").edges(cq.selectors.RadiusNthSelector(1)).fillet(1.2)
    s = s.union(box(r0 + 0.5, r1 + 6.0, -5.0, 5.0, 0, h).edges("|Z").edges(">X").fillet(2.0))
    s = s.cut(cq.Workplane("XY").circle(r0).extrude(h))
    s = s.cut(box(r0 - 1.0, r1 + 7.0, -0.7, 0.7, -1, h + 1))
    s = s.cut(cyl(1.35, (r1 + 2.2, 0, h / 2), (0, 1, 0), 14.0))
    return s


def print_mass(solid, rho, infill=None):
    v, a = solid.val().Volume(), solid.val().Area()
    skin = min(a * C.PRINT_SHELL_MM, v)
    return rho * 1e-6 * (skin + (C.PRINT_INFILL if infill is None else infill) * (v - skin))


wr, geo_r = wing("rear"); wf, geo_f = wing("front")
GEO = dict(rear=geo_r, front=geo_f)
bp = backpack(); cap = bottom_cap(); nl = nacelle_with_lugs(); nr = nl.mirror("XZ"); col_part = fan_collar()
parts = {"backpack": bp, "bottom_cap": cap, "wing_rear_left": wr, "wing_rear_right": wr.mirror("XZ"), "wing_front_left": wf, "wing_front_right": wf.mirror("XZ"),
         "nacelle_left": nl, "nacelle_right": nr, "fan_collar": col_part}
QTY = {"fan_collar": 4}

res = {"parts": {}, "frame_mass_g": {}, "version": "v7 backpack + wing trusses carrying the nacelles at nose and tail"}
print(f"{'part':15s} {'qty':>3s} {'bbox mm':>22s} {'vol cm3':>8s}   PLA 15% / 25% [g]   ABS 15% [g]")
for name, p in parts.items():
    cq.exporters.export(p, os.path.join(OUT, name + ".step"))
    cq.exporters.export(p, os.path.join(OUT, name + ".stl"), tolerance=0.06, angularTolerance=0.25)
    bb = p.val().BoundingBox(); cm = p.val().Center(); inertia = cq.Shape.matrixOfInertia(p.val())
    masses = {"PLA": print_mass(p, C.MATERIALS["PLA"]["rho"], 0.15), "PLA25": print_mass(p, C.MATERIALS["PLA"]["rho"], 0.25), "ABS": print_mass(p, C.MATERIALS["ABS"]["rho"], 0.15)}
    res["parts"][name] = dict(qty=QTY.get(name, 1), bbox_mm=[bb.xlen, bb.ylen, bb.zlen], bbox_min=[bb.xmin, bb.ymin, bb.zmin], volume_cm3=p.val().Volume() / 1e3, area_cm2=p.val().Area() / 1e2,
                              mass_g=masses, com_mm=[cm.x, cm.y, cm.z], inertia_unit_mm5=[[inertia[i][j] for j in range(3)] for i in range(3)])
    print(f"{name:15s} {QTY.get(name,1):3d} {bb.xlen:6.0f} x{bb.ylen:5.0f} x{bb.zlen:5.0f} {p.val().Volume()/1e3:8.1f}   {masses['PLA']:6.1f} / {masses['PLA25']:6.1f}        {masses['ABS']:6.1f}")
asm = cq.Assembly()
for name, p in parts.items():
    if name != "fan_collar":
        asm.add(p, name=name)
asm.export(os.path.join(OUT, "flight_frame_assembly.step"))
FAST_G = 14.0   # 4 tip + 6 root + 2 cap screws with inserts, 4 collar screws, pack strap (estimate)
for m in ("PLA", "PLA25", "ABS"):
    res["frame_mass_g"][m] = sum(v["mass_g"][m] * v["qty"] for v in res["parts"].values()) + FAST_G
print("frame incl. %.0f g fasteners: PLA 15 %% %.0f g, PLA 25 %% %.0f g, ABS 15 %% %.0f g" % (FAST_G, res["frame_mass_g"]["PLA"], res["frame_mass_g"]["PLA25"], res["frame_mass_g"]["ABS"]))
res["span_y_mm"] = 2 * (LY + R_OUT); res["length_x_mm"] = TAB_X[1] - TAB_X[0] + 2 * C.TAB_R_MM
res["fix_box_mm"] = [XR - 2.0, XR + 12.5, -24.0, 24.0, C.ROOF_Z_MM - 40.0, C.ROOF_Z_MM + 3.0]      # FE support region: the faces bearing on the trunk
res["pod_z_mm"] = [POD_BOT, POD_TOP]; res["wings"] = dict(rear=geo_r, front=geo_f)
print(f"assembled envelope: {res['span_y_mm']:.0f} mm wide x {res['length_x_mm']:.0f} mm long; nacelles from z = {POD_BOT:.0f} to {POD_TOP:.0f} mm")

# ---- collision primitives (trunk frame, metres) ----------------------------------------------------
T0 = np.array([0, 0, 120.0])
edf = json.load(open(os.path.join(ROOT, "docs", "flight_edf.json")))
bxs = lambda x0, x1, y0, y1, z0, z1, n: dict(name=n, type="box", pos=((np.array([x0 + x1, y0 + y1, z0 + z1]) / 2 - T0) * 1e-3).tolist(),
                                              size=(np.array([x1 - x0, y1 - y0, z1 - z0]) / 2 * 1e-3).tolist())
col = [bxs(BOX[0], BOX[1], BOX[2], BOX[3], BOX[4] - 3.0, BOX[5], "backpack")]
for sy, side in ((1, "left"), (-1, "right")):
    for wname, g_ in (("rear", geo_r), ("front", geo_f)):
        for k, (a_, b_) in enumerate(g_["bars"][:3]):                                  # the chords of each wing; the web lies between them
            a, b = np.array(g_["nodes"][a_]) * [1, sy, 1], np.array(g_["nodes"][b_]) * [1, sy, 1]
            col.append(dict(name=f"wing_{wname}_{side}_{k}", type="capsule", fromto=((a - T0) * 1e-3).tolist() + ((b - T0) * 1e-3).tolist(), radius=WB * 0.6e-3))
    yy = sorted((sy * (LY - R_OUT), sy * (LY + R_OUT)))
    col.append(bxs(TAB_X[0] - C.TAB_R_MM * 0.6, TAB_X[1] + C.TAB_R_MM * 0.6, yy[0], yy[1], POD_BOT, POD_TOP, f"nacelle_{side}"))
res["collision"] = col
res["edf_collision"] = dict(type="cylinder", radius_mm=C.DUCT_OD_MM / 2, z0_mm=-C.DUCT_LEN_MM / 2 - edf["motor_out_mm"], z1_mm=C.DUCT_LEN_MM / 2 - 2.0)   # below / inside the nacelle
json.dump(res, open(os.path.join(ROOT, "docs", "flight_cad_results.json"), "w"), indent=1)

colors = {"backpack": "#d8d8d8", "bottom_cap": "#e9762b", "nacelle_left": "#f2f2f2", "nacelle_right": "#f2f2f2", "wing_rear_left": "#9aa7b5", "wing_rear_right": "#9aa7b5", "wing_front_left": "#c9b27c", "wing_front_right": "#c9b27c"}
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
for ax, ((i, j), lab) in zip(axs, [((1, 0), "top: y, x forward [mm]"), ((0, 2), "side: x, z [mm]"), ((1, 2), "rear: y, z [mm]")]):
    for name, p in parts.items():
        if name == "fan_collar":
            continue
        v, tri = p.val().tessellate(0.4)
        v = np.array([[q.x, q.y, q.z] for q in v])
        ax.add_collection(PolyCollection([v[list(t)][:, [i, j]] for t in tri], fc=colors[name], ec="#555", lw=0.05, alpha=0.5))
    lim = {0: [C.TRUNK_X0_MM, C.TRUNK_X0_MM + C.TRUNK_LEN_MM], 1: [-C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2], 2: [C.TRUNK_BOTTOM_Z_MM, C.ROOF_Z_MM]}
    ax.add_patch(plt.Rectangle((lim[i][0], lim[j][0]), lim[i][1] - lim[i][0], lim[j][1] - lim[j][0], fc="none", ec="k", ls="--", lw=0.8))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(lab, fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Frame v6: backpack, rear wings (blue-gray), front wings (tan), nacelles, bottom cap (orange); dashed = trunk", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_frame.pdf"))
print("wrote cad/flight/out/*.step|stl, docs/flight_cad_results.json, docs/figures/flight_frame.pdf")
