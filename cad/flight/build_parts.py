#!/usr/bin/env python3
"""Printed parts of the flying-duck frame, v8: backpack, two sculpted wings and two nacelles (CadQuery).

    .venv-cad/bin/python cad/flight/build_parts.py      # after scripts/tools/flight_design.py and build_edf.py

History: v3 (build_parts_v3.py) crossed the roof behind the neck; v4 (build_parts_v4.py) put a hoop beside the thighs;
v5 hung each nacelle on one thin swept arm; v6/v7 (build_parts_v7.py) used exposed flat trusses, which looked
industrial. v8 keeps the backpack and nacelles and joins them with one closed, sculpted shell per side.

Parts (duck standing, world frame, millimetres, x forward, y left, z up):
  backpack            rounded box on the trunk's rear face: flight pack upright, FC/ESC stack above it, hook over the
                      roof's rear edge. The wing roots bolt onto its side walls from inside the bay.
  bottom_cap          closes the bay from below and carries the hook under the trunk; two screws from underneath.
  wing_left/right     a closed shell lofted through superelliptic sections: deep and narrow at the root, flat and wide
                      where it meets the nacelle's flank, with a tongue under the nacelle (two screws from below).
                      The motor wires run inside it. Stiffness comes from the closed section and its infill.
  nacelle_left/right  stadium-shaped pod around two fans. Each fan slides in from below against an internal shoulder
                      (thrust in compression); the nacelle's own lip is the bell-mouth inlet.
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
BAY_X = (XR - W - C.BAY_MM["depth"], XR - W)           # lower bay, around the pack: 38 deep (pack 34)
BAY_Y = C.BAY_MM["half_w"]
HD = C.BAY_HEAD_MM                                     # upper bay ("head"): 47 deep, 48 wide for the stack (ESC 44 x 45.6)
BAYU_X, BAYU_Y, Z_HEAD = (XR - W - HD["depth"], XR - W), HD["half_w"], HD["z0"]
ST = C.FC_STACK_MM
BAY_Z = (C.BATT_POS[2] * 1e3 - bz / 2 - 0.5, C.FC_POS[2] * 1e3 + ST["h"] / 2 + ST["boss"])
STACK_XY = [((BAYU_X[0] + BAYU_X[1]) / 2 + sx * ST["hole"] / 2, sy * ST["hole"] / 2) for sx in (1, -1) for sy in (1, -1)]
USB_Z = BAY_Z[1] - ST["boss"] - 5.0                    # FC is the board next to the roof; its USB-C faces the rear wall (check on the board)
BOX = (BAY_X[0] - W, XR, -(BAY_Y + W), BAY_Y + W, BAY_Z[0], BAY_Z[1] + W)   # outer box of the lower part (open at the bottom)
WR = 6.5                                                                     # head rear wall: thick enough for the lid's two inserts (no external lugs)
BOXU = (BAYU_X[0] - WR, XR, -(BAYU_Y + W), BAYU_Y + W, Z_HEAD, BAY_Z[1] + W)  # outer box of the head
M3_CLEAR, INSERT_D, INSERT_L = 3.4, 4.0, 5.5
YB = BAYU_Y + W                                                               # backpack outer half-width
WT, WB = C.WING_T_MM, C.WING_BAR_MM
FANS_L = [(XC + LX, LY, +1), (XC - LX, LY, -1)]                              # x, y, yaw-cant sign


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl(r, p0, d, length):
    d = cq.Vector(*d).normalized()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, length, cq.Vector(*p0) - d * (length / 2), d))


Y_ROOT, Y_TIP = YB, LY - R_OUT + 1.5
Z_TIP = POD_BOT + HP / 2
GEO = dict(root_x=C.BLADE_ROOT_MM["x"], root_z=[C.BLADE_ROOT_MM["z"] - 9.0, C.BLADE_ROOT_MM["z"] + 9.0], tongue_y=[LY - 10.0, LY + 10.0])
# side joint (v8.1): the tongue's two screws from below left the wing-nacelle joint with no fastener across the flank.
# Two horizontal M3 screws now go through the wing tip itself into inserts in the nacelle's flank, in the web between
# the fans (8.6 mm of material there). Each head goes down a channel that opens on the wing's top surface.
SJ = C.SIDE_JOINT_MM
Y_FLANK = LY - R_OUT
SIDE_X = [XC + C.BLADE_TIP_MM["dx"] + d for d in SJ["dx"]]
GEO.update(side_land_y=Y_FLANK, side_x=SIDE_X, side_y=Y_FLANK - SJ["grip"], side_z=Z_TIP)


def blade_section(t):
    """Centre (x, y, z) and half-sizes (a along x, b along z) of the wing's section at station t in [0, 1]."""
    r, tp = C.BLADE_ROOT_MM, C.BLADE_TIP_MM
    sm = 3 * t * t - 2 * t ** 3
    y = Y_ROOT - 1.0 + t * (Y_TIP + 1.0 - Y_ROOT)
    x = r["x"] + (XC + tp["dx"] - r["x"]) * t ** 2.4                                   # stay rearward, sweep forward only near the nacelle
    z = r["z"] + (Z_TIP - r["z"]) * sm
    fl = (1 - t) ** 5                                                                      # root fairing: the section swells into the backpack wall
    a = r["a"] * (1 + 0.25 * fl) - 2.0 * math.sin(math.pi * min(t / 0.7, 1.0)) ** 2 * (1 - t) + (tp["a"] - r["a"]) * t ** 2.2   # slim waist, wide tip
    b = tp["b"] + (r["b"] * (1 + 0.08 * fl) - tp["b"]) * (1 - t ** 2.6)
    return np.array([x, y, z]), a, b


def wing(side=1):
    """One sculpted wing: a closed shell lofted through superelliptic sections from the backpack's side wall to the
    nacelle's inboard flank, plus a tongue under the nacelle that takes two screws from below."""
    wires = []
    for t in np.linspace(0, 1, 13):
        c, a_, b_ = blade_section(t)
        th = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        e = 2.0 / (C.BLADE_EXP[0] + (C.BLADE_EXP[1] - C.BLADE_EXP[0]) * t ** 3)
        pts = [cq.Vector(c[0] + a_ * np.sign(math.cos(q)) * abs(math.cos(q)) ** e, c[1], c[2] + b_ * np.sign(math.sin(q)) * abs(math.sin(q)) ** e) for q in th]
        wires.append(cq.Wire.assembleEdges([cq.Edge.makeSpline(pts, periodic=True)]))
    s = cq.Workplane("XY").add(cq.Solid.makeLoft(wires))
    tg = C.TONGUE_MM
    ton = cq.Workplane("XY").workplane(offset=POD_BOT - tg["t"]).center(XC, (Y_TIP - 16.0 + LY + tg["reach"]) / 2).slot2D(LY + tg["reach"] - (Y_TIP - 16.0) + tg["w"], tg["w"], 90).extrude(tg["t"])
    ton = ton.faces("<Z").edges().fillet(1.8)
    s = s.union(ton)
    s = s.cut(box(-300, 300, -300, Y_ROOT, 0, 400))                                        # flat root face on the backpack's side wall
    s = s.cut(cq.Workplane("XY").workplane(offset=POD_BOT).center(XC, LY).slot2D(2 * LX + 2 * R_OUT, 2 * R_OUT).extrude(HP + 20))   # mate with the nacelle's flank
    for x in SIDE_X:                                                                       # side screws: clearance hole, then the head's channel
        s = s.cut(cyl(M3_CLEAR / 2, (x, Y_FLANK, Z_TIP), (0, 1, 0), 2 * SJ["grip"] + 2)).cut(cyl(SJ["bore"], (x, GEO["side_y"] - 25.0, Z_TIP), (0, 1, 0), 50.0))
    for y in GEO["tongue_y"]:                                                              # tongue screws, from below, counterbored
        s = s.cut(cyl(M3_CLEAR / 2, (XC, y, POD_BOT - tg["t"] - 1), (0, 0, 1), 30)).cut(cyl(3.0, (XC, y, POD_BOT - tg["t"] - 1), (0, 0, 1), 3.0))
    for z in GEO["root_z"]:                                                                # root inserts: the screws come from inside the bay
        s = s.cut(cyl(INSERT_D / 2, (GEO["root_x"], Y_ROOT, z), (0, 1, 0), 2 * INSERT_L))
    return s if side == 1 else s.mirror("XZ")


LID_Y, TAB_Y = (-14.0, 14.0), 10.0
LUG_X = BAYU_X[0] - WR / 2


def head_solid():
    u0, u1, v0, v1, w0, w1 = BOXU
    head = box(u0, u1, v0, v1, w0, w1).edges("|X or |Y or |Z").edges(cq.selectors.BoxSelector((u0 - 1, v0 - 1, w0 + 5), (u1 - 3, v1 + 1, w1 + 1))).fillet(C.PACK_FILLET_MM)
    return head.faces("<Z").edges(cq.selectors.BoxSelector((u0 - 1, v0 - 1, w0 - 1), (u1 - 3, v1 + 1, w0 + 1))).chamfer(min(BAYU_Y - BAY_Y, BAY_X[0] - BAYU_X[0]) - 0.3)   # sloped underside


def top_lid():
    """Lid of the head. The FC/ESC stack hangs under it on four bosses (inserts, long screws from below, done on the bench);
    a tab at the front goes into a slot in the front wall, two screws at the rear go down into the backpack's lugs."""
    u0, u1, v0, v1, w0, w1 = BOXU
    zt = BAY_Z[1]
    s = head_solid().intersect(box(u0 - 1, u1 + 1, v0 - 1, v1 + 1, zt, w1 + 1))
    for y in LID_Y:
        s = s.cut(cyl(M3_CLEAR / 2, (LUG_X, y, zt), (0, 0, 1), 30)).cut(cyl(3.0, (LUG_X, y, w1), (0, 0, 1), 3.2))   # head sunk 1.6 mm
    for x, y in STACK_XY:
        s = s.union(cyl(4.0, (x, y, zt - ST["boss"] / 2 + 0.25), (0, 0, 1), ST["boss"] + 0.5))
        s = s.cut(cyl(INSERT_D / 2, (x, y, zt - ST["boss"]), (0, 0, 1), 2 * INSERT_L))
    s = s.union(box(BAYU_X[1] - 4.0, BAYU_X[1] - 0.2, -TAB_Y, TAB_Y, zt - 5.0, zt + 0.5))            # front block
    s = s.union(box(BAYU_X[1] - 0.5, XR - 0.5, -TAB_Y, TAB_Y, zt - 5.0, zt - 2.5))                   # tab into the front wall
    return s


def backpack():
    x0, x1, y0, y1, z0, z1 = BOX
    s = box(x0, x1, y0, y1, z0, z1).edges("|Z").edges(cq.selectors.BoxSelector((x0 - 1, y0 - 1, z0 - 1), (x1 - 3, y1 + 1, z1 + 1))).fillet(C.PACK_FILLET_MM)
    u0, u1, v0, v1, w0, w1 = BOXU
    s = s.union(head_solid())
    # hook over the roof's rear edge and two thin fingers on the rear corners (location only)
    s = s.union(box(XR - 0.1, XR + 12.0, -24.0, 24.0, C.ROOF_Z_MM + 0.4, C.ROOF_Z_MM + 2.9).edges("|Z").fillet(4.0))
    for side in (1, -1):
        yy = sorted((side * (C.TRUNK_WID_MM / 2 + 0.4), side * (C.TRUNK_WID_MM / 2 + 2.4)))
        s = s.union(box(XR - 0.1, XR + 9.0, yy[0], yy[1], 132.0, 158.0))
        s = s.union(box(XR - 5.0, XR, min(side * YB, side * (C.TRUNK_WID_MM / 2 + 2.4)), max(side * YB, side * (C.TRUNK_WID_MM / 2 + 2.4)), 128.0, 162.0))
    s = s.cut(box(BAY_X[0], BAY_X[1], -BAY_Y, BAY_Y, BAY_Z[0] - 1.0, BAY_Z[1]))                # the bay, open at the bottom
    s = s.cut(box(BAYU_X[0], BAYU_X[1], -BAYU_Y, BAYU_Y, Z_HEAD + W, BAY_Z[1]))                # the head's bay
    s = s.cut(box(XR, XR + 200, -C.TRUNK_WID_MM / 2, C.TRUNK_WID_MM / 2, C.TRUNK_BOTTOM_Z_MM, C.ROOF_Z_MM + 0.4))   # trunk volume
    s = s.cut(box(u0 - 1, u1 + 1, v0 - 1, v1 + 1, BAY_Z[1], w1 + 1))                           # open top: closed by top_lid()
    for y in LID_Y:                                                                            # lid screws: inserts in the thick rear wall
        s = s.cut(cyl(INSERT_D / 2, (LUG_X, y, BAY_Z[1]), (0, 0, 1), 2 * INSERT_L))
    s = s.cut(box(BAYU_X[1] - 0.1, XR + 0.1, -TAB_Y - 0.3, TAB_Y + 0.3, BAY_Z[1] - 5.2, BAY_Z[1] - 2.3))   # slot for the lid's front tab
    s = s.cut(box(BAYU_X[0] - WR - 1, BAYU_X[0] + 1, -7.0, 7.0, USB_Z - 4.0, USB_Z + 4.0).edges("|X").fillet(2.0))   # rear slot: USB-C and the UART lead to the duck
    for side in (1, -1):
        for z in GEO["root_z"]:                                                                # wing-root screws, heads inside the bay
            s = s.cut(cyl(M3_CLEAR / 2, (GEO["root_x"], side * (BAYU_Y + W / 2), z), (0, 1, 0), 3 * W))
        s = s.cut(cyl(INSERT_D / 2, (BAY_X[0] - W / 2 + 4.5, side * (BAY_Y - 4.0), BAY_Z[0]), (0, 0, 1), 2 * INSERT_L + 2))     # bottom-cap inserts
        s = s.cut(cyl(3.0, (GEO["root_x"], side * (BAYU_Y + W / 2), C.BLADE_ROOT_MM["z"]), (0, 1, 0), 3 * W))  # motor wires pass through the wing root, hidden
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
    """The nacelle with the two inserts for the wing's tongue screws, between the fans, from below."""
    s = nacelle_left()
    for y in GEO["tongue_y"]:
        s = s.cut(cyl(INSERT_D / 2, (XC, y, POD_BOT), (0, 0, 1), 2 * INSERT_L))
    for x in SIDE_X:                                                                       # side inserts, horizontal
        s = s.cut(cyl(INSERT_D / 2, (x, Y_FLANK, Z_TIP), (0, 1, 0), 2 * INSERT_L))
    return s


def fan_collar():
    """Trim ring on the shroud under the nacelle: a smooth tapered band that continues the nacelle's underside down
    onto the fan. No ear and no slit; one M2.5 grub screw through its thick upper edge, on the inboard side, holds it."""
    r0, h = C.DUCT_OD_MM / 2 + 0.1, 7.0
    prof = cq.Workplane("XZ").moveTo(r0, 0).lineTo(r0 + 1.0, 0).threePointArc((r0 + 2.6, h * 0.45), (r0 + 3.4, h)).lineTo(r0, h).close()
    s = prof.revolve(360, (0, 0, 0), (0, 1, 0))
    s = s.cut(cyl(1.05, (r0 - 1.0, 0, h - 2.2), (1, 0, 0), 6.0))
    return s


def print_mass(solid, rho, infill=None):
    v, a = solid.val().Volume(), solid.val().Area()
    skin = min(a * C.PRINT_SHELL_MM, v)
    return rho * 1e-6 * (skin + (C.PRINT_INFILL if infill is None else infill) * (v - skin))


bp = backpack(); lid = top_lid(); cap = bottom_cap(); nl = nacelle_with_lugs(); nr = nl.mirror("XZ"); col_part = fan_collar(); wl = wing()
parts = {"backpack": bp, "top_lid": lid, "bottom_cap": cap, "wing_left": wl, "wing_right": wl.mirror("XZ"), "nacelle_left": nl, "nacelle_right": nr, "fan_collar": col_part}
QTY = {"fan_collar": 4}

res = {"parts": {}, "frame_mass_g": {}, "version": "v8 backpack + sculpted blade wings + nacelles"}
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
FAST_G = 17.9   # 4 tongue + 4 side + 4 root + 2 cap + 2 lid + 4 stack screws with inserts, 4 collar screws, pack strap (estimate)
for m in ("PLA", "PLA25", "ABS"):
    res["frame_mass_g"][m] = sum(v["mass_g"][m] * v["qty"] for v in res["parts"].values()) + FAST_G
print("frame incl. %.0f g fasteners: PLA 15 %% %.0f g, PLA 25 %% %.0f g, ABS 15 %% %.0f g" % (FAST_G, res["frame_mass_g"]["PLA"], res["frame_mass_g"]["PLA25"], res["frame_mass_g"]["ABS"]))
# every screw (left side; the right side mirrors): head position, direction towards the driver, parts present at that step
_tg = C.TONGUE_MM
res["screws"] = ([dict(name=f"tongue y={y - LY:+.0f}", head=[XC, y, POD_BOT - _tg["t"]], out=[0, 0, -1], present=["wing_left", "nacelle_left", "fans"]) for y in GEO["tongue_y"]]
                 + [dict(name=f"side x={x - XC:+.0f}", head=[x, GEO["side_y"], Z_TIP], out=[0, -1, 0], present=["wing_left", "nacelle_left", "fans"]) for x in SIDE_X]
                 + [dict(name=f"root z={z:.0f}", head=[GEO["root_x"], BAYU_Y, z], out=[0, -1, 0], present=["wing_left", "backpack"]) for z in GEO["root_z"]]
                 + [dict(name="lid", head=[LUG_X, LID_Y[1], BOXU[5]], out=[0, 0, 1], present=["wing_left", "backpack", "top_lid"])]
                 + [dict(name="cap", head=[BAY_X[0] - W / 2 + 4.5, BAY_Y - 4.0, BOX[4] - 3.0], out=[0, 0, -1], present=["backpack", "bottom_cap", "trunk"])])
res["bay"] = dict(x=list(BAY_X), half_w=BAY_Y, head_x=list(BAYU_X), head_half_w=BAYU_Y, head_z0=Z_HEAD, z=list(BAY_Z), wall=W, stack_xy=STACK_XY, lid_xy=[[LUG_X, y] for y in LID_Y], lid_z=BAY_Z[1], stack_z=BAY_Z[1] - ST["boss"], usb_z=USB_Z)
res["fans_left"] = dict(x=[XC - LX, XC + LX], y=LY, r=C.DUCT_OD_MM / 2, z0=ZM - C.DUCT_LEN_MM / 2 - 14.0, z1=ZM + C.DUCT_LEN_MM / 2)
res["span_y_mm"] = 2 * (LY + R_OUT); res["length_x_mm"] = 2 * (LX + R_OUT)
res["fix_box_mm"] = [XR - 2.0, XR + 12.5, -24.0, 24.0, C.ROOF_Z_MM - 40.0, C.ROOF_Z_MM + 3.0]      # FE support region: the faces bearing on the trunk
res["pod_z_mm"] = [POD_BOT, POD_TOP]; res["wing"] = dict(GEO, stations=[[*blade_section(t)[0].tolist(), blade_section(t)[1], blade_section(t)[2]] for t in np.linspace(0, 1, 9)])
print(f"assembled envelope: {res['span_y_mm']:.0f} mm wide x {res['length_x_mm']:.0f} mm long; nacelles from z = {POD_BOT:.0f} to {POD_TOP:.0f} mm")

# ---- collision primitives (trunk frame, metres) ----------------------------------------------------
T0 = np.array([0, 0, 120.0])
edf = json.load(open(os.path.join(ROOT, "docs", "flight_edf.json")))
bxs = lambda x0, x1, y0, y1, z0, z1, n: dict(name=n, type="box", pos=((np.array([x0 + x1, y0 + y1, z0 + z1]) / 2 - T0) * 1e-3).tolist(),
                                              size=(np.array([x1 - x0, y1 - y0, z1 - z0]) / 2 * 1e-3).tolist())
col = [bxs(BOX[0], BOX[1], BOX[2], BOX[3], BOX[4] - 3.0, BOX[5], "backpack"), bxs(*BOXU, "backpack_head")]
for sy, side in ((1, "left"), (-1, "right")):
    ts = np.linspace(0.04, 0.9, 5)
    for k in range(len(ts) - 1):                                                       # the wing as a chain of capsules along its axis
        (pa, a0, b0), (pb, a1, b1) = blade_section(ts[k]), blade_section(ts[k + 1])
        col.append(dict(name=f"wing_{side}_{k}", type="capsule", fromto=((pa * [1, sy, 1] - T0) * 1e-3).tolist() + ((pb * [1, sy, 1] - T0) * 1e-3).tolist(),
                        radius=float(min(a0 + a1, b0 + b1) / 2 * 0.95e-3)))
    yy = sorted((sy * (LY - R_OUT), sy * (LY + R_OUT)))
    col.append(bxs(XC - LX - R_OUT * 0.75, XC + LX + R_OUT * 0.75, yy[0], yy[1], POD_BOT, POD_TOP, f"nacelle_{side}"))
res["collision"] = col
res["edf_collision"] = dict(type="cylinder", radius_mm=C.DUCT_OD_MM / 2, z0_mm=-C.DUCT_LEN_MM / 2 - edf["motor_out_mm"], z1_mm=C.DUCT_LEN_MM / 2 - 2.0)   # below / inside the nacelle
json.dump(res, open(os.path.join(ROOT, "docs", "flight_cad_results.json"), "w"), indent=1)

colors = {"backpack": "#d8d8d8", "top_lid": "#e9762b", "bottom_cap": "#e9762b", "nacelle_left": "#f2f2f2", "nacelle_right": "#f2f2f2", "wing_left": "#e4e4e4", "wing_right": "#e4e4e4"}
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
fig.suptitle("Frame v8: backpack, sculpted wings, nacelles, bottom cap (orange); dashed = trunk", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "flight_frame.pdf"))
print("wrote cad/flight/out/*.step|stl, docs/flight_cad_results.json, docs/figures/flight_frame.pdf")
