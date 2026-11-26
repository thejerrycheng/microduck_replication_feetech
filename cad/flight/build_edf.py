#!/usr/bin/env python3
"""Detailed model of the 50 mm, 12-blade ducted fan (QX-Motor QF2611 class) for visuals and for collision sizing.

    .venv-cad/bin/python cad/flight/build_edf.py

Fan axis = +z (thrust direction; air enters at the top, leaves at the bottom), origin at the shroud's mid-length.
Listed dimensions (seller pages): shroud OD 54.0, length 39.1, motor diameter 25.8, 12 blades, 3.0 mm shaft. Everything
else (lip radius, blade chord and twist, stator count, motor length) is a representative guess: this is a visual and
envelope model, not a manufacturing drawing of the fan. Parts: shroud (with inlet lip and stator vanes + motor mount),
rotor (hub, spinner, 12 twisted blades), motor can. Writes STEP/STL to cad/flight/out/ and docs/flight_edf.json.
"""
import json, math, os, sys
import cadquery as cq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools"))
import flight_config as C

OUT = os.path.join(ROOT, "cad", "flight", "out")
OD, LEN = C.DUCT_OD_MM, C.DUCT_LEN_MM
R_O, R_I = OD / 2, 25.2                 # 50 mm fan + tip gap
Z0, Z1 = -LEN / 2, LEN / 2
LIP_R, LIP_OUT = 4.0, 0.0               # no intake ring: the nacelle's own bell-mouth is the inlet lip (frame v5)
HUB_R, N_BLADES, N_STATORS = 13.0, 12, 5
MOTOR_R, MOTOR_LEN, MOTOR_OUT = 12.9, 30.0, 14.0   # can sticks 14 mm out of the exhaust side
ROTOR_Z = 5.0                           # rotor disc mid-plane above the shroud's mid-length


def shroud():
    prof = [(R_I, Z0), (R_O, Z0), (R_O, Z1), (R_I, Z1)]
    s = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    # stator vanes + motor mount tube in the exhaust half
    tube = cq.Workplane("XY").workplane(offset=Z0).circle(MOTOR_R + 1.6).circle(MOTOR_R).extrude(16)
    s = s.union(tube)
    for k in range(N_STATORS):
        vane = cq.Workplane("XY").workplane(offset=Z0).center((MOTOR_R + 1.0 + R_I + 0.6) / 2, 0).rect(R_I + 0.6 - MOTOR_R - 1.0, 1.4).extrude(13)
        s = s.union(vane.rotate((0, 0, 0), (0, 0, 1), 360 / N_STATORS * k + 18))
    return s


def blade():
    """One blade: a thin elliptical section swept from hub to tip with 24 deg of twist (52 deg pitch at the hub, 28 at the tip)."""
    span = (R_I - 1.6) - (HUB_R - 0.6)
    return cq.Workplane("YZ").workplane(offset=HUB_R - 0.6).transformed(rotate=(0, 0, 52.0)).ellipse(6.8, 0.65).twistExtrude(span, -24.0)


def rotor(spin=+1):
    hub = cq.Workplane("XY").workplane(offset=ROTOR_Z - 7.5).circle(HUB_R).extrude(15)
    nose = [(0, ROTOR_Z + 7.5), (HUB_R, ROTOR_Z + 7.5), (HUB_R * 0.86, ROTOR_Z + 11.5), (HUB_R * 0.55, ROTOR_Z + 15.5), (HUB_R * 0.2, ROTOR_Z + 18.0), (0, ROTOR_Z + 18.6)]
    r = hub.union(cq.Workplane("XZ").polyline(nose).close().revolve(360, (0, 0, 0), (0, 1, 0)))
    b = blade().translate((0, 0, ROTOR_Z))
    if spin < 0:
        b = b.mirror("XZ")
    for k in range(N_BLADES):
        r = r.union(b.rotate((0, 0, 0), (0, 0, 1), 360 / N_BLADES * k))
    return r


def motor():
    m = cq.Workplane("XY").workplane(offset=Z0 - MOTOR_OUT).circle(MOTOR_R - 0.1).extrude(MOTOR_LEN)
    m = m.faces("<Z").edges().chamfer(1.5)
    for k in range(3):   # three phase leads leaving the back
        m = m.union(cq.Workplane("XY").workplane(offset=Z0 - MOTOR_OUT - 5).center(5.5 * math.cos(2.1 * k), 5.5 * math.sin(2.1 * k)).circle(1.1).extrude(5.5))
    return m


parts = {"edf_shroud": shroud(), "edf_rotor_ccw": rotor(+1), "edf_rotor_cw": rotor(-1), "edf_motor": motor()}
info = dict(od_mm=OD, lip_od_mm=OD + 2 * LIP_OUT, length_mm=LEN, motor_out_mm=MOTOR_OUT, spinner_top_mm=ROTOR_Z + 18.6, z_min_mm=Z0 - MOTOR_OUT - 5, z_max_mm=max(Z1, ROTOR_Z + 18.6),
            collision=dict(type="cylinder", radius_mm=OD / 2 + LIP_OUT, z0_mm=Z0 - MOTOR_OUT, z1_mm=Z1 + 2.0), blades=N_BLADES, stators=N_STATORS, parts={})
for name, p in parts.items():
    cq.exporters.export(p, os.path.join(OUT, name + ".step"))
    cq.exporters.export(p, os.path.join(OUT, name + ".stl"), tolerance=0.08, angularTolerance=0.35)
    bb = p.val().BoundingBox()
    n_tri = os.path.getsize(os.path.join(OUT, name + ".stl"))
    info["parts"][name] = dict(bbox_mm=[bb.xlen, bb.ylen, bb.zlen], volume_cm3=p.val().Volume() / 1e3)
    print(f"{name:14s} bbox {bb.xlen:5.1f} x {bb.ylen:5.1f} x {bb.zlen:5.1f} mm, volume {p.val().Volume()/1e3:5.1f} cm^3, stl {n_tri/1e3:.0f} kB")
json.dump(info, open(os.path.join(ROOT, "docs", "flight_edf.json"), "w"), indent=1)
print("envelope: lip OD %.1f mm, from z %.1f to %.1f mm about the shroud mid-plane" % (info["lip_od_mm"], info["z_min_mm"], info["z_max_mm"]))
