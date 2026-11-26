#!/usr/bin/env python3
"""Build the flying-duck simulation model from the design: rigid-body properties of the flight hardware and MuJoCo scenes.

    python scripts/tools/flight_model.py        # after flight_design.py and cad/flight/build_parts.py

Writes
  microduck_rl/flight/flight_pack.json            mass, CoM, inertia of everything added to the trunk + the four fans
                                                  (position, thrust axis, spin) in the trunk_base frame: read by the
                                                  Isaac Lab tasks and by the MuJoCo rollout
  app/assets/microduck/scene_flight[_backlash].xml  the official scenes with the pack welded to the trunk (visual meshes,
                                                  fan sites); body masses scaled to the HD1910 build (+5 g per servo)
"""
import json, math, os, shutil, sys
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_config as C

ASSETS = os.path.join(ROOT, "app", "assets", "microduck")
res = json.load(open(os.path.join(ROOT, "docs", "flight_results.json")))
cad = json.load(open(os.path.join(ROOT, "docs", "flight_cad_results.json")))
TRUNK0 = np.array([0.0, 0.0, 0.12])     # trunk_base origin in the design (world, standing) frame
xc = res["layout"]["x_c"]
MAT = "ABS"


def box_I(m, s):
    return m / 12 * np.diag([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])


items = []   # (name, mass kg, com in trunk frame, inertia about own com in trunk axes)
fans = []
f = C.FANS[C.PRIMARY]
r, h = C.DUCT_OD_MM * 0.5e-3, C.DUCT_LEN_MM * 1e-3
g = math.radians(C.FAN_CANT_DEG)
for name, sx, sy, s in (("FL", 1, 1, 1), ("RL", -1, 1, -1), ("FR", 1, -1, -1), ("RR", -1, -1, 1)):
    p = np.array([xc + sx * C.FAN_LX, sy * C.FAN_LY, C.FAN_Z]) - TRUNK0
    rad = np.array([sx * C.FAN_LX, sy * C.FAN_LY, 0.0]); rad /= np.linalg.norm(rad)
    axis = np.array([0, 0, 1.0]) * math.cos(g) + s * np.cross([0, 0, 1.0], rad) * math.sin(g)
    m = f["mass_g"] * 1e-3
    items.append((f"fan_{name}", m, p, np.diag([m * (3 * r * r + h * h) / 12] * 2 + [m * r * r / 2])))
    fans.append(dict(name=name, pos=p.tolist(), axis=axis.tolist(), yaw_sign=s, spin=-s))  # spin: +1 = CCW seen from above
bud = res["primary"]["items"]
m = bud["battery"] * 1e-3
items.append(("pack", m, np.array(C.BATT_POS) - TRUNK0, box_I(m, C.BATT_SIZE)))
m = (bud["ESC"] + bud["flight controller"]) * 1e-3
items.append(("esc_fc", m, np.array(C.FC_POS) - TRUNK0, box_I(m, (0.036, 0.036, 0.016))))
m = bud["wiring"] * 1e-3
items.append(("wiring", m, np.array([xc, 0, C.FAN_Z]) - TRUNK0, box_I(m, (0.08, 0.16, 0.01))))
printed = 0.0
for name, part in cad["parts"].items():
    m = part["mass_g"][MAT] * 1e-3
    printed += m
    I = np.array(part["inertia_unit_mm5"]) * 1e-15 * (m / (part["volume_cm3"] * 1e-6))
    items.append((name, m, np.array(part["com_mm"]) * 1e-3 - TRUNK0, I))
m = cad["frame_mass_g"][MAT] * 1e-3 - printed
items.append(("fasteners", m, np.array([xc, 0, C.FAN_Z]) - TRUNK0, box_I(m, (0.06, 0.1, 0.01))))

M = sum(i[1] for i in items)
com = sum(i[1] * i[2] for i in items) / M
I = np.zeros((3, 3))
for _, mi, pi, Ii in items:
    d = pi - com
    I += Ii + mi * ((d @ d) * np.eye(3) - np.outer(d, d))
budget_added = (sum(bud.values()) - bud["duck"]) * 1e-3
print(f"flight hardware: {M*1e3:.1f} g (budget {budget_added*1e3:.1f} g), CoM in trunk frame {np.round(com*1e3,1)} mm")
print("inertia about its CoM [g m^2]:", np.round(I * 1e3, 3).tolist())
assert abs(M - budget_added) < 2e-3, "pack mass and budget disagree"
pack = dict(mass=M, com=com.tolist(), inertia=I.tolist(), fans=fans, thrust_centroid_x=xc - TRUNK0[0],
            items=[dict(name=n, mass=mi, com=pi.tolist()) for n, mi, pi, _ in items],
            servo_mass_scale=(res["duck_kg"]) / res["duck_model_kg"], fan_rotor_radius=0.025, duct_od=C.DUCT_OD_MM * 1e-3, duct_len=h,
            fan=dict(kv_rpm_per_v=C.FAN_KV, bench_thrust_n=C.FAN_BENCH_THRUST_G * 1e-3 * C.G, bench_v=C.FAN_BENCH_V, bench_i=C.FAN_BENCH_I,
                     i_limit=C.FAN_I_LIMIT, eta_inst=C.FAN_INSTALL_ETA, area=math.pi / 4 * (f["d_mm"] ** 2 - f["hub_mm"] ** 2) * 1e-6),
            battery=dict(n_s=4, q_ah=C.BATT_REF_AH, r_cell=C.CELL_MOHM_AH * 1e-3 / C.BATT_REF_AH))
json.dump(pack, open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json"), "w"), indent=1)

# ---- MuJoCo scenes ---------------------------------------------------------------------------------
for part in cad["parts"]:
    shutil.copy(os.path.join(ROOT, "cad", "flight", "out", part + ".stl"), os.path.join(ASSETS, "assets", "flight_" + part + ".stl"))
for scene in ("scene.xml", "scene_backlash.xml"):
    spec = mujoco.MjSpec.from_file(os.path.join(ASSETS, scene))
    for b in spec.bodies:
        if b.name and b.name != "world" and b.mass > 0:
            b.mass *= pack["servo_mass_scale"]; b.inertia = np.array(b.inertia) * pack["servo_mass_scale"]
    trunk = spec.body("trunk_base")
    fp = trunk.add_body(name="flight_pack", pos=com.tolist())
    fp.mass = M
    fp.ipos = [0, 0, 0]
    fp.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    fp.explicitinertial = True
    rgba = dict(frame_left=[0.82, 0.38, 0.24, 1], frame_right=[0.82, 0.38, 0.24, 1], pack_tray=[0.2, 0.2, 0.22, 1])
    for part in cad["parts"]:
        mesh = spec.add_mesh(name="flight_" + part, file="flight_" + part + ".stl", scale=[1e-3] * 3)
        fp.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="flight_" + part, pos=(-TRUNK0 - com).tolist(), contype=0, conaffinity=0,
                    group=2, rgba=rgba.get(part, [0.16, 0.42, 0.59, 1]), mass=0)
    bp = np.array(C.BATT_POS) - TRUNK0 - com
    fp.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(np.array(C.BATT_SIZE) / 2).tolist(), pos=bp.tolist(), contype=0, conaffinity=0, group=2, rgba=[0.1, 0.1, 0.1, 1], mass=0)
    for fn in fans:
        z = np.array(fn["axis"]); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.c_[x, y, z].flatten())
        fp.add_site(name="fan_" + fn["name"], pos=(np.array(fn["pos"]) - com).tolist(), quat=q.tolist(), size=[0.004] * 3)
        fp.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=(np.array(fn["pos"]) - com).tolist(), quat=q.tolist(),
                    contype=0, conaffinity=0, group=2, rgba=[0.08, 0.08, 0.08, 1], mass=0)
    model = spec.compile()
    out = os.path.join(ASSETS, scene.replace("scene", "scene_flight"))
    open(out, "w").write(spec.to_xml())
    chk = mujoco.MjModel.from_xml_path(out)
    print(f"{os.path.relpath(out, ROOT)}: total mass {chk.body_subtreemass[1]*1e3:.1f} g, {chk.nbody} bodies")
