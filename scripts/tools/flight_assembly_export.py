#!/usr/bin/env python3
"""Geometry for the web assembly guide of the flying-duck frame.

    ~/.venvs/microduck_live/bin/python scripts/tools/flight_assembly_export.py [--out DIR]

Reads the MuJoCo scene (app/assets/microduck/scene_flight.xml) at the STAND keyframe and writes every visual mesh in
world coordinates (metres, z up), grouped by part: the duck (merged and decimated by vertex clustering), the printed
parts, the four fans (shroud, rotor, motor, trim ring) and boxes for the pack and the FC/ESC stack.
Output: DIR/parts.bin (float32 positions, uint32 indices) and DIR/parts.json (offsets, colours).
"""
import argparse, json, os, struct, sys
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_config as C

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assembly")); ap.add_argument("--grid_mm", type=float, default=1.6)
a = ap.parse_args()
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "app", "assets", "microduck", "scene_flight.xml")); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")); mujoco.mj_forward(m, d)
fp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "flight_pack")
FAN = ["FL", "RL", "FR", "RR"]


def geom_mesh(g):
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f


def cluster(v, f, h):
    """Vertex-clustering decimation on a grid of pitch h."""
    key = np.floor(v / h).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    nv = np.zeros((cnt.size, 3)); np.add.at(nv, inv, v); nv /= cnt[:, None]
    f = inv[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = f[np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]]
    return nv, f


def box(c, half):
    c, half = np.asarray(c, float), np.asarray(half, float)
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


groups, fan_k, cad_off = {}, {}, None
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] > 2:
        continue
    v, f = geom_mesh(g)
    name = m.mesh(m.geom_dataid[g]).name
    if m.geom_bodyid[g] != fp:
        key = "duck"
    elif name.startswith("flight_edf") or name == "flight_fan_collar":
        kind = name.replace("flight_", "").replace("edf_", "").replace("_ccw", "").replace("_cw", "").replace("fan_collar", "ring")
        fan_k[kind] = fan_k.get(kind, -1) + 1
        key = f"{kind}_{FAN[fan_k[kind]]}"
    else:
        key = name.replace("flight_", "")
    gv, gf = groups.get(key, (np.zeros((0, 3)), np.zeros((0, 3), int)))
    groups[key] = (np.vstack([gv, v]), np.vstack([gf, f + len(gv)]))
groups["duck"] = cluster(*groups["duck"], a.grid_mm * 1e-3)
for k in list(groups):
    if k.startswith("rotor"):
        groups[k] = cluster(*groups[k], 0.5e-3)

# pack and stack: boxes at their CAD positions (CAD frame = duck standing, mm; offset found from the backpack's top face)
import re
raw = np.fromfile(os.path.join(ROOT, "cad", "flight", "out", "backpack.stl"), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), offset=84)["v"].reshape(-1, 3)
cad_off = groups["backpack"][0].max(0) - raw.max(0) * 1e-3
bs = np.array(C.BATT_SIZE)
groups["battery"] = box(np.array(C.BATT_POS) + cad_off, bs / 2)
groups["stack"] = box(np.array(C.FC_POS) + cad_off, [0.018, 0.018, 0.008])

COL = dict(duck="#d9d9d6", backpack="#f2f2ee", bottom_cap="#ff671f", wing="#f2f2ee", nacelle="#f2f2ee", shroud="#4a4a4a", rotor="#f4af23", motor="#212721", ring="#ff671f", battery="#2b2f36", stack="#1f6f54")
os.makedirs(a.out, exist_ok=True)
man, blob = dict(units="m", up="z", parts=[]), b""
for k, (v, f) in groups.items():
    vb, fb = v.astype("<f4").tobytes(), f.astype("<u4").tobytes()
    man["parts"].append(dict(name=k, color=next(c for n, c in COL.items() if k.startswith(n)), v_off=len(blob), v_n=len(v), f_off=len(blob) + len(vb), f_n=len(f),
                             center=v.mean(0).round(4).tolist()))
    blob += vb + fb
    print(f"{k:16s} {len(v):7d} verts {len(f):7d} tris")
open(os.path.join(a.out, "parts.bin"), "wb").write(blob)
json.dump(man, open(os.path.join(a.out, "parts.json"), "w"))
print(f"wrote {a.out}/parts.bin ({len(blob) / 1e6:.2f} MB), parts.json; CAD offset {cad_off.round(4)}")
