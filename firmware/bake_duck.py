"""Bake the standing Microduck into web/duck.bin for the studio's 3-D view. Dev tool, run once:

    ~/.venvs/microduck_live/bin/python firmware/bake_duck.py

Layout: [uint32 n][n bytes JSON][payload]. JSON = {meshes:[{nv, ni}], parts:[{mesh, rgba, servo, sensor, pos, rot}],
servos:[{id, name, at, axis}], sensors:[{id, at, dir, up}], bodies:[{name, pos, quat}]}; per mesh the payload holds int16 xyz * nv (1e-5 m, mesh frame), int8 normal * nv,
uint32 index * ni, each block padded to 4 bytes. Parts place meshes in the world (z up, soles on z = 0).
"""
import json, struct, sys, types
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(HERE))
import engine                       # only for its crease-smoothed mesh builder
from feetech import DUCK            # {id: joint name}

m = mujoco.MjModel.from_xml_path(str(APP / "assets/microduck/scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[:3] = 0, 0, d.qpos[2]
mujoco.mj_forward(m, d)
z0 = min(d.geom_xpos[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] > 0) - 0.012   # soles on z = 0

geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
         and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
housings = [g for g in geoms if "xl330" in m.mesh(m.geom_dataid[g]).name]

# each servo = the housing nearest its joint; the mouth (no joint in the MJCF) takes the one left over
servos, used = [], {}
for sid, name in DUCK.items():
    if name == "mouth":
        continue
    j = m.joint(name).id
    g = min((g for g in housings if g not in used), key=lambda g: np.linalg.norm(d.geom_xpos[g] - d.xanchor[j]))
    used[g] = sid
    servos.append(dict(id=sid, name=name, at=d.xanchor[j], axis=d.xaxis[j]))
(g,) = [g for g in housings if g not in used]
used[g] = mouth = next(s for s, n in DUCK.items() if n == "mouth")
servos.append(dict(id=mouth, name="mouth", at=d.geom_xpos[g], axis=np.array([0.0, 1.0, 0.0])))

# parts that are a sensor: they light up in the studio the way the servo housings do
PART = dict(lens="camera", m12_lens_holder="camera", speaker="speaker", elec_rpi_robot_hat_pcb="mic")
GRID = 0.0003                        # weld vertices onto a 0.3 mm grid: 430 k source triangles -> a few MB


def mesh(i):
    """Mesh i, welded and crease-smoothed, in its own frame: (float32 xyz, float32 normal, uint32 index)."""
    V = m.mesh_vert[m.mesh_vertadr[i]:][:m.mesh_vertnum[i]]
    F = m.mesh_face[m.mesh_faceadr[i]:][:m.mesh_facenum[i]]
    U, inv = np.unique(np.round(V / GRID).astype(np.int64), axis=0, return_inverse=True)
    F = inv.reshape(-1)[F]
    F = F[np.linalg.norm(np.cross(U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]), axis=1) > 0]   # welded flat
    one = types.SimpleNamespace(mesh_vertadr=[0], mesh_vertnum=[len(U)], mesh_faceadr=[0], mesh_facenum=[len(F)],
                                mesh_vert=U * GRID, mesh_face=F)
    raw = engine.Duck._build_mesh(types.SimpleNamespace(model=one, _mesh_cache={}), 0, 40.0)
    nv, ni = struct.unpack("<2I", raw[:8])
    return (np.frombuffer(raw, np.float32, nv * 3, 8).reshape(-1, 3),
            np.frombuffer(raw, np.float32, nv * 3, 8 + nv * 12).reshape(-1, 3), np.frombuffer(raw, np.uint32, ni, 8 + nv * 24))


meshes, parts, blob = {}, [], b""
for g in geoms:
    i = int(m.geom_dataid[g])
    if i not in meshes:                                   # each mesh once; the 15 housings share one
        P, N, I = mesh(i)
        assert np.abs(P).max() < 0.327, "mesh does not fit int16 at 1e-5 m"
        meshes[i] = dict(nv=len(P), ni=len(I))
        blob += np.round(P * 1e5).astype("<i2").tobytes() + b"\0" * (len(P) * 6 % 4)
        blob += np.round(np.nan_to_num(N) * 127).astype("i1").tobytes() + b"\0" * (-len(P) * 3 % 4) + I.tobytes()
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    parts.append(dict(mesh=list(meshes).index(i), rgba=[round(float(x), 3) for x in rgba], servo=used.get(g, 0),
                      sensor=PART.get(m.mesh(i).name, ""),
                      body=m.body(m.geom_bodyid[g]).name,            # for a posed duck (the iOS app): the part's body
                      lpos=[round(float(x), 6) for x in m.geom_pos[g]], lquat=[round(float(x), 6) for x in m.geom_quat[g]],
                      pos=[round(float(x), 5) for x in d.geom_xpos[g] - [0, 0, z0]],
                      rot=[round(float(x), 5) for x in d.geom_xmat[g]]))

# sensors: where each sits and, for the two that look at something, which way. Positions are the model's sites.
# The speaker is its own part in the model. The microphone is the MEMS one on the robot HAT, so it sits at that
# board's centre; where on the board it is soldered is not in the model.
part = lambda name: next(d.geom_xpos[g].copy() for g in geoms if m.mesh(m.geom_dataid[g]).name == name)
site = lambda n: d.site_xpos[m.site(n).id]
look, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])   # the standing duck faces +x; both sit on its face
sensors = [dict(id="camera", at=site("head_camera"), dir=look, up=up), dict(id="tof", at=site("tof"), dir=look, up=up),
           dict(id="imu", at=site("imu")), dict(id="mic", at=part("elec_rpi_robot_hat_pcb")),
           dict(id="speaker", at=part("speaker"))]
for s in sensors:
    s["at"] = s["at"] - [0, 0, z0]
    for k in ("at", "dir", "up"):
        if k in s:
            s[k] = [round(float(x), 4) for x in s[k]]

for s in servos:
    s["at"] = [round(float(x), 4) for x in s["at"] - [0, 0, z0]]
    s["axis"] = [round(float(x), 3) for x in s["axis"]]
# the standing pose of every body (world frame, quaternion w x y z), so a viewer can pose the parts by body
bodies = [dict(name=m.body(b).name, pos=[round(float(x), 5) for x in d.xpos[b] - [0, 0, z0]],
               quat=[round(float(x), 5) for x in d.xquat[b]]) for b in range(1, m.nbody)]
head = json.dumps(dict(meshes=list(meshes.values()), parts=parts, servos=servos, sensors=sensors, bodies=bodies, z0=round(float(z0), 5)),
                  separators=(",", ":")).encode()
head += b" " * (-len(head) % 4)
out = HERE / "web/duck.bin"
out.write_bytes(struct.pack("<I", len(head)) + head + blob)
print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  {len(parts)} parts  {len(servos)} servos")
for s in sorted(servos, key=lambda s: s["id"]):
    print(f"  {s['id']:>3}  {s['name']:<16} {s['at']}")
for s in sensors:
    print(f"  {s['id']:<8} {s['at']}  {s.get('dir', '')}")
