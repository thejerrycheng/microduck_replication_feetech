#!/usr/bin/env python3
"""Swept volumes of the head/neck and of a leg over their full joint ranges, in the trunk_base frame.

    python scripts/tools/flight_envelopes.py

Every joint of the group is stepped across its range (7 values per head joint, 5 per leg joint); poses in which the
moving parts pass through the trunk shell or across the mid-plane are dropped (the model has no self-collision in
this sweep). Writes docs/flight_envelopes.npz (point clouds, thinned to a 4 mm voxel grid) and
docs/flight_envelopes.json. The flight frame and fans must stay outside these clouds (cad/flight/build_parts.py checks).
"""
import itertools, json, os
import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "app", "assets", "microduck", "scene.xml")); d = mujoco.MjData(m)
k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"); mujoco.mj_resetDataKeyframe(m, d, k); mujoco.mj_forward(m, d)
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
jq = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
rng = lambda n: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
SHELL = np.array([[-0.0465, 0.0355], [-0.032, 0.032], [0.0, 0.042]])   # trunk shell box in the trunk frame
q0 = d.qpos.copy()


def body_pts(names, stride):
    Rt, pt = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
    P = []
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name in names and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][::stride]
            P.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - pt) @ Rt)
    return np.vstack(P)


def sweep(joints, bodies, check, n, stride, shrink=0.004, midplane=None):
    grids = [np.linspace(*rng(j), n) for j in joints]
    vox, kept, total = set(), 0, 0
    for vals in itertools.product(*grids):
        d.qpos[:] = q0
        for j, v in zip(joints, vals):
            d.qpos[jq(j)] = v
        mujoco.mj_kinematics(m, d)
        total += 1
        c = body_pts(check, stride)
        inside = np.all((c > SHELL[:, 0] + shrink) & (c < SHELL[:, 1] - shrink), axis=1)
        if inside.mean() > 0.002 or (midplane is not None and (c[:, 1] < midplane).mean() > 0.002):
            continue
        kept += 1
        P = body_pts(bodies, stride)
        vox.update(map(tuple, np.round(P / 0.004).astype(np.int32)))
    d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
    return np.array(sorted(vox), dtype=np.float32) * 0.004, kept, total


head, hk, ht = sweep(["neck_pitch", "head_pitch", "head_yaw", "head_roll"], ["neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"],
                     ["yaw_roll_motion", "jaw_soft"], n=7, stride=8)
leg, lk, lt = sweep(["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"], ["yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left"],
                    ["upper_leg_left", "leg", "ankle_left"], n=5, stride=8, midplane=-0.004)
np.savez_compressed(os.path.join(ROOT, "docs", "flight_envelopes.npz"), head=head, leg_left=leg)
mm = lambda a: [int(round(v * 1e3)) for v in a]
rep = dict(head=dict(poses_kept=hk, poses=ht, min_mm=mm(head.min(0)), max_mm=mm(head.max(0))), leg_left=dict(poses_kept=lk, poses=lt, min_mm=mm(leg.min(0)), max_mm=mm(leg.max(0))))
print(f"head/neck: {hk}/{ht} poses kept, swept box min {rep['head']['min_mm']} max {rep['head']['max_mm']} mm (trunk frame; roof at z = 42, floor at z = -120)")
print(f"left leg : {lk}/{lt} poses kept, swept box min {rep['leg_left']['min_mm']} max {rep['leg_left']['max_mm']} mm")
print("\nhead/neck: lowest z [mm] by region")
tab = {}
for (x0, x1) in ((-140, -100), (-100, -70), (-70, -47), (-47, -20), (-20, 10), (10, 40), (40, 180)):
    row = []
    for (y0, y1) in ((0, 35), (35, 60), (60, 80), (80, 150)):
        s = (head[:, 0] * 1e3 >= x0) & (head[:, 0] * 1e3 < x1) & (np.abs(head[:, 1]) * 1e3 >= y0) & (np.abs(head[:, 1]) * 1e3 < y1)
        row.append(int(round(head[s, 2].min() * 1e3)) if s.any() else None)
    tab[f"x{x0}_{x1}"] = row
    print(f"  x [{x0:4d},{x1:4d}): |y| 0-35 {row[0]}  35-60 {row[1]}  60-80 {row[2]}  80+ {row[3]}")
rep["head_lowest_z_by_region"] = tab
print("\nleft leg: highest z and outermost y [mm] by x band")
tab = {}
for (x0, x1) in ((-140, -100), (-100, -70), (-70, -47), (-47, -20), (-20, 10), (10, 40), (40, 80), (80, 130)):
    s = (leg[:, 0] * 1e3 >= x0) & (leg[:, 0] * 1e3 < x1)
    if s.any():
        tab[f"x{x0}_{x1}"] = [int(round(leg[s, 2].max() * 1e3)), int(round(leg[s, 1].max() * 1e3))]
        print(f"  x [{x0:4d},{x1:4d}): z max {tab[f'x{x0}_{x1}'][0]:4d}  y max {tab[f'x{x0}_{x1}'][1]:4d}")
rep["leg_highest_z_outermost_y_by_x"] = tab
print("\nleft leg: highest z [mm] by lateral band (where can a boom pass over the leg?)")
tab = {}
for (y0, y1) in ((30, 50), (50, 70), (70, 90), (90, 110), (110, 135)):
    s = (leg[:, 1] * 1e3 >= y0) & (leg[:, 1] * 1e3 < y1)
    if s.any():
        tab[f"y{y0}_{y1}"] = int(round(leg[s, 2].max() * 1e3)); print(f"  y [{y0:3d},{y1:3d}): z max {tab[f'y{y0}_{y1}']:4d}")
rep["leg_highest_z_by_y"] = tab
json.dump(rep, open(os.path.join(ROOT, "docs", "flight_envelopes.json"), "w"), indent=1)
