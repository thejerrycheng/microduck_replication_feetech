#!/usr/bin/env python3
"""Score a set of obstacle boxes (frame, fans, pack) by the share of head/neck and leg poses they block.

    python scripts/tools/flight_clearance.py            # compares the candidate layouts defined below

Poses: the same grids as flight_envelopes.py (7 values per head joint, 5 per leg joint), impossible poses dropped.
A pose is blocked when any surface point of the moving bodies comes within MARGIN of an obstacle box (trunk frame).
Reports blocked share for head and for one leg (obstacles are mirrored), and the largest neck-pitch-back / hip-pitch
range that stays free with the other joints of the group at their stand value. Importable: `blocked(boxes)`.
"""
import itertools, json, os, sys
import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MARGIN = 0.003
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "app", "assets", "microduck", "scene.xml")); d = mujoco.MjData(m)
k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"); mujoco.mj_resetDataKeyframe(m, d, k); mujoco.mj_forward(m, d)
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
jq = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
rng = lambda n: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
SHELL = np.array([[-0.0465, 0.0355], [-0.032, 0.032], [0.0, 0.042]])
q0 = d.qpos.copy()
HEAD_J, HEAD_B, HEAD_C = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], ["neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"], ["yaw_roll_motion", "jaw_soft"]
LEG_J, LEG_B, LEG_C = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"], ["yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left"], ["upper_leg_left", "leg", "ankle_left"]


def pts(names, stride=10):
    Rt, pt = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
    P = []
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name in names and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][::stride]
            P.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - pt) @ Rt)
    return np.vstack(P).astype(np.float32)


def poses(joints, bodies, check, n, midplane=None):
    out = []
    for vals in itertools.product(*[np.linspace(*rng(j), n) for j in joints]):
        d.qpos[:] = q0
        for j, v in zip(joints, vals):
            d.qpos[jq(j)] = v
        mujoco.mj_kinematics(m, d)
        c = pts(check)
        if np.all((c > SHELL[:, 0] + 0.004) & (c < SHELL[:, 1] - 0.004), axis=1).mean() > 0.002 or (midplane is not None and (c[:, 1] < midplane).mean() > 0.002):
            continue
        out.append((np.array(vals), pts(bodies)))
    d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
    return out


_cache = {}


def _poses(which):
    if which not in _cache:
        _cache[which] = poses(HEAD_J, HEAD_B, HEAD_C, 7) if which == "head" else poses(LEG_J, LEG_B, LEG_C, 5, midplane=-0.004)
    return _cache[which]


def hits(P, boxes):
    """boxes: (k, 6) [x0, x1, y0, y1, z0, z1] in metres, y >= 0 side; mirrored copies are added here."""
    B = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    M = B.copy(); M[:, 2], M[:, 3] = -B[:, 3], -B[:, 2]
    B = np.vstack([B, M])
    lo, hi = B[:, [0, 2, 4]] - MARGIN, B[:, [1, 3, 5]] + MARGIN
    return bool(((P[:, None, :] > lo[None]) & (P[:, None, :] < hi[None])).all(-1).any())


def blocked(boxes):
    res = {}
    for which, joints in (("head", HEAD_J), ("leg", LEG_J)):
        ps = _poses(which)
        hit = np.array([hits(P, boxes) for _, P in ps])
        res[which] = float(hit.mean())
    # single-joint free ranges from the stand pose (the motions that matter most)
    for j, bodies in (("neck_pitch", HEAD_B), ("head_pitch", HEAD_B), ("head_yaw", HEAD_B), ("left_hip_pitch", LEG_B), ("left_hip_roll", LEG_B), ("left_hip_yaw", LEG_B), ("left_knee", LEG_B)):
        lo, hi = rng(j); free = []
        for v in np.linspace(lo, hi, 61):
            d.qpos[:] = q0; d.qpos[jq(j)] = v; mujoco.mj_kinematics(m, d)
            free.append(not hits(pts(bodies, 6), boxes))
        d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
        vals = np.degrees(np.linspace(lo, hi, 61))
        i0 = int(np.argmin(np.abs(np.linspace(lo, hi, 61) - q0[jq(j)])))
        a = b = i0
        if not free[i0]:
            res[j] = None; continue
        while a > 0 and free[a - 1]: a -= 1
        while b < 60 and free[b + 1]: b += 1
        res[j] = [float(vals[a]), float(vals[b]), float(vals[0]), float(vals[-1])]
    return res


def fan_boxes(xc, lx, ly, z0, z1, r=0.0285):
    return [[xc + sx * lx - r, xc + sx * lx + r, ly - r, ly + r, z0, z1] for sx in (1, -1)]


if __name__ == "__main__":
    xc = -0.0107
    C = {
        "v3 roof spars (current)": [[-0.036, -0.022, 0, 0.102, 0.043, 0.061], [0.001, 0.015, 0, 0.102, 0.043, 0.061], [xc - 0.042, xc + 0.042, 0.072, 0.132, 0.043, 0.061]]
                                   + fan_boxes(xc, 0.042, 0.102, 0.032, 0.072) + [[-0.0865, -0.0495, 0, 0.019, -0.035, 0.042]],
        "A high pods, rear mast + cross-beam": [[-0.062, -0.048, 0, 0.012, 0.0, 0.110], [-0.062, -0.048, 0, 0.125, 0.098, 0.112], [-0.062, 0.031, 0.117, 0.129, 0.098, 0.112]]
                                   + fan_boxes(xc, 0.042, 0.123, 0.096, 0.136) + [[-0.0865, -0.0495, 0, 0.019, -0.050, 0.0]],
        "B high pods, two side arms at roof level": [[-0.030, -0.018, 0.032, 0.123, 0.044, 0.058], [-0.030, -0.018, 0.117, 0.129, 0.044, 0.110], [-0.053, 0.031, 0.117, 0.129, 0.098, 0.112]]
                                   + fan_boxes(xc, 0.042, 0.123, 0.096, 0.136) + [[-0.0865, -0.0495, 0, 0.019, -0.050, 0.0]],
        "C wide pods at trunk height, side arms": [[-0.030, -0.018, 0.032, 0.165, 0.020, 0.036]] + fan_boxes(xc, 0.042, 0.165, 0.010, 0.050) + [[-0.0865, -0.0495, 0, 0.019, -0.050, 0.0]],
        "D high pods, V arms from the rear corners": [[-0.058, -0.046, 0.020, 0.034, 0.0, 0.045]] + [[-0.058, -0.046, 0.030 + i * 0.0155, 0.047 + i * 0.0155, 0.040 + i * 0.0105, 0.056 + i * 0.0105] for i in range(6)]
                                   + [[-0.058, 0.031, 0.117, 0.129, 0.098, 0.112]] + fan_boxes(xc, 0.042, 0.123, 0.096, 0.136) + [[-0.0865, -0.0495, 0, 0.019, -0.050, 0.0]],
        "pack only, low on the back": [[-0.0865, -0.0495, 0, 0.019, -0.050, 0.0]],
        "pack only, v3 position": [[-0.0865, -0.0495, 0, 0.019, -0.035, 0.042]],
    }
    out = {}
    for name, boxes in C.items():
        r = blocked(boxes); out[name] = r
        fr = lambda j: "blocked at stand" if r[j] is None else f"{r[j][0]:+.0f}..{r[j][1]:+.0f} of {r[j][2]:+.0f}..{r[j][3]:+.0f}"
        print(f"{name}\n   poses blocked: head {r['head']*100:.0f} %, leg {r['leg']*100:.0f} % | free range from stand [deg]: neck pitch {fr('neck_pitch')}, head pitch {fr('head_pitch')}, "
              f"head yaw {fr('head_yaw')}, hip pitch {fr('left_hip_pitch')}, hip roll {fr('left_hip_roll')}, hip yaw {fr('left_hip_yaw')}, knee {fr('left_knee')}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "flight_clearance_candidates.json"), "w"), indent=1)
