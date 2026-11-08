#!/usr/bin/env python
"""Jump ceiling, second probe: let the trained SIT policy make a balanced deep crouch, then fire the legs open loop.

The first probe (scripts/probe_jump_ceiling.py) commanded scripted crouches and 575 of 576 toppled before the
extension could be fired: a static joint-ratio crouch is not balanced on this robot. The sit policy IS a balanced
58 mm crouch (z 0.057 m, mean action, 16/16 seated), so each environment now runs the sit policy for ``--crouch_s``
seconds and then holds an extension target ``q_home - gain * (q_crouch - q_home)`` on the leg joints (overshoot past
standing; the BAM servo saturates its PWM on the large error), with or without a head flick. Reports ballistic rise
of the trunk above the standing baseline while both feet are unloaded, air time, peak vertical speed.

    uv run --project external/IsaacLab python scripts/probe_jump_ceiling_v2.py --policy <exported policy.pt> presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

GAINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
HEAD = (0.0, -1.0, 1.0)          # neck/head pitch flick added at extension onset
ANKLE_EXTRA = (0.0, 0.3, -0.3)   # extra ankle push on top of the mirrored target
REPEATS = 4                      # seeds of the sit policy's own variability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Sit-Flat-HD1910")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--crouch_s", type=float, default=3.0)
    parser.add_argument("--hold_s", type=float, default=0.6)
    parser.add_argument("--out", default="docs/jump_ceiling_v2.json")
    args, hydra = parser.parse_known_args()

    grid = list(itertools.product(GAINS, HEAD, ANKLE_EXTRA, range(REPEATS)))
    n = len(grid)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n, overrides=hydra)
    env_cfg.play_mode()
    env_cfg.scene.num_envs = n
    env_cfg.curriculum = None
    env_cfg.episode_length_s = 12.0
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    names = env.action_manager.get_term("joint_pos")._joint_names
    idx = {k: names.index(k) for k in names}
    legs = [idx[f"{s}_{j}"] for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle")]
    ankles = [idx["left_ankle"], idx["right_ankle"]]
    feet = [i for i, b in enumerate(sensor.body_names) if "ankle" in b]
    assert len(feet) == 2, sensor.body_names
    home = robot.data.default_joint_pos.torch[:, : len(names)] if robot.data.default_joint_pos.torch.shape[1] >= len(names) else None
    jids, _ = robot.find_joints(names, preserve_order=True)
    dt = env.step_dt

    def z():
        return robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    def tilt_deg():
        gz = robot.data.projected_gravity_b.torch[:, 2]
        return torch.rad2deg(torch.acos(torch.clamp(-gz, -1.0, 1.0)))

    def airborne():
        f = sensor.data.net_forces_w[:, feet, :].norm(dim=-1)
        return (f < 1.0).all(dim=1)

    obs, _ = env.reset()
    with torch.no_grad():
        # 1 s standing baseline under the policy? No: the sit policy sits immediately, so take the baseline from
        # the default standing pose of the first step.
        base = z().clone()
        for _ in range(int(args.crouch_s / dt)):
            obs, *_ = env.step(policy(obs["policy"]))
        q_home = robot.data.default_joint_pos.torch[:, jids]
        q_crouch = robot.data.joint_pos.torch[:, jids].clone()
        crouch_z = z().clone()
        crouch_tilt = tilt_deg().clone()
        crouch_ok = (crouch_z < base - 0.03) & (crouch_tilt < 25.0) & (~airborne())
        act = torch.zeros(n, len(names), device=env.device)
        meta = []
        for e, (g, head, ank, rep) in enumerate(grid):
            for j in legs:
                act[e, j] = -g * (q_crouch[e, j] - q_home[e, j])  # target = home - g*(crouch-home)
            for j in ankles:
                act[e, j] += ank * torch.sign(q_home[e, j])
            act[e, idx["neck_pitch"]] = head
            act[e, idx["head_pitch"]] = head
            meta.append({"gain": g, "head": head, "ankle_extra": ank, "rep": rep})
        rise = torch.zeros(n, device=env.device); run = torch.zeros(n, device=env.device)
        air = torch.zeros(n, device=env.device); vz = torch.zeros(n, device=env.device); mt = torch.zeros(n, device=env.device)
        for _ in range(int(args.hold_s / dt)):
            env.step(act)
            a = airborne()
            rise = torch.where(a, torch.maximum(rise, z() - base), rise)
            run = torch.where(a, run + 1, torch.zeros_like(run)); air = torch.maximum(air, run)
            vz = torch.maximum(vz, robot.data.root_lin_vel_w.torch[:, 2]); mt = torch.maximum(mt, tilt_deg())
    rows = []
    for e in range(n):
        rows.append({**meta[e], "crouch_ok": bool(crouch_ok[e]), "crouch_drop_mm": round(float(base[e] - crouch_z[e]) * 1000, 1),
                     "rise_mm": round(float(rise[e]) * 1000, 1), "air_time_s": round(float(air[e]) * dt, 2),
                     "vz_peak_m_s": round(float(vz[e]), 3), "max_tilt_deg": round(float(mt[e]), 1)})
    valid = sorted([r for r in rows if r["crouch_ok"]], key=lambda r: -r["rise_mm"])
    best_by = {}
    for key in ("gain", "head", "ankle_extra"):
        for r in valid:
            k = f"{key}={r[key]}"
            if k not in best_by or r["rise_mm"] > best_by[k]["rise_mm"]:
                best_by[k] = r
    report = {"task": args.task, "policy": args.policy, "overrides": hydra, "num_variants": n, "num_valid_crouches": len(valid),
              "crouch_q_minus_home_example": [round(float(v), 3) for v in (q_crouch[0] - q_home[0])],
              "top10": valid[:10], "best_by": best_by, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"valid crouches {len(valid)}/{n}; example crouch-home:", report["crouch_q_minus_home_example"])
    for r in valid[:10]:
        print(r)
    print("best by:", json.dumps(best_by))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
