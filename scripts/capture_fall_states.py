#!/usr/bin/env python
"""Build a bank of REAL fall end-states for the get-up policy's start distribution.

Runs the walk and run policies on the full-collision scene, shoves them (2.0-3.0 m/s velocity impulses in random
directions) until they fall, waits until the body has settled (tilt > 60 deg for 1 s, |v| < 0.05 m/s,
|omega| < 0.5 rad/s), and records root orientation (xyzw), trunk height above the origin and the 14 servo
angles. Output ``docs/fall_bank.json`` (list of dicts), consumed by ``mdp.reset_fallen_state`` (family "bank").

    uv run --project external/IsaacLab python scripts/capture_fall_states.py --target 300 presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

POLICIES = {"walk": ("checkpoints/walk_allcoll_16498/policy.pt", 0.3), "run": ("checkpoints/run_allcoll_4498/policy.pt", 0.6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--out", default="docs/fall_bank.json")
    args, hydra = ap.parse_known_args()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=hydra)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0})
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    servo_ids, servo_names = robot.find_joints([r"^(?!passive_).*"])
    buf = env.command_manager.get_term("base_velocity").command
    n, dt = env.num_envs, env.step_dt
    bank = []
    with torch.no_grad():
        while len(bank) < args.target:
            for name, (path, vx) in POLICIES.items():
                pol = torch.jit.load(path, map_location=env.device).eval()
                obs, _ = env.reset()
                fallen_since = torch.full((n,), -1.0, device=env.device)
                recorded = torch.zeros(n, dtype=torch.bool, device=env.device)
                for t in range(int(8.0 / dt)):
                    buf[:] = 0.0
                    buf[:, 0] = vx
                    if t == int(1.5 / dt):  # one big shove per episode, random direction and size
                        ang = torch.rand(n, device=env.device) * 2 * math.pi
                        mag = 2.0 + torch.rand(n, device=env.device)
                        v = robot.data.root_lin_vel_w.torch.clone()
                        v[:, 0] += mag * torch.cos(ang)
                        v[:, 1] += mag * torch.sin(ang)
                        robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat([v, robot.data.root_ang_vel_w.torch], 1), env_ids=torch.arange(n, device=env.device))
                    obs, *_ = env.step(pol(obs["policy"]))
                    g = robot.data.projected_gravity_b.torch
                    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
                    down = tilt > 60.0
                    now = t * dt
                    fallen_since = torch.where(down & (fallen_since < 0), torch.full_like(fallen_since, now), fallen_since)
                    fallen_since = torch.where(~down, torch.full_like(fallen_since, -1.0), fallen_since)
                    still = (torch.linalg.norm(robot.data.root_lin_vel_w.torch, dim=-1) < 0.05) & (torch.linalg.norm(robot.data.root_ang_vel_w.torch, dim=-1) < 0.5)
                    ready = down & (fallen_since >= 0) & (now - fallen_since > 1.0) & still & ~recorded
                    for i in torch.nonzero(ready).flatten().tolist():
                        q = robot.data.root_link_quat_w.torch[i]
                        z = float(robot.data.root_link_pos_w.torch[i, 2] - env.scene.env_origins[i, 2])
                        bank.append({"source": name, "quat_xyzw": [round(float(v), 5) for v in q], "z": round(z, 4),
                                     "tilt_deg": round(float(tilt[i]), 1), "joint_pos": [round(float(v), 4) for v in robot.data.joint_pos.torch[i, servo_ids]]})
                        recorded[i] = True
                print(f"{name}: bank size {len(bank)}")
                if len(bank) >= args.target:
                    break
    # the joint order must match the action/observation order (left leg, head, right leg)
    json.dump(bank[: args.target], open(args.out, "w"), indent=0)
    tilts = [b["tilt_deg"] for b in bank]
    print("saved", len(bank[: args.target]), "states; tilt deg min/median/max", min(tilts), sorted(tilts)[len(tilts) // 2], max(tilts), "joint order:", servo_names)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
