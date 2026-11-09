#!/usr/bin/env python
"""Sit/stand TRANSITION time, deterministic (mean action): stand 3 s, command SIT, time to reach the seated band
(trunk z < sit_z + 0.012), hold 3 s, command STAND, time to the standing band (z > stand_z - 0.010). Medians and
p90 over envs, plus falls.

    uv run --project external/IsaacLab python scripts/eval_transition.py --policy <policy.pt> presets=backlash --out docs/transition_<name>.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--out", default=None)
    args, extra = ap.parse_known_args()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=extra)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.sit_prob = 0.0
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0})
    sit_z, stand_z = float(cfg.commands.base_velocity.sit_z), float(cfg.commands.base_velocity.stand_z)
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    term = env.command_manager.get_term("base_velocity")
    obs, _ = env.reset()
    n, dt = env.num_envs, env.step_dt
    z0 = env.scene.env_origins[:, 2]

    def trunk_z():
        return robot.data.root_link_pos_w.torch[:, 2] - z0

    def tilt_deg():
        g = robot.data.projected_gravity_b.torch
        return torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))

    results = {}
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    with torch.no_grad():
        for phase, flag, target, dur in (("settle", 0.0, None, 3.0), ("sit", 1.0, "sit", 3.0), ("stand", 0.0, "stand", 3.0)):
            term._command[:, 0] = flag
            reached = torch.full((n,), float("nan"), device=env.device)
            for k in range(int(dur / dt)):
                obs, *_ = env.step(policy(obs["policy"]))
                z = trunk_z()
                fell |= tilt_deg() > 60.0
                if target == "sit":
                    hit = z < sit_z + 0.012
                elif target == "stand":
                    hit = z > stand_z - 0.010
                else:
                    hit = torch.zeros(n, dtype=torch.bool, device=env.device)
                reached = torch.where(hit & torch.isnan(reached), torch.full_like(reached, (k + 1) * dt), reached)
            if target is not None:
                ok = ~torch.isnan(reached)
                t = reached[ok]
                results[target] = {
                    "reached_frac": round(float(ok.float().mean()), 3),
                    "time_s": {"median": round(float(t.median()), 2) if t.numel() else None, "p90": round(float(t.quantile(0.9)), 2) if t.numel() else None,
                               "max": round(float(t.max()), 2) if t.numel() else None},
                    "z_end_median_m": round(float(z.median()), 4),
                }
                print(target, results[target])
    results["fell_frac"] = round(float(fell.float().mean()), 3)
    print("fell", results["fell_frac"])
    env.close()
    if args.out:
        json.dump({"policy": args.policy, "num_envs": n, "sit_z": sit_z, "stand_z": stand_z, **results}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
