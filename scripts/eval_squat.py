#!/usr/bin/env python
"""Does the walk policy squat when it stops or starts a turn? Mean action, 16 robots, full-collision scene.

Phases: walk 0.3 m/s (3 s) -> STOP zero command (3 s) -> TURN 1.0 rad/s in place (3 s) -> STOP (2 s).
Reports per phase the median and minimum trunk height over robots and time (standing = 0.115 m), tilt, falls.

    uv run --project external/IsaacLab python scripts/eval_squat.py --policy checkpoints/walk_allcoll_16498/policy.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

PHASES = [("walk 0.3", (0.3, 0.0, 0.0), 3.0), ("stop", (0.0, 0.0, 0.0), 3.0), ("turn 1.0 rad/s", (0.0, 0.0, 1.0), 3.0), ("stop", (0.0, 0.0, 0.0), 2.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--out", default=None)
    args, hydra = ap.parse_known_args()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=hydra)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0, "standing_z_min": 0.115, "standing_z_max": 0.115})
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    pol = torch.jit.load(args.policy, map_location=env.device).eval()
    obs, _ = env.reset()
    buf = env.command_manager.get_term("base_velocity").command
    report = []
    with torch.no_grad():
        for name, cmd, dur in PHASES:
            zs, tilts = [], []
            for _ in range(int(dur / env.step_dt)):
                buf[:, 0], buf[:, 1], buf[:, 2] = cmd
                obs, *_ = env.step(pol(obs["policy"]))
                z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                g = robot.data.projected_gravity_b.torch
                zs.append(z.clone()); tilts.append(torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))))
            Z = torch.stack(zs); T = torch.stack(tilts)
            # skip the first 0.5 s of each phase (transition) for the steady numbers, but report the dip too
            k = int(0.5 / env.step_dt)
            report.append({"phase": name, "z_median_m": round(float(Z[k:].median()), 4), "z_min_m": round(float(Z.min()), 4),
                           "z_dip_first_0.5s_min_m": round(float(Z[:k].min()), 4), "tilt_median_deg": round(float(T[k:].median()), 1),
                           "fallen_frac": round(float((T[-1] > 60).float().mean()), 3)})
            print(report[-1])
    if args.out:
        json.dump({"policy": args.policy, "phases": report}, open(args.out, "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
