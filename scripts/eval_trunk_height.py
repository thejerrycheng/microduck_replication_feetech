#!/usr/bin/env python
"""Trunk-height distribution of a velocity policy while it runs a fixed command.

Rolls an exported TorchScript policy with the twist pinned to ``--vx`` on the task's play-mode env and reports
percentiles of the trunk height over the steady part of the rollout (after ``--settle_s``). Written to size the
RunStand v2 expert gate (``gate_trunk_m``), which must sit BELOW anything a healthy running gait reaches.

Run::

    uv run --project external/IsaacLab python scripts/eval_trunk_height.py --task <task> --policy <policy.pt> --vx 1.1 presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-RunStand-Flat-HD1910-AllColl")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--duration_s", type=float, default=8.0)
    ap.add_argument("--settle_s", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args()
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    obs, _ = env.reset()
    buf = env.command_manager.get_term("base_velocity").command
    dt = env.step_dt
    zs, tilts = [], []
    with torch.no_grad():
        for i in range(int(args.duration_s / dt)):
            buf[:] = 0.0
            buf[:, 0] = args.vx
            obs, *_ = env.step(policy(obs["policy"]))
            if i * dt >= args.settle_s:
                z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                g = robot.data.projected_gravity_b.torch
                tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
                zs.append(z.clone())
                tilts.append(tilt.clone())
    z = torch.cat(zs)
    tilt = torch.cat(tilts)
    up = tilt < 30.0  # exclude frames of a fall from the gait statistics
    zu = z[up]
    q = lambda p: round(float(torch.quantile(zu, p)), 4)
    report = {
        "policy": args.policy, "task": args.task, "vx": args.vx, "num_envs": args.num_envs, "frames": int(z.numel()),
        "upright_frac": round(float(up.float().mean()), 3),
        "trunk_min": round(float(zu.min()), 4), "trunk_p01": q(0.01), "trunk_p05": q(0.05), "trunk_p50": q(0.5), "trunk_p95": q(0.95), "trunk_max": round(float(zu.max()), 4),
        "frac_below_0.085": round(float((zu < 0.085).float().mean()), 4), "frac_below_0.095": round(float((zu < 0.095).float().mean()), 4),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
