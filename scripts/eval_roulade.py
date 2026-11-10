#!/usr/bin/env python
"""Deterministic (mean-action) evaluation of a roulade checkpoint from STANDING starts.

Exports the checkpoint with play (``<run>/exported_model_<N>/``), then rolls out ``--num_envs`` robots for one
5 s episode in play mode and reports: rotation frontier distribution [deg], fraction that rolled past 330 deg
over the head, fraction that ALSO ended upright at standing height (the deployable success), peak |omega_y|.

    uv run --project external/IsaacLab python scripts/eval_roulade.py --run <run dir> --checkpoint model_2000.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from microduck_rl.mdp import roulade as R  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Roulade-Flat-HD1910")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=4.9)
    parser.add_argument("--out", default=None)
    args, extra = parser.parse_known_args()
    run = args.run.resolve()
    name = args.checkpoint.replace(".pt", "").replace("model_", "")
    export_dir = run / f"exported_model_{name}"
    policy_path = export_dir / "policy.pt"
    if not policy_path.exists():
        play = Path(__file__).resolve().parent / "play.py"
        subprocess.run(
            [sys.executable, str(play), "--rl_library", "rsl_rl", "--task", args.task, "--checkpoint", str(run / args.checkpoint),
             "--num_envs", "4", "--video", "--video_length", "1", "--video_interval", "1000000", *extra],
            check=True, cwd=Path(__file__).resolve().parent.parent / "external" / "IsaacLab",
        )
        export_dir.mkdir(exist_ok=True)
        for f in ("policy.pt", "policy.onnx", "policy.onnx.data"):
            if (run / "exported" / f).exists():
                shutil.copy(run / "exported" / f, export_dir / f)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=extra)
    env_cfg.play_mode()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    robot = env.scene["robot"]
    obs, _ = env.reset()
    steps = int(args.seconds / env.step_dt)
    peak_omega = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for _ in range(steps):
            obs, *_ = env.step(policy(obs["policy"]))
            peak_omega = torch.maximum(peak_omega, robot.data.root_link_ang_vel_b.torch[:, 1].abs())
    from microduck_rl.mdp.roulade import roulade_head_error
    from isaaclab.managers import SceneEntityCfg as _SE
    head_cfg = _SE("robot", joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll"]); head_cfg.resolve(env.scene)
    head_err = roulade_head_error(env, head_cfg)
    frontier = torch.rad2deg(env._roulade_max)
    head = env._roulade_head_latch
    z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(R._upright(robot), -1.0, 1.0)))
    completed = (frontier >= 330.0) & head
    standing = (tilt < 20.0) & (z > 0.095)
    report = {
        "checkpoint": str(run / args.checkpoint), "num_envs": args.num_envs, "seconds": args.seconds,
        "frontier_deg": {"min": round(float(frontier.min()), 1), "p10": round(float(frontier.quantile(0.1)), 1),
                          "median": round(float(frontier.median()), 1), "p90": round(float(frontier.quantile(0.9)), 1), "max": round(float(frontier.max()), 1)},
        "head_latched_frac": round(float(head.float().mean()), 3),
        "completed_frac": round(float(completed.float().mean()), 3),
        "completed_and_standing_frac": round(float((completed & standing).float().mean()), 3),
        "final_standing_frac": round(float(standing.float().mean()), 3),
        "final_head_err_rad": {"median": round(float(head_err.median()), 3), "p90": round(float(head_err.quantile(0.9)), 3), "max": round(float(head_err.max()), 3)},
        "final_head_err_deg_median": round(float(torch.rad2deg(head_err.median())), 1),
        "final_z_median_m": round(float(z.median()), 4), "final_tilt_median_deg": round(float(tilt.median()), 1),
        "peak_abs_omega_y_median": round(float(peak_omega.median()), 2),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
