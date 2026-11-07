#!/usr/bin/env python
"""Roll out a sit/stand checkpoint and report what the trunk actually does.

Exports the checkpoint with play (into ``<run>/exported_<name>/``), then runs it deterministically for one
episode on ``--num_envs`` robots and prints, at 1 s intervals, the median trunk height and tilt, and at
the end the fraction of robots seated upright (|z - sit_z| < 15 mm and tilt < 25 deg), standing
(|z - stand_z| < 15 mm, tilt < 25 deg) or fallen (tilt > 60 deg).

Run::

    uv run --project external/IsaacLab python scripts/eval_posture.py --task Microduck-Sit-Flat-HD1910 \
        --run <run dir> --checkpoint model_750.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Sit-Flat-HD1910")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--out", default=None)
    args, _ = parser.parse_known_args()
    extra = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from microduck_rl.tasks.sitstand_env_cfg import SIT_Z, STAND_Z

    run = args.run.resolve()
    name = Path(args.checkpoint).stem
    export_dir = run / f"exported_{name}"
    policy_path = export_dir / "policy.pt"
    if not policy_path.exists():
        play = Path(__file__).resolve().parent / "play.py"
        subprocess.run(
            [sys.executable, str(play), "--rl_library", "rsl_rl", "--task", args.task, "--checkpoint",
             str(run / args.checkpoint), "--num_envs", "4", "--video", "--video_length", "1", "--video_interval", "1000000", *extra],
            check=True, cwd=Path(__file__).resolve().parent.parent / "external" / "IsaacLab",
        )
        export_dir.mkdir(exist_ok=True)
        for f in ("policy.pt", "policy.onnx", "policy.onnx.data"):
            if (run / "exported" / f).exists():
                shutil.copy(run / "exported" / f, export_dir / f)

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.play_mode()
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    robot = env.scene["robot"]
    obs, _ = env.reset()
    steps = int(args.seconds / env.step_dt)
    per_second = int(round(1.0 / env.step_dt))
    timeline = []
    with torch.no_grad():
        for t in range(steps):
            obs, _, _, _, _ = env.step(policy(obs["policy"]))
            if (t + 1) % per_second == 0 or t == steps - 1:
                z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                g = robot.data.projected_gravity_b.torch
                tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
                flag = env.command_manager.get_command("base_velocity")[:, 0]
                timeline.append({"t_s": round((t + 1) * env.step_dt, 2), "z_median_m": round(float(z.median()), 4),
                                 "tilt_median_deg": round(float(tilt.median()), 1), "tilt_p90_deg": round(float(tilt.quantile(0.9)), 1),
                                 "sit_cmd_frac": round(float(flag.mean()), 2)})
    z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    g = robot.data.projected_gravity_b.torch
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
    seated = ((z - SIT_Z).abs() < 0.015) & (tilt < 25)
    standing = ((z - STAND_Z).abs() < 0.015) & (tilt < 25)
    fallen = tilt > 60
    report = {"checkpoint": str(run / args.checkpoint), "num_envs": args.num_envs, "timeline": timeline,
              "final": {"seated_upright_frac": float(seated.float().mean()), "standing_frac": float(standing.float().mean()),
                        "fallen_frac": float(fallen.float().mean()), "z_median_m": float(z.median()), "tilt_median_deg": float(tilt.median())}}
    for row in timeline:
        print(row)
    print("final:", json.dumps(report["final"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
