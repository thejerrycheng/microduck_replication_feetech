#!/usr/bin/env python
"""Deterministic (mean-action) evaluation of a jump checkpoint from STANDING starts, one episode.

Reports valid-takeoff rate, rise above the standing baseline [mm] (min/median/p90/max), air time, stable- and
durable-landing fractions, body-contact rate and peak foot force -- the numbers the acceptance of a jump is
written against. Exports the checkpoint with play first (``<run>/exported_model_<N>/``).

    uv run --project external/IsaacLab python scripts/eval_jump.py --run <run dir> --checkpoint model_1000.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from microduck_rl.mdp import jump as J  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Jump-Flat-HD1910")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=32)
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
    obs, _ = env.reset()
    steps = env.max_episode_length - 1
    peak_foot = torch.zeros(env.num_envs, device=env.device)
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    snapshot = None
    with torch.no_grad():
        for t in range(steps):
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
            s = J.update_state(env)
            f = env.scene.sensors[J.FEET_SENSOR].data.net_forces_w.torch[:, J._ids(env)["feet_rows"], :]
            peak_foot = torch.maximum(peak_foot, torch.linalg.norm(f, dim=-1).sum(dim=1))
            if t == steps - 1:
                snapshot = {k: v.clone() for k, v in s.items() if isinstance(v, torch.Tensor)}
            ended |= term
    s = snapshot
    rise = s["rise_frontier"] * 1000.0
    report = {
        "checkpoint": str(run / args.checkpoint), "num_envs": args.num_envs,
        "valid_takeoff_rate": round(float(s["valid_takeoff"].float().mean()), 3),
        "rise_mm": {"min": round(float(rise.min()), 1), "median": round(float(rise.median()), 1), "p90": round(float(rise.quantile(0.9)), 1), "max": round(float(rise.max()), 1)},
        "airtime_ms_median": round(float(s["airtime_frontier"].median()) * 1000.0, 0),
        "stable_landing_rate": round(float(s["stable_latched"].float().mean()), 3),
        "durable_rate": round(float(s["durable"].float().mean()), 3),
        "failed_rate": round(float(s["failed"].float().mean()), 3),
        "body_contact_rate": round(float(s["body_contact_ever"].float().mean()), 3),
        "terminated_early_rate": round(float(ended.float().mean()), 3),
        "peak_foot_force_N": {"median": round(float(peak_foot.median()), 1), "max": round(float(peak_foot.max()), 1)},
    }
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
