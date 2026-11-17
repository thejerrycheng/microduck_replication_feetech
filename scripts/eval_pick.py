#!/usr/bin/env python
"""Deterministic (mean-action) evaluation of a pick checkpoint: walk to a randomised object, latch it with the beak,
lift it. Reports grasp rate, lift rate (held and raised >= --lift_m above where it lay at the end), time to grasp,
falls and the camera-seen fraction. Exports the checkpoint with play first (``<run>/exported_model_<N>/``).

    uv run --project external/IsaacLab python scripts/eval_pick.py --run <run dir> --checkpoint model_1000.pt presets=backlash
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
from microduck_rl.mdp import pick  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Pick-Flat-HD1910-AllColl")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--lift_m", type=float, default=0.04)
    parser.add_argument("--out", default=None)
    args, extra = parser.parse_known_args()
    run = args.run.resolve()
    name = args.checkpoint.replace("model_", "").replace(".pt", "")
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
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    obs, _ = env.reset()
    st = pick._state(env)
    n = env.num_envs
    steps = env.max_episode_length - 1
    dt = env.step_dt
    grasp_step = torch.full((n,), -1, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    seen_steps = torch.zeros(n, device=env.device)
    ended = torch.zeros(n, dtype=torch.bool, device=env.device)
    with torch.no_grad():
        for t in range(steps):
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
            newly = (grasp_step < 0) & st.grasped & ~ended
            grasp_step[newly] = t
            seen_steps += st.seen.float() * (~ended).float()
            fell |= term & ~ended
            ended |= term | trunc
            if t == steps - 2:
                obj_z = env.scene["object"].data.root_link_pos_w.torch[:, 2]
                lifted = st.grasped & ((obj_z - st.spawn_z) >= args.lift_m)
                held = st.grasped.clone()
    g = grasp_step >= 0
    res = {
        "run": str(run), "checkpoint": args.checkpoint, "num_envs": n,
        "grasp_rate": float(g.float().mean()),
        "held_at_end": float(held.float().mean()),
        "lifted_at_end": float(lifted.float().mean()),
        "fell": float(fell.float().mean()),
        "time_to_grasp_s": {"median": float((grasp_step[g].float() * dt).median()) if g.any() else None,
                            "p90": float(torch.quantile(grasp_step[g].float() * dt, 0.9)) if g.any() else None},
        "seen_frac": float((seen_steps / steps).mean()),
        "object_mass_kg": {"min": float(env.scene["object"].data.body_mass.torch[:, 0].min()), "max": float(env.scene["object"].data.body_mass.torch[:, 0].max())},
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    env.close()


if __name__ == "__main__":
    main()
