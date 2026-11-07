#!/usr/bin/env python
"""Measure left/right gait symmetry of a walking checkpoint.

Rolls the exported TorchScript policy (``exported/policy.pt`` next to the checkpoint, written by play)
deterministically with a fixed forward command and reports, per foot, mean air time, mean contact time
and step count, and per leg the hip-pitch / knee / ankle motion amplitude. A balanced gait gives
left/right ratios near 1.0.

Run::

    uv run --project external/IsaacLab python scripts/eval_gait_symmetry.py --run <run dir> presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Velocity-Flat-HD1910")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", default=None, help="TorchScript policy (default: <run>/exported/policy.pt)")
    parser.add_argument("--checkpoint", default=None, help="model_N.pt in --run: export it first (play) into <run>/exported_N/")
    parser.add_argument("--min_flight_s", type=float, default=0.05, help="Touchdowns after shorter flights are contact chatter, not steps")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    args, _ = parser.parse_known_args()

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    policy_path = Path(args.policy) if args.policy else args.run / "exported" / "policy.pt"
    if args.checkpoint:
        import shutil
        import subprocess

        name = Path(args.checkpoint).stem
        export_dir = args.run / f"exported_{name}"
        policy_path = export_dir / "policy.pt"
        if not policy_path.exists():
            play = Path(__file__).resolve().parent / "play.py"
            extra = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
            subprocess.run(
                [sys.executable, str(play), "--rl_library", "rsl_rl", "--task", args.task, "--checkpoint",
                 str(args.run.resolve() / args.checkpoint), "--num_envs", "4", "--video", "--video_length", "1", "--video_interval", "1000000", *extra],
                check=True, cwd=Path(__file__).resolve().parent.parent / "external" / "IsaacLab",
            )
            export_dir.mkdir(exist_ok=True)
            for f in ("policy.pt", "policy.onnx", "policy.onnx.data"):
                src = args.run / "exported" / f
                if src.exists():
                    shutil.copy(src, export_dir / f)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.play_mode()
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    foot_ids, _ = sensor.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
    leg = {}
    for side in ("left", "right"):
        ids, _ = robot.find_joints([f"{side}_hip_pitch", f"{side}_knee", f"{side}_ankle"], preserve_order=True)
        leg[side] = ids
    cmd = env.command_manager.get_term("base_velocity")
    robot_foot_ids, _ = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)

    obs, _ = env.reset()
    steps = int(args.seconds / env.step_dt)
    air, contact, touchdowns, foot_xy, last_air = [], [], [], [], []
    qmin = {s: None for s in leg}
    qmax = {s: None for s in leg}
    with torch.no_grad():
        for _ in range(steps):
            cmd.vel_command_b[:, 0] = args.vx
            cmd.vel_command_b[:, 1:] = 0.0
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            air.append(sensor.data.current_air_time.torch[:, foot_ids].clone())
            contact.append(sensor.data.current_contact_time.torch[:, foot_ids].clone())
            touchdowns.append(sensor.compute_first_contact(env.step_dt).torch[:, foot_ids].clone())
            foot_xy.append(robot.data.body_link_pos_w.torch[:, robot_foot_ids, :2].clone())
            last_air.append(sensor.data.last_air_time.torch[:, foot_ids].clone())
            for s, ids in leg.items():
                q = robot.data.joint_pos.torch[:, ids]
                qmin[s] = q if qmin[s] is None else torch.minimum(qmin[s], q)
                qmax[s] = q if qmax[s] is None else torch.maximum(qmax[s], q)
    air = torch.stack(air)
    contact = torch.stack(contact)
    steps_per_foot = ((torch.stack(touchdowns) > 0.5) & (torch.stack(last_air) >= args.min_flight_s)).float().sum(dim=0)
    # Mean duration of completed phases: average the per-step value where the phase is active.
    def phase_mean(x):
        active = x > 0
        return (x * active).sum(dim=0) / active.sum(dim=0).clamp_min(1)
    air_m, con_m = phase_mean(air).mean(dim=0), phase_mean(contact).mean(dim=0)
    amp = {s: (qmax[s] - qmin[s]).mean(dim=0) for s in leg}
    # Stride length: distance between successive touchdown positions of the same foot; its coefficient of
    # variation across strides is the step-consistency number (0 = every step the same length).
    td = (torch.stack(touchdowns) > 0.5) & (torch.stack(last_air) >= args.min_flight_s)  # real steps only
    xy = torch.stack(foot_xy)  # (T, envs, 2, 2)
    stride = {"left": [], "right": []}
    for e in range(xy.shape[1]):
        for f, side in enumerate(("left", "right")):
            idx = torch.nonzero(td[:, e, f]).flatten()
            if len(idx) > 2:
                pts = xy[idx, e, f]
                stride[side].append((pts[1:] - pts[:-1]).norm(dim=-1))
    stride_stats = {}
    for side, lst in stride.items():
        d = torch.cat(lst) if lst else torch.zeros(1)
        stride_stats[side] = {"mean_m": float(d.mean()), "std_m": float(d.std()), "cv": float(d.std() / d.mean().clamp_min(1e-6))}
    report = {
        "checkpoint_policy": str(policy_path),
        "vx_cmd": args.vx,
        "air_time_s": {"left": float(air_m[0]), "right": float(air_m[1])},
        "contact_time_s": {"left": float(con_m[0]), "right": float(con_m[1])},
        "steps_per_episode": {"left": float(steps_per_foot[:, 0].mean()), "right": float(steps_per_foot[:, 1].mean())},
        "stride_length": stride_stats,
        "joint_amplitude_rad": {
            s: dict(zip(("hip_pitch", "knee", "ankle"), [round(float(v), 3) for v in amp[s]])) for s in leg
        },
    }
    report["ratios_left_over_right"] = {
        "air_time": report["air_time_s"]["left"] / max(report["air_time_s"]["right"], 1e-6),
        "contact_time": report["contact_time_s"]["left"] / max(report["contact_time_s"]["right"], 1e-6),
        "steps": report["steps_per_episode"]["left"] / max(report["steps_per_episode"]["right"], 1e-6),
        "knee_amplitude": amp["left"][1].item() / max(amp["right"][1].item(), 1e-6),
        "stride_length": stride_stats["left"]["mean_m"] / max(stride_stats["right"]["mean_m"], 1e-6),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
