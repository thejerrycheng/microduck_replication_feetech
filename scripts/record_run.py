#!/usr/bin/env python
"""Record a labelled straight-run clip: one duck, mean action, a fixed forward command, following camera, with
the measured forward speed burned in.

    uv run --project external/IsaacLab python scripts/record_run.py --task Microduck-RunV3-Flat-HD1910-AllColl --policy <policy.pt> --vx 0.9 presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
from microduck_rl.viz import play_visualizer_cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-RunV3-Flat-HD1910-AllColl")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--vx", type=float, default=0.9)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--name", default="run_rollout")
    args, extra = ap.parse_known_args()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, overrides=extra)
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True)
    viz.window.size = (640, 360)
    cfg.sim.visualizer_cfgs = [viz]
    os.makedirs(args.out, exist_ok=True)
    total = int(args.seconds * 50)
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, video_length=total, video_interval=0,
                                            frame_stride=2, fps=25, output_filename_prefix=f"{args.name}_raw")]
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    buf = env.command_manager.get_term("base_velocity").command
    obs, _ = env.reset()
    dt = env.step_dt
    speeds = []
    theta = math.pi * 0.55
    fell = False
    with torch.no_grad():
        for k in range(total):
            buf[:] = 0.0
            buf[:, 0] = args.vx
            obs, *_ = env.step(policy(obs["policy"]))
            v = float(robot.data.root_lin_vel_b.torch[0, 0])
            if k >= 50:
                speeds.append(v)
            p = robot.data.root_link_pos_w.torch[0]
            for vz in env.sim.visualizers:
                vz.set_camera_view((float(p[0]) + 0.6 * math.cos(theta), float(p[1]) + 0.6 * math.sin(theta), 0.28), (float(p[0]) + 0.1, float(p[1]), 0.07))
            g = robot.data.projected_gravity_b.torch[0]
            fell |= float(-g[2]) < math.cos(math.radians(60.0))
    env.close()
    mean_v = sum(speeds) / max(1, len(speeds))
    raw = os.path.join(args.out, f"{args.name}_raw_0000.mp4")
    out = os.path.join(args.out, f"{args.name}.mp4")
    label = args.label or os.path.basename(args.policy)
    txt = f"command {args.vx:.1f} m/s   achieved {mean_v:.2f} m/s (mean after 1 s){'   FELL' if fell else ''}"
    draw = [f"drawtext=text='{txt}':x=16:y=16:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6",
            f"drawtext=text='{label}':x=16:y=h-40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
    os.remove(raw)
    print(f"wrote {out}  achieved {mean_v:.3f} m/s fell={fell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
