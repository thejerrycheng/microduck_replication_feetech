#!/usr/bin/env python
"""Record a labelled sit/stand clip: one duck, mean action, stand 2 s -> SIT -> STAND -> SIT -> STAND (2.5 s each).

    uv run --project external/IsaacLab python scripts/record_sitstand.py --policy <policy.pt> --label "fast (0.4 s ramp)" presets=backlash
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
    ap.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--name", default="sitstand_rollout")
    ap.add_argument("--seg", type=float, default=2.5)
    args, extra = ap.parse_known_args()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, overrides=extra)
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.sit_prob = 0.0
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0})
    viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True)
    viz.window.size = (640, 360)
    cfg.sim.visualizer_cfgs = [viz]
    os.makedirs(args.out, exist_ok=True)
    plan = [("stand", 0.0, 2.0), ("SIT", 1.0, args.seg), ("STAND", 0.0, args.seg), ("SIT", 1.0, args.seg), ("STAND", 0.0, args.seg)]
    total = int(sum(d for _, _, d in plan) * 50)
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, video_length=total, video_interval=0,
                                            frame_stride=2, fps=25, output_filename_prefix=f"{args.name}_raw")]
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    term = env.command_manager.get_term("base_velocity")
    obs, _ = env.reset()
    dt = env.step_dt
    t = 0.0
    theta = math.pi * 0.7
    segs = []
    with torch.no_grad():
        for name, flag, dur in plan:
            term._command[:, 0] = flag
            t0 = t
            for _ in range(int(dur / dt)):
                obs, *_ = env.step(policy(obs["policy"]))
                t += dt
                p = robot.data.root_link_pos_w.torch[0]
                theta += 0.2 * dt
                for vz in env.sim.visualizers:
                    vz.set_camera_view((float(p[0]) + 0.5 * math.cos(theta), float(p[1]) + 0.5 * math.sin(theta), 0.25), (float(p[0]), float(p[1]), 0.07))
            segs.append((name, t0, t))
    env.close()
    raw = os.path.join(args.out, f"{args.name}_raw_0000.mp4")
    out = os.path.join(args.out, f"{args.name}.mp4")
    draw = [f"drawtext=text='command {n}':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6:enable='between(t,{a:.2f},{b:.2f})'" for n, a, b in segs]
    if args.label:
        draw.append(f"drawtext=text='{args.label}':x=16:y=h-40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
    os.remove(raw)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
