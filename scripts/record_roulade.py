#!/usr/bin/env python
"""Record a labelled front-flip (roulade) clip: one duck, standing start, mean action, side camera, with the
rotation reached and whether it ended standing burned in. Repeats the 5 s episode ``--takes`` times in one clip.

    uv run --project external/IsaacLab python scripts/record_roulade.py --run <run dir> --checkpoint model_2000.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
from microduck_rl.viz import play_visualizer_cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-Roulade-Flat-HD1910")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default="videos")
    args, extra = ap.parse_known_args()
    run = args.run.resolve()
    name = args.checkpoint.replace(".pt", "").replace("model_", "")
    policy_path = run / f"exported_model_{name}" / "policy.pt"
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, overrides=extra)
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True)
    viz.window.size = (640, 360)
    cfg.sim.visualizer_cfgs = [viz]
    os.makedirs(args.out, exist_ok=True)
    steps = int(args.seconds * 50)
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, video_length=steps * args.takes, video_interval=0,
                                            frame_stride=2, fps=25, output_filename_prefix=f"roulade_{name}_raw")]
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    dt = env.step_dt
    segs = []
    t = 0.0
    with torch.no_grad():
        for take in range(args.takes):
            obs, _ = env.reset()
            t0 = t
            for _ in range(steps):
                obs, *_ = env.step(policy(obs["policy"]))
                t += dt
                p = robot.data.root_link_pos_w.torch[0]
                # side view: the roll is about the body y axis, so look from +y
                for vz in env.sim.visualizers:
                    vz.set_camera_view((float(p[0]) + 0.05, float(p[1]) + 0.65, 0.22), (float(p[0]) + 0.05, float(p[1]), 0.08))
            rot = float(torch.rad2deg(env._roulade_max)[0])
            g = robot.data.projected_gravity_b.torch[0]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
            z = float(p[2] - env.scene.env_origins[0, 2])
            standing = tilt < 20.0 and z > 0.095
            segs.append((take, t0, t, rot, standing))
            print(f"take {take}: rotation {rot:.0f} deg, standing at end {standing}")
    env.close()
    raw = os.path.join(args.out, f"roulade_{name}_raw_0000.mp4")
    out = os.path.join(args.out, f"roulade_{name}.mp4")
    draw = [f"drawtext=text='take {k + 1}  rolled {rot:.0f} deg  {'landed standing' if st else 'did not land standing'}':x=16:y=16:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6:enable='between(t,{a:.2f},{b:.2f})'" for k, a, b, rot, st in segs]
    draw.append(f"drawtext=text='front flip (roulade) ckpt {name}, standing start, mean action':x=16:y=h-40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
    os.remove(raw)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
