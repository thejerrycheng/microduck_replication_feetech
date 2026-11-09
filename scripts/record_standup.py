#!/usr/bin/env python
"""Record a labelled get-up rollout clip: one duck, mean action, each start family in turn, following camera.

    uv run --project external/IsaacLab python scripts/record_standup.py --run <run dir> --checkpoint model_2500.pt presets=backlash

Families (5 s each, no pushes unless named): prone, supine, side, random, bank, standing_pushed (1.0 m/s shoves every
1.5 s). Frames are held in RAM until the clip closes, so the clip is small (640x360, 25 fps) and short.
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
from microduck_rl.viz import play_visualizer_cfg  # noqa: E402

from eval_standup import FAMILY_PARAMS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-StandUp-Flat-HD1910")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--families", default="prone,supine,side,random,bank,standing_pushed")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--push", type=float, default=1.0)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--zero_twist", action="store_true")
    ap.add_argument("--size", default="640x360")
    args, extra = ap.parse_known_args()
    run = args.run.resolve()
    name = args.checkpoint.replace(".pt", "").replace("model_", "")
    export_dir = run / f"exported_model_{name}"
    policy_path = export_dir / "policy.pt"
    if not policy_path.exists():
        play = Path(__file__).resolve().parent / "play.py"
        subprocess.run([sys.executable, str(play), "--rl_library", "rsl_rl", "--task", args.task, "--checkpoint", str(run / args.checkpoint),
                        "--num_envs", "4", "--video", "--video_length", "1", "--video_interval", "1000000", *extra],
                       check=True, cwd=Path(__file__).resolve().parent.parent / "external" / "IsaacLab")
        export_dir.mkdir(exist_ok=True)
        for f in ("policy.pt", "policy.onnx", "policy.onnx.data"):
            if (run / "exported" / f).exists():
                shutil.copy(run / "exported" / f, export_dir / f)
    families = args.families.split(",")
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, overrides=extra)
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    if args.zero_twist and hasattr(cfg.commands.base_velocity, "ranges"):
        rng = cfg.commands.base_velocity.ranges
        for n in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            if hasattr(rng, n):
                setattr(rng, n, (0.0, 0.0))
        if hasattr(cfg.commands.base_velocity, "rel_standing_envs"):
            cfg.commands.base_velocity.rel_standing_envs = 1.0
    w, h = (int(v) for v in args.size.split("x"))
    viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True)
    viz.window.size = (w, h)
    cfg.sim.visualizer_cfgs = [viz]
    os.makedirs(args.out, exist_ok=True)
    steps_per = int(args.seconds * 50)
    total = steps_per * len(families)
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, video_length=total, video_interval=0,
                                            frame_stride=2, fps=25, output_filename_prefix="standup_raw")]
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    term = env.event_manager.get_term_cfg("set_fallen_state")
    dt = env.step_dt
    segs = []
    theta = math.pi * 0.75
    t = 0.0
    with torch.no_grad():
        for fam in families:
            term.params.update(FAMILY_PARAMS[fam])
            obs, _ = env.reset()
            t0 = t
            stood = None
            for k in range(steps_per):
                if fam == "standing_pushed" and k % 75 == 25:
                    ang = torch.rand(1, device=env.device) * 2 * math.pi
                    v = robot.data.root_lin_vel_w.torch.clone()
                    v[:, 0] += args.push * torch.cos(ang)
                    v[:, 1] += args.push * torch.sin(ang)
                    robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat([v, robot.data.root_ang_vel_w.torch], 1), env_ids=torch.arange(1, device=env.device))
                obs, *_ = env.step(policy(obs["policy"]))
                t += dt
                p = robot.data.root_link_pos_w.torch[0]
                theta += 0.25 * dt
                eye = (float(p[0]) + 0.55 * math.cos(theta), float(p[1]) + 0.55 * math.sin(theta), 0.30)
                for vz in env.sim.visualizers:
                    vz.set_camera_view(eye, (float(p[0]), float(p[1]), 0.06))
                g = robot.data.projected_gravity_b.torch[0]
                up = float(-g[2]) > math.cos(math.radians(20.0)) and float(p[2] - env.scene.env_origins[0, 2]) > 0.095
                if up and stood is None and fam != "standing_pushed":
                    stood = k * dt
            segs.append((fam, t0, t, stood))
            print(fam, "stood at", stood)
    env.close()
    raw = os.path.join(args.out, "standup_raw_0000.mp4")
    labelled = os.path.join(args.out, f"standup_{name}_rollouts.mp4")
    draw = []
    for fam, a, b, stood in segs:
        txt = {"prone": "face down", "supine": "face up", "side": "on the side", "random": "random pose + joints",
               "bank": "real fall pose (bank)", "standing_pushed": f"standing, {args.push:.1f} m/s shoves"}[fam]
        extra_txt = f"  up in {stood:.2f} s" if stood is not None else ("" if fam == "standing_pushed" else "  did not stand")
        draw.append(f"drawtext=text='{txt}{extra_txt}':x=16:y=16:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6:enable='between(t,{a:.2f},{b:.2f})'")
    draw.append(f"drawtext=text='get-up policy ckpt {name}, mean action':x=16:y=h-40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", labelled], check=True)
    os.remove(raw)
    print("wrote", labelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
