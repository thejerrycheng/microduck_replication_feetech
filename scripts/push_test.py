#!/usr/bin/env python
"""Push-disturbance test: shove the duck while it walks or runs, measure falls and recovery, record a clip.

Pushes are velocity impulses on the trunk (what training's ``push_by_setting_velocity`` does), in a random
horizontal direction, escalating through ``--levels`` [m/s]; each level is applied ``--repeats`` times, 3 s apart.
Statistics on ``--num_envs`` robots (no video): fraction fallen (tilt > 60 deg) within 2 s of each push and
fraction back upright (tilt < 20 deg, trunk > 0.095 m) within 2 s. Then one robot is recorded with the camera
following, pushes labelled in the clip. Full-collision scene, standing start, mean actions.

    uv run --project external/IsaacLab python scripts/push_test.py --policy walk --vx 0.3 presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
from microduck_rl.viz import play_visualizer_cfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

POLICIES = {
    "walk": "checkpoints/walk_allcoll_16498/policy.pt",
    "run": "checkpoints/run_allcoll_4498/policy.pt",
    "walk_reduced": "checkpoints/walk_sym_15000/policy.pt",
    "stand": "checkpoints/sitstand_4000/policy.pt",  # sit/stand policy with the flag at 0 = standing hold
    "stand_final": "logs/rsl_rl/microduck_sitstand_flat_hd1910_backlash/2026-11-08_16-59-09/exported_model_15000/policy.pt",
}


def make_env(task, n, hydra, video_dir=None, video_steps=0):
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=n, overrides=hydra)
    cfg.play_mode()
    cfg.scene.num_envs = n
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0, "standing_z_min": 0.115, "standing_z_max": 0.115})
    if video_dir:
        viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True, enable_markers=True)
        viz.window.size = (854, 480)
        cfg.sim.visualizer_cfgs = [viz]
        cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=video_dir, video_length=video_steps, video_interval=0, frame_stride=2, fps=25, output_filename_prefix="push_raw")]
    return gym.make(task, cfg=cfg).unwrapped


def tilt_deg(robot):
    g = robot.data.projected_gravity_b.torch
    return torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    ap.add_argument("--policy", default="walk", choices=list(POLICIES))
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--levels", default="0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--out", default="rollouts/push")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_video", action="store_true")
    args, hydra = ap.parse_known_args()
    levels = [float(v) for v in args.levels.split(",")]
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dt = 0.02
    period = int(3.0 / dt)
    schedule = [(1.0 + k * 3.0, lvl) for k, lvl in enumerate([l for l in levels for _ in range(args.repeats)])]
    total_s = schedule[-1][0] + 3.0

    def run(env, record):
        """record=True: one continuous take for the clip. record=False: statistics, RESET before every push so
        each level starts upright (without this a fall at a low level is counted again at every later one)."""
        robot = env.scene["robot"]
        arrow = VisualizationMarkers(RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/push_arrow")) if record else None
        push_vis = {"until": -1.0, "dir": 0.0, "level": 0.0}
        pol = torch.jit.load(POLICIES[args.policy], map_location=env.device).eval()
        obs, _ = env.reset()
        buf = env.command_manager.get_term("base_velocity").command
        n = env.num_envs
        stats = []
        theta = math.pi * 0.75
        pending = None
        t = 0.0
        steps = int(total_s / dt)
        with torch.no_grad():
            for i in range(steps):
                buf[:] = 0.0
                buf[:, 0] = args.vx
                for t_push, lvl in schedule:
                    if not record and abs(t - (t_push - 1.0)) < dt / 2:
                        obs, _ = env.reset()  # fresh upright start, 1 s of walking/standing before the push
                    if abs(t - t_push) < dt / 2:
                        ang = torch.rand(n, device=env.device) * 2 * math.pi
                        v = robot.data.root_lin_vel_w.torch.clone()
                        v[:, 0] += lvl * torch.cos(ang)
                        v[:, 1] += lvl * torch.sin(ang)
                        vel = torch.cat([v, robot.data.root_ang_vel_w.torch], dim=1)
                        robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=torch.arange(n, device=env.device))
                        pending = {"t": t, "level": lvl, "fell": torch.zeros(n, dtype=torch.bool, device=env.device)}
                        push_vis = {"until": t + 0.8, "dir": float(ang[0]), "level": lvl}
                obs, *_ = env.step(pol(obs["policy"]))
                t += dt
                tl = tilt_deg(robot)
                z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                if pending is not None:
                    pending["fell"] |= tl > 60.0
                    if t - pending["t"] >= 2.0 - dt / 2:
                        up = (tl < 20.0) & (z > 0.095)
                        stats.append({"t": round(pending["t"], 2), "level": pending["level"], "fell_frac": round(float(pending["fell"].float().mean()), 3),
                                      "upright_after_2s_frac": round(float(up.float().mean()), 3), "recovered_after_fall_frac": round(float((pending["fell"] & up).float().sum() / max(1.0, float(pending["fell"].float().sum()))), 3)})
                        pending = None
                if record:
                    p = robot.data.root_link_pos_w.torch[0]
                    if arrow is not None:
                        if t < push_vis["until"]:
                            # arrow tail at the push origin, pointing along the push, length ~ magnitude
                            L = 0.12 + 0.12 * push_vis["level"]
                            d = push_vis["dir"]
                            tail = torch.tensor([[float(p[0]) - L * math.cos(d), float(p[1]) - L * math.sin(d), float(p[2]) + 0.06]], device=env.device)
                            q = math_utils.quat_from_euler_xyz(torch.zeros(1, device=env.device), torch.zeros(1, device=env.device), torch.tensor([d], device=env.device))
                            arrow.set_visibility(True)
                            arrow.visualize(tail, q, torch.tensor([[L, 0.06 + 0.02 * push_vis["level"], 0.06 + 0.02 * push_vis["level"]]], device=env.device))
                        else:
                            arrow.set_visibility(False)
                    theta += 0.25 * dt
                    for viz in env.sim.visualizers:
                        viz.set_camera_view((float(p[0]) + 0.8 * math.cos(theta), float(p[1]) + 0.8 * math.sin(theta), 0.35), (float(p[0]), float(p[1]), 0.07))
        return stats

    env = make_env(args.task, args.num_envs, hydra)
    stats = run(env, record=False)
    env.close()
    by_level = {}
    for s in stats:
        by_level.setdefault(s["level"], []).append(s)
    summary = {lvl: {"fell_frac": round(sum(x["fell_frac"] for x in xs) / len(xs), 3), "upright_after_2s_frac": round(sum(x["upright_after_2s_frac"] for x in xs) / len(xs), 3),
                     "recovered_after_fall_frac": round(sum(x["recovered_after_fall_frac"] for x in xs) / len(xs), 3)} for lvl, xs in by_level.items()}
    print("policy", args.policy, "vx", args.vx, json.dumps(summary, indent=1))
    with open(os.path.join(args.out, f"push_{args.policy}_vx{args.vx}.json"), "w") as f:
        json.dump({"policy": args.policy, "vx": args.vx, "num_envs": args.num_envs, "per_push": stats, "by_level": summary}, f, indent=1)

    if args.no_video:
        return 0
    env = make_env(args.task, 1, hydra, video_dir=args.out, video_steps=int(total_s / dt) + 2)
    run(env, record=True)
    env.close()
    raw = os.path.join(args.out, "push_raw_0000.mp4")
    out = os.path.join(args.out, f"push_{args.policy}_vx{args.vx}.mp4")
    draw = [f"drawtext=text='{args.policy.upper()} policy, command {args.vx} m/s':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8"]
    for t_push, lvl in schedule:
        draw.append(f"drawtext=text='PUSH {lvl:.1f} m/s impulse (red arrow)':x=20:y=64:fontsize=34:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=8:enable='between(t,{t_push:.2f},{t_push + 1.0:.2f})'")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], check=True)
    os.replace(raw, os.path.join(args.out, f"push_{args.policy}_vx{args.vx}_raw.mp4"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
