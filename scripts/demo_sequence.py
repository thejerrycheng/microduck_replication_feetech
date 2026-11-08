#!/usr/bin/env python
"""Operator-style demo: one duck, three policies, a scripted command sequence, a camera that follows and orbits.

Segments (what a gamepad user would do): walk forward -> run forward -> stop -> sit -> stand up -> turn 180 deg
-> run forward -> walk a circle at constant yaw rate. Policies are hot-swapped by writing the 3-wide twist slot
(the sit/stand policy reads a 0/1 flag there; the velocity policies read vx, vy, yaw rate) and switching the
network, exactly as the runtime does. Mean actions throughout. The running policy only runs straight (the
official sprint recipe commands no yaw), so turning and the circle use the walking policy.

    uv run --project external/IsaacLab python scripts/demo_sequence.py presets=backlash --out rollouts/demo
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
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

WALK = "logs/rsl_rl/microduck_walk_flat_symmetry_hd1910_backlash/2026-11-07_09-39-14/exported_model_15000/policy.pt"
RUN = "logs/rsl_rl/microduck_run_flat_hd1910_backlash/2026-11-07_19-05-22/exported_model_2999/policy.pt"
SITSTAND = "logs/rsl_rl/microduck_sitstand_flat_hd1910_backlash/2026-11-08_09-43-28/exported_model_4000/policy.pt"

# (label, policy, command [slot0, slot1, slot2], duration s, stop condition)
SEGMENTS = [
    ("WALK forward 0.3 m/s", "walk", (0.3, 0.0, 0.0), 4.0, None),
    ("RUN forward 0.7 m/s", "run", (0.7, 0.0, 0.0), 4.0, None),
    ("STOP (walk policy, zero command)", "walk", (0.0, 0.0, 0.0), 0.8, None),
    ("STAND STILL (flag=0)", "sitstand", (0.0, 0.0, 0.0), 1.2, None),
    ("SIT (flag=1)", "sitstand", (1.0, 0.0, 0.0), 4.0, None),
    ("STAND UP (flag=0)", "sitstand", (0.0, 0.0, 0.0), 3.0, None),
    ("TURN 180 deg (yaw 1.0 rad/s)", "walk", (0.0, 0.0, 1.0), 5.0, "yaw_pi"),
    ("RUN forward 0.7 m/s", "run", (0.7, 0.0, 0.0), 4.0, None),
    ("CIRCLE: walk 0.3 m/s, yaw 0.6 rad/s (r=0.5 m)", "walk", (0.3, 0.0, 0.6), 11.0, None),
    ("STOP (walk policy, zero command)", "walk", (0.0, 0.0, 0.0), 0.8, None),
    ("STAND STILL (flag=0)", "sitstand", (0.0, 0.0, 0.0), 1.2, None),
]


def yaw_of(q):  # xyzw
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    parser.add_argument("--out", default="rollouts/demo")
    parser.add_argument("--orbit_rate", type=float, default=0.25, help="camera orbit [rad/s] around the duck")
    parser.add_argument("--cam_dist", type=float, default=0.75)
    parser.add_argument("--cam_height", type=float, default=0.32)
    parser.add_argument("--seed", type=int, default=0)
    args, hydra = parser.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    total_steps = int(sum(s[3] for s in SEGMENTS) / 0.02) + 5
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, overrides=hydra)
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.episode_length_s = 1000.0  # one continuous take, no resets
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    # The sit/stand env spawns half its robots seated; the demo starts standing.
    cfg.events.set_ground_state.params.update({"sitting_prob": 0.0, "standing_prob": 1.0, "standing_z_min": 0.115, "standing_z_max": 0.115})
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    # Outside the CLI launcher nothing adds the headless renderer a recorder needs: declare it explicitly.
    # 1280x720 at 25 fps: the recorder holds every frame in RAM until the clip closes (1920x1080 x 1700 frames
    # was 10 GB and got the process killed).
    viz = play_visualizer_cfg().replace(backend=NewtonGLBackendCfg(), headless=True)
    viz.window.size = (1280, 720)
    cfg.sim.visualizer_cfgs = [viz]
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, video_length=total_steps, video_interval=0,
                                            frame_stride=2, fps=25, output_filename_prefix="demo_raw")]
    env = gym.make(args.task, cfg=cfg).unwrapped
    dev = env.device
    policies = {k: torch.jit.load(p, map_location=dev).eval() for k, p in (("walk", WALK), ("run", RUN), ("sitstand", SITSTAND))}
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("base_velocity")
    obs, _ = env.reset()
    dt = env.step_dt
    t = 0.0
    log = []
    theta = math.pi * 0.75  # camera start: front-left of the duck
    with torch.no_grad():
        for label, pol, cmd, dur, stop in SEGMENTS:
            t0 = t
            yaw0 = float(yaw_of(robot.data.root_link_quat_w.torch[0]))
            turned = 0.0
            prev_yaw = yaw0
            p0 = robot.data.root_link_pos_w.torch[0, :2].clone()
            while t - t0 < dur:
                cmd_term._command[:, 0], cmd_term._command[:, 1], cmd_term._command[:, 2] = cmd
                action = policies[pol](obs["policy"])
                obs, *_ = env.step(action)
                t += dt
                # camera: follow the duck, orbit slowly, look slightly ahead of it
                p = robot.data.root_link_pos_w.torch[0]
                theta += args.orbit_rate * dt
                eye = (float(p[0]) + args.cam_dist * math.cos(theta), float(p[1]) + args.cam_dist * math.sin(theta), args.cam_height)
                target = (float(p[0]), float(p[1]), 0.07)
                for viz in env.sim.visualizers:
                    viz.set_camera_view(eye, target)
                yaw = float(yaw_of(robot.data.root_link_quat_w.torch[0]))
                d = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                turned += d
                prev_yaw = yaw
                if stop == "yaw_pi" and abs(turned) >= math.pi:
                    break
            p1 = robot.data.root_link_pos_w.torch[0, :2]
            z = float(robot.data.root_link_pos_w.torch[0, 2])
            g = robot.data.projected_gravity_b.torch[0]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
            dist = float(torch.linalg.norm(p1 - p0))
            log.append({"label": label, "policy": pol, "command": cmd, "t_start": round(t0, 2), "t_end": round(t, 2),
                        "distance_m": round(dist, 3), "mean_speed_m_s": round(dist / max(t - t0, 1e-6), 3),
                        "turned_deg": round(math.degrees(turned), 1), "final_z_m": round(z, 3), "final_tilt_deg": round(tilt, 1)})
            print(log[-1])
    env.close()
    raw = os.path.join(args.out, "demo_raw_0000.mp4")
    labelled = os.path.join(args.out, "duck_demo_sequence.mp4")
    draw = []
    for s in log:
        text = s["label"].replace(":", "\\:").replace("'", "")
        draw.append(f"drawtext=text='{text}  [{s['policy']} policy]':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8:enable='between(t,{s['t_start']},{s['t_end']})'")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", ",".join(draw), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", labelled], check=True)
    fell = any(seg["final_tilt_deg"] > 45.0 for seg in log)
    with open(os.path.join(args.out, "duck_demo_sequence.json"), "w") as f:
        json.dump({"seed": args.seed, "fell": fell, "segments": log}, f, indent=1)
    print("FELL" if fell else "CLEAN")
    print("wrote", labelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
