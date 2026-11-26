#!/usr/bin/env python3
"""Preview clip of the MuJoCo-trained fan policy (checkpoints/flight_hover_mujoco/policy.npz): a scripted velocity-command sequence
(take off, hover, forward, climb, sidestep, yaw, descend).

    python scripts/tools/flight_rl_render.py [--out videos/flight_rl_mujoco_preview.mp4] [--seconds 8]

A MuJoCo viewer preview for checking the policy, not a presentation render (those are replayed in Isaac Sim).
"""
import argparse, os, subprocess, sys
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_rl_mujoco as R

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "videos", "flight_rl_mujoco_preview.mp4"))
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--thrust", type=float, default=1.0); ap.add_argument("--com_mm", type=float, default=0.0)
a = ap.parse_args()
P = dict(np.load(os.path.join(R.CK, "policy.npz")))
env = R.Env(7); SCHED = [(0.0, [0, 0, 0.5, 0]), (1.2, [0, 0, 0, 0]), (3.0, [0.4, 0, 0, 0]), (5.0, [0, 0, 0.4, 0]), (6.0, [0, 0.25, 0, 0]), (8.0, [0, 0, 0, 0.8]), (10.0, [0, 0, -0.3, 0]), (11.0, [0, 0, 0, 0])]
env.fixed = dict(thrust=a.thrust, com_x=a.com_mm * 1e-3, schedule=SCHED)
obs = env.reset()
W, H, FPS = 1280, 720, 50
env.m.vis.global_.offwidth, env.m.vis.global_.offheight = W, H
ren = mujoco.Renderer(env.m, H, W)
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -12, 1.5
os.makedirs(os.path.dirname(a.out), exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
look = np.array([0.0, 0.0, 0.3]); crashed_at = None; zs = []
for k in range(int(a.seconds * FPS)):
    if crashed_at is None:
        obs, r, bad, to, info = env.step(R.policy_np(P, obs)); zs.append(info["z"])
        if bad:
            crashed_at = k / FPS
    else:                                   # let it fall with the fans cut so the clip shows what happened
        env.d.qfrc_applied[:] = 0; env.plant.command(np.zeros(4))
        for _ in range(R.DECIM):
            env.d.qfrc_applied[:] = 0; env.plant.substep(); mujoco.mj_step(env.m, env.d)
    p = env.d.xpos[env.trunk]
    look = 0.9 * look + 0.1 * np.array([p[0], p[1], max(p[2], 0.2)])
    cam.lookat[:] = look
    ren.update_scene(env.d, cam, opt)
    ff.stdin.write(ren.render().tobytes())
ff.stdin.close(); ff.wait()
print(f"wrote {a.out}: " + (f"CRASHED at {crashed_at:.2f} s" if crashed_at is not None else f"flew {a.seconds:.0f} s, final height {zs[-1]:.2f} m") + f", peak height {max(zs):.2f} m")
