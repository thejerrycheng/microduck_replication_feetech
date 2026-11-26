#!/usr/bin/env python3
"""Preview clip of the model-based baseline: thrust-assisted run -> take-off -> hover -> land (flight_rollout.rollout).

    ~/.venvs/microduck_live/bin/python scripts/tools/flight_baseline_render.py

A MuJoCo preview for checking behaviour, not a presentation render (those are replayed in Isaac Sim).
"""
import os, subprocess, sys
import mujoco
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import flight_rollout as FR

out, L, (m, duck) = FR.rollout(record=True)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k in ("success", "run_speed", "max_tilt_flight", "z_err_hover", "z_peak")})
d = mujoco.MjData(m)
W, H = 1280, 720
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
ren = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 120, -12, 1.6
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
dst = os.path.join(ROOT, "videos", "flight_baseline_mujoco_preview.mp4")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "50", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst], stdin=subprocess.PIPE)
look = np.array([0.0, 0.0, 0.25])
for q in L["qpos"]:
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    p = d.xpos[trunk]; look = 0.9 * look + 0.1 * np.array([p[0], p[1], max(p[2], 0.2)]); cam.lookat[:] = look
    ren.update_scene(d, cam, opt); ff.stdin.write(ren.render().tobytes())
ff.stdin.close(); ff.wait(); print("wrote", dst)
