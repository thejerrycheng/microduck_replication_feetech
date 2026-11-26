#!/usr/bin/env python3
"""MuJoCo preview sheet of the flying-duck model (six views, one with the collision geometry) -> cad/flight/preview_mujoco.png.
A viewer preview for checking the model, not a presentation render."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "app", "assets", "microduck", "scene_flight.xml")); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")); mujoco.mj_forward(m, d)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 960
r = mujoco.Renderer(m, 960, 1280)
views = (("three-quarter front", 140, -18, 0.85, (-0.01, 0, 0.16), 0), ("rear three-quarter", 35, -14, 0.85, (-0.01, 0, 0.16), 0), ("top", 180, -89, 0.9, (-0.01, 0, 0.16), 0),
         ("nacelle close-up", 150, -28, 0.36, (0.0, 0.13, 0.225), 0), ("collision geometry (red)", 140, -18, 0.85, (-0.01, 0, 0.16), 1), ("side", 90, -5, 0.85, (-0.01, 0, 0.16), 0))
fig, axs = plt.subplots(2, 3, figsize=(16.5, 8.4))
for ax, (title, az, el, dist, look, col) in zip(axs.flat, views):
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, col, 0, 0]
    cam = mujoco.MjvCamera(); cam.lookat[:] = look; cam.azimuth, cam.elevation, cam.distance = az, el, dist
    r.update_scene(d, cam, opt); ax.imshow(r.render().copy()); ax.set_title(title); ax.axis("off")
fig.tight_layout(); out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cad", "flight", "preview_mujoco.png"); fig.savefig(out, dpi=95); print("wrote", out)
