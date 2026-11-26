#!/usr/bin/env python
"""Replay a recorded flying-duck rollout kinematically in Isaac Sim and render it with RTX, with fan exhaust smoke.

    ./isaaclab.sh -p scripts/tools/isaacsim_replay_flight.py --traj logs/flight/rollout_baseline.npz --out videos/flight_baseline

Input: the .npz written by scripts/tools/flight_rollout.py (per-frame world pose of every body, fan speeds, thrust).
The robot is the official MicroDuck USD with physics off; every body prim is posed from the recording, the printed
frame's STL parts ride on the trunk, and each fan carries an NVIDIA Flow smoke emitter whose density and exit speed
follow the recorded thrust (exit speed = sqrt(T / (rho A)), about 55 m/s at hover, scaled down for the look).
Frames go to <out>/rgb_####.png; ffmpeg assembles the mp4.

NOT YET RUN (2026-11-25): written while the GPU desktop was unreachable. The body-prim lookup is by name and prints
what it could not match; the Flow attribute names follow omni.flowusd's schema and may need adjusting per version.
"""
import argparse
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", default="videos/flight_replay")
ap.add_argument("--usd", default=None, help="robot USD (default: the Isaac Lab Nucleus MicroDuck)")
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--no_smoke", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": args.width, "height": args.height})

import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pack = json.load(open(os.path.join(ROOT, "microduck_rl", "flight", "flight_pack.json")))
traj = np.load(args.traj, allow_pickle=True)
names = [str(n) for n in traj["body_names"]]
xpos, xquat = traj["xpos"], traj["xquat"]          # (T, nbody, 3), (T, nbody, 4) wxyz
thrust = traj["thrust"]

if args.usd is None:
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    args.usd = f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/microduck_allcollisions.usd"

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
robot = stage.DefinePrim("/World/Duck", "Xform")
robot.GetReferences().AddReference(args.usd)
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateDisplayColorAttr([(0.55, 0.56, 0.58)])
dome = UsdLux.DomeLight.Define(stage, "/World/Dome"); dome.CreateIntensityAttr(900)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun"); sun.CreateIntensityAttr(2500); sun.AddRotateXYZOp().Set((50, 0, 30))
for _ in range(30):
    app.update()

# physics off: the recording drives every body
for prim in Usd.PrimRange(robot):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
body_prim = {}
for prim in Usd.PrimRange(robot):
    if prim.GetName() in names and prim.IsA(UsdGeom.Xformable):
        body_prim.setdefault(prim.GetName(), prim)
missing = [n for n in names if n not in body_prim and n not in ("world", "flight_pack") and "ball" not in n]
print("[replay] bodies matched:", len(body_prim), "| not found in the USD:", missing)


def set_pose(prim, p, q_wxyz):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, p)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q_wxyz[0]), Gf.Vec3d(*map(float, q_wxyz[1:]))))


# the printed frame, pack and fans ride on the trunk (design frame = trunk frame + (0, 0, 0.12))
trunk_i = names.index("trunk_base")
rig = UsdGeom.Xform.Define(stage, "/World/FlightRig")
color = {"frame_left": (0.82, 0.38, 0.24), "frame_right": (0.82, 0.38, 0.24), "pack_tray": (0.2, 0.2, 0.22)}
for part in ("frame_left", "frame_right", "lift_hook_left", "lift_hook_right", "pack_tray"):
    m = trimesh.load(os.path.join(ROOT, "cad", "flight", "out", part + ".stl"))
    mesh = UsdGeom.Mesh.Define(stage, f"/World/FlightRig/{part}")
    mesh.CreatePointsAttr([Gf.Vec3f(*(v * 1e-3 - np.array([0, 0, 0.12]))) for v in m.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(m.faces)); mesh.CreateFaceVertexIndicesAttr(m.faces.flatten().tolist())
    mesh.CreateDisplayColorAttr([color.get(part, (0.16, 0.42, 0.59))])
for f in pack["fans"]:
    c = UsdGeom.Cylinder.Define(stage, f"/World/FlightRig/fan_{f['name']}")
    c.CreateRadiusAttr(pack["duct_od"] / 2); c.CreateHeightAttr(pack["duct_len"]); c.CreateDisplayColorAttr([(0.06, 0.06, 0.06)])
    z = np.array(f["axis"]); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = Gf.Matrix4d(); M.SetRotateOnly(Gf.Matrix3d(*x, *y, *z)); M.SetTranslateOnly(Gf.Vec3d(*f["pos"]))
    UsdGeom.Xformable(c).AddTransformOp().Set(M)

# smoke: one Flow emitter per fan, just under the duct exit, pointing down the thrust axis
emitters = []
if not args.no_smoke:
    ext = omni.kit.app.get_app().get_extension_manager()
    ext.set_extension_enabled_immediate("omni.flowusd", True)
    import carb
    carb.settings.get_settings().set("/rtx/flow/enabled", True)
    carb.settings.get_settings().set("/rtx/flow/rayTracedReflectionsEnabled", True)
    sim = stage.DefinePrim("/World/Flow/simulate", "FlowSimulate")
    sim.CreateAttribute("densityCellSize", Sdf.ValueTypeNames.Float).Set(0.006)
    stage.DefinePrim("/World/Flow/offscreen", "FlowOffscreen")
    stage.DefinePrim("/World/Flow/render", "FlowRender")
    for f in pack["fans"]:
        e = stage.DefinePrim(f"/World/FlightRig/smoke_{f['name']}", "FlowEmitterSphere")
        e.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(0.022)
        e.CreateAttribute("fuel", Sdf.ValueTypeNames.Float).Set(0.0)
        e.CreateAttribute("temperature", Sdf.ValueTypeNames.Float).Set(0.3)
        e.CreateAttribute("coupleRateSmoke", Sdf.ValueTypeNames.Float).Set(4.0)
        exit_pos = np.array(f["pos"]) - np.array(f["axis"]) * (pack["duct_len"] / 2 + 0.01)
        UsdGeom.Xformable(e).AddTranslateOp().Set(Gf.Vec3d(*exit_pos))
        emitters.append((e, np.array(f["axis"])))

cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateFocalLengthAttr(28.0)
rp = rep.create.render_product("/World/Cam", (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
os.makedirs(args.out, exist_ok=True)
writer.initialize(output_dir=os.path.abspath(args.out), rgb=True)
writer.attach([rp])

RHO, AREA = 1.225, pack["fan"]["area"]
stride = max(int(round(1.0 / (args.fps * float(traj["t"][1] - traj["t"][0])))), 1)
for k in range(0, len(xpos), stride):
    for n, prim in body_prim.items():
        i = names.index(n)
        set_pose(prim, xpos[k, i], xquat[k, i])
    set_pose(rig.GetPrim(), xpos[k, trunk_i], xquat[k, trunk_i])
    for (e, axis), T in zip(emitters, thrust[k]):
        v_e = float(np.sqrt(max(T, 0.0) / (RHO * AREA)))
        e.GetAttribute("smoke").Set(min(v_e / 60.0, 1.0) * 1.5) if e.HasAttribute("smoke") else e.CreateAttribute("smoke", Sdf.ValueTypeNames.Float).Set(min(v_e / 60.0, 1.0) * 1.5)
        vel = -axis * v_e * 0.08          # visual scale: the real jet would cross the frame in a few milliseconds
        (e.GetAttribute("velocity") if e.HasAttribute("velocity") else e.CreateAttribute("velocity", Sdf.ValueTypeNames.Float3)).Set(Gf.Vec3f(*map(float, vel)))
    p = xpos[k, trunk_i]
    eye = Gf.Vec3d(float(p[0]) + 0.9, float(p[1]) - 1.2, 0.55)
    M = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(float(p[0]), float(p[1]), max(float(p[2]), 0.15)), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(M)
    rep.orchestrator.step(rt_subframes=8)
rep.orchestrator.wait_until_complete()
subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob", "-i", os.path.join(args.out, "rgb_*.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", args.out.rstrip("/") + ".mp4"], check=False)
app.close()
