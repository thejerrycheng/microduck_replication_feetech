# Pick-up environment (`Microduck-Pick-Flat-HD1910-AllColl`)

Walk to an object lying on the floor, put the beak on it, lift it. Built 2026-11-17 on walk v4 / velstand (same
61-float observation, 14-float action, 50 Hz), so `velstand_21495` warm-starts it and the app contract is unchanged.
Code: `microduck_rl/mdp/pick.py`, `microduck_rl/tasks/pick_env_cfg.py`; chain `scripts/train_pick.sh`;
evaluator `scripts/eval_pick.py`.

## Why not pixels, depth estimation or a point cloud

The replica has one RGB camera in the head (kissqy's HD1910 MJCF carries a `head_camera` site; a ToF site too) and
the objects rest on the floor. A detector gives the object's pixel; the ray through that pixel meets the ground plane
at a known 3D point, because the camera pose follows from the head servo angles and the trunk's IMU tilt. That is a
closed-form range with no learned depth, no point cloud and no rendering gap to close: the policy consumes an
object position, and the simulator produces that position from the true one through the same limitations the real
pipeline has (see "sensing model"). Monocular depth or a point cloud would only be needed for objects off the floor.
If a pixel policy is ever wanted, this state policy is the teacher to distil it from.

## Observation contract (the ten command floats, slots 51-60 of the 61)

| slot | value |
|---|---|
| 51-53 | object estimate (x, y, z) in the base-yaw frame [m] (forward, left, up from the trunk origin) |
| 54 | seen this step (1/0) |
| 55 | held (1/0) |
| 56 | object size: 2 x half-width [m] |
| 57-60 | zero |

Slots 48-50 (velocity command) carry the **object-seek twist**, computed from the same estimate by
`ObjectSeekCommand`: `vx = clip(1.5 (d - 0.13), 0, 0.4) max(cos b, 0)`, `wz = clip(2 b, -1, 1)`, zero once
`d < 0.15 m and |b| < 15 deg` or the object is held (d = planar distance, b = bearing). Run the same law on the robot.

On the robot: hold the last estimate when the detector has nothing (seen = 0), set held = 1 when the beak switch /
current / magnet sensor says so (or from the estimate reaching the beak), size from the detection box width x range.

## Sensing model in simulation (`pick_grasp`)

* visible only inside the head camera frustum: HFOV 62.2 deg, VFOV 48.8 deg, range 1.5 m (Pi camera v2 numbers);
* detection dropout 10 % per frame; position noise N(0, 0.01 + 0.03 d) m; estimate held while unseen;
* observation delayed 1-3 control steps (20-60 ms), lag redrawn on 3 % of frames.

ASSUMED, measure on the robot: camera at (0.012, 0, -0.070) m in the head frame, optical axis 20 deg below the
head's forward axis. The head body frame has +x up and -z forward (`HEAD_FORWARD`, `HEAD_UP` in `pick.py`).

## Grasp model

The 14-servo replica has no jaw servo, so the beak is a passive gripper (magnet, hook or sticky pad on the lower
beak; the exact device is open). In simulation the object latches when the beak tip (`BEAK_TIP_OFFSET`, 78 mm
from the head origin, matching kissqy's `mouth_tip`) is within 15 mm of the object's top and the beak points down
(forward axis z < -0.25). While held the object is attached to the beak and **its mass is added to the head body**,
so the neck and leg servos carry the real load; a reset releases it and restores the head mass.

## Randomisation

* shape per environment: cuboids 2.5-5 cm, spheres r 1.5-2.5 cm, cylinders, a capsule (mesh convex hulls: the
  MJWarp solver only lets mesh geoms differ between environments);
* mass per reset: 5-40 g (`OBJECT_MASS_RANGE_KG`; kissqy's GroundPick trains the mouth payload at 10-40 g);
* placement per reset: 0.25-0.9 m ahead, bearing +-60 deg, random yaw;
* firmware P per servo in [5, 8], supply, friction, mass, CoM, backlash, servo-bus dropouts: the walk's own DR.

## Rewards and termination

Walk v4's tracking and regularisation terms (tracking the seek twist), head-pose terms off, plus
`pick_grasp` 2 (1/step while held), `object_progress` 10 (metres closed per step, clipped +-5 cm, zero while held),
`beak_to_object` 2 (exp(-d/4 cm) of tip to object top inside 0.3 m), `object_lift` 3 (fraction of 6 cm raised while
held). Episode 15 s; `fell_over` terminates (no get-up in v1).

## Evaluation (`scripts/eval_pick.py`, deterministic, 32 robots)

grasp rate, held at end, lifted >= 4 cm at end, falls, time to grasp (median, p90), fraction of frames the object
was seen. Results `docs/pick_<ckpt>.json`.

## Open items

* the passive gripper on the real beak (and its sensor for `held`);
* measure the camera pose and FOV; set `HEAD_CAMERA_OFFSET`, `HEAD_CAMERA_TILT_DEG`;
* fall recovery while holding (VelStand's fallen spawns are off in v1);
* carry-to-target and release are not in the task yet.

## Sources read for the design

kissqy `Microduck-kissqy` (GroundPick task config: phased bow, mouth payload 10-40 g force, mouth proximity /
perpendicularity rewards; `robot_walk.xml` head_camera, mouth_tip, tof sites); Ley-WL `microduck-rl` pick models
(beak to 22 mm of the floor, 32/32, no object); SLIM / "Learning Multi-Stage Pick-and-Place with a Legged Mobile
Manipulator" (quadruped + arm, sim-to-real via a segmentation/depth bottleneck); Isaac Lab `core/lift` (mesh
convex-hull objects per environment on Newton).
