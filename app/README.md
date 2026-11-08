# Mac live app: what the engine needs (contract)

The app itself is built on the Mac. This folder holds the pieces it needs from the training side.

## Assets
`assets/microduck/` is Pollen Robotics' MJCF (Apache-2.0, licence included): `scene_backlash.xml` includes
`robot_groundcontact_backlash.xml` (the model our sit/stand, roll and jump policies train on; the walk/run
exports named `*_allcoll_*` are fine-tuned on Isaac Lab's equivalent). `scene.xml` is the non-backlash
variant. Keyframe `INIT` stands the robot at z 0.12. Pollen's reference CPU runner is
`microduck_rl/scripts/infer_policy.py` (BAM servo model wired through `bam.mujoco.MujocoController`).

## Policies (`../exports/*.onnx`, 61 -> 14, observation normaliser baked in, 50 Hz)
| file | role | command slot use |
|---|---|---|
| `walk_allcoll_hd1910_iter16498.onnx` | walking, full-collision model | twist = (vx [-0.4,0.6], vy [-0.3,0.3], yaw [-1,1]) |
| `run_allcoll_hd1910_iter4498.onnx` | running (straight only) | twist = (vx [0.2,1.2], 0, 0) |
| `sitstand_hd1910_iter4000.onnx` | sit <-> stand | twist = (flag, 0, 0): 0 stand, 1 sit |
| `walk_sym_hd1910_iter15000.onnx`, `run_v2_hd1910_iter2999.onnx` | the reduced-collision originals | same |
| `sit_hd1910_iter12000.onnx` | sit-only (standing start) | twist = (1, 0, 0) |

## Observation (61 floats, this order, raw units; the ONNX normalises)
1. `imu` (6): base angular velocity in the trunk frame [rad/s] (3), then projected gravity in the trunk frame (3)
   = R(q)^T (0, 0, -1).
2. `joint_pos` (14): current servo angle minus DEFAULT_POSE, in the order below.
3. `joint_vel` (14): servo velocities [rad/s], same order.
4. `actions` (14): the previous policy output (zeros at start).
5. `velocity_commands` (3), `head_pose_commands` (4, zeros), `body_pose_commands` (6, zeros).

Joint order (actuators and observations): left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee,
right_ankle. DEFAULT_POSE = [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579,
0.0049, -0.4530] rad.

## Action
servo target = DEFAULT_POSE + action (scale 1.0, no clipping in training beyond the joint limits). Control at
50 Hz, physics 0.005 s (decimation 4). Keep ONE shared last-action buffer across policies when hot-swapping.

## Servo (Feetech HD1910, BAM m6 fit used in training)
kt 0.6238 N.m/A, R 4.910 ohm, kp_fw 5, error gain 0.166 x 1.3217 duty/rad per unit, max PWM 0.97, no current
limit, supply 7.4-8.0 V (sag 0-0.2 V/A, floor 7.0 V), command delay 3-6 physics steps, backlash +-1 deg
hinges. Full dict: `microduck_rl/assets/microduck_hd1910.py::HD1910_BAM_FIT`.

## Operator logic that worked in simulation (`scripts/demo_sequence.py`)
Start standing (sit/stand env seats half its spawns by default). Idle = sit/stand policy with flag 0. Arrow
keys drive the walk policy; running = run policy with vx in [0.2, 1.2] (it cannot turn). Stop = walk policy at
zero command for ~0.8 s, then the stand flag. Sit/stand toggles the flag; the transition takes ~2-3 s.
Space (jump) and R (front roll) are reserved for the policies training next.
