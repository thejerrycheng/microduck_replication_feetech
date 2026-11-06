#!/usr/bin/env python
"""Measure the HD1910 Microduck's standing and seated trunk rest heights in the Isaac Lab/Newton model.

The sit/stand reward targets (``STAND_Z``, ``SIT_Z`` in ``sitstand_env_cfg.py``) are upstream's MuJoCo
measurements; this script settles the robot at the HOME pose and at the SIT keyframe under zero action
and prints the trunk height, tilt and applied neck torque so those constants can be checked rather than
trusted. Also reports whether ``applied_torque`` is populated under the Newton BAM drive (the
``joint_torque_rate_l2`` and ``neck_strain_leaned`` terms read it).

Run::

    uv run --project external/IsaacLab python scripts/measure_rest_heights.py [presets=backlash]
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-SitStand-Flat-HD1910")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--settle_s", type=float, default=3.0)
    parser.add_argument("--out", default=None)
    args, overrides = parser.parse_known_args()

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from microduck_rl.tasks.sitstand_env_cfg import SITTING_TARGET_OVERRIDES

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    # No randomisation, no pushes, no hardening events: measure the nominal model.
    for name in ("push_robot", "seated_head_jerk", "seated_tip", "randomize_com", "randomize_head_com",
                 "mass_inertia", "randomize_joint_friction", "randomize_armature", "encoder_bias"):
        setattr(env_cfg.events, name, None)
    env_cfg.events.set_ground_state.params["sitting_joint_noise_std"] = 0.0
    env_cfg.events.set_ground_state.params["sitting_tilt_max"] = 0.0
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    servo_ids = env.action_manager.get_term("joint_pos")._joint_ids
    head_ids, _ = robot.find_joints(["neck_pitch", "head_pitch", "head_yaw", "head_roll"], preserve_order=True)
    steps = int(args.settle_s / env.step_dt)
    report = {}
    for mode, sit_prob, stand_prob in (("stand", 0.0, 1.0), ("sit", 1.0, 0.0)):
        env_cfg.events.set_ground_state.params["sitting_prob"] = sit_prob
        env_cfg.events.set_ground_state.params["standing_prob"] = stand_prob
        env.event_manager.get_term_cfg("set_ground_state").params.update(sitting_prob=sit_prob, standing_prob=stand_prob)
        env.command_manager.get_term("base_velocity").cfg.sit_prob = sit_prob
        env.reset()
        # Hold the pose with the servos: the action is the offset from the default pose.
        target = torch.zeros(env.num_envs, len(servo_ids), device=env.device)
        if mode == "sit":
            default = robot.data.default_joint_pos.torch[:, servo_ids]
            for index, value in SITTING_TARGET_OVERRIDES.items():
                target[:, int(index)] = value - default[:, int(index)]
        for _ in range(steps):
            env.step(target)
        z = (robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).cpu()
        gravity_b = robot.data.projected_gravity_b.torch.cpu()
        tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0)))
        tau = robot.data.applied_torque.torch[:, head_ids].abs().cpu()
        report[mode] = {
            "trunk_z_m": {"median": float(z.median()), "min": float(z.min()), "max": float(z.max())},
            "tilt_deg": {"median": float(tilt.median()), "max": float(tilt.max())},
            "neck_abs_torque_nm": {"median": float(tau.median()), "max": float(tau.max())},
            "applied_torque_populated": bool(robot.data.applied_torque.torch.abs().sum() > 0),
        }
        print(mode, json.dumps(report[mode]))
    env.close()
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
