#!/usr/bin/env python
"""Check the flying-duck plant inside Isaac Lab / Newton: the pack is on the trunk, and the fans lift the duck.

    uv run --project external/IsaacLab python scripts/check_flight_plant.py presets=backlash [--out docs/flight_isaac_check.json]

Nominal model (no randomisation, no pushes). Reports the trunk mass with and without the flight hardware and the
total robot mass; then, with zero joint actions and no attitude control, three constant throttles (0 %, the model's
hover throttle, 100 %): thrust per fan, bus voltage and current, and the climb rate reached in the first 0.7 s.
Pass: pack mass on the trunk, no thrust at 0 %, thrust within 10 % of weight at the hover throttle, more than
1.5 x weight at 100 %, and the force composed on the trunk equal to four fans at the cant angle.
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--out", default=None)
    args, _ = parser.parse_known_args()

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from microduck_rl.mdp import flight

    task = "Microduck-Takeoff-Flat-HD1910-AllColl"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
    for name in ("push_robot", "randomize_com", "randomize_head_com", "mass_inertia", "battery_mass", "randomize_joint_friction",
                 "randomize_armature", "encoder_bias"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.events.flight_pack.params.update(mass_scale_range=(1.0, 1.0), com_offset_m=0.0)
    f = env_cfg.actions.fans
    f.thrust_scale_range = f.per_fan_thrust_range = f.torque_coeff_range = f.rotor_inertia_range = f.cell_resistance_range = (1.0, 1.0)
    f.soc_range = (0.8, 0.8)
    f.offset, f.scale = 0.0, 1.0                       # action = throttle
    env_cfg.curriculum = None
    env_cfg.terminations.fell_over = None
    env_cfg.terminations.ceiling = None
    env = gym.make(task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    trunk = int(robot.find_bodies("trunk_base")[0][0])
    mass = robot.data.body_mass.torch
    fans = env.action_manager.get_term("fans")
    pack = flight.FLIGHT_PACK["mass"]
    rep = dict(trunk_mass_kg=float(mass[0, trunk]), robot_mass_kg=float(mass[0].sum()), pack_mass_kg=pack,
               trunk_com_b=[float(v) for v in robot.data.body_com_pos_b.torch[0, trunk, :3]], action_dim=int(env.action_manager.total_action_dim))
    print(f"[check] trunk {rep['trunk_mass_kg']*1e3:.0f} g (pack {pack*1e3:.0f} g of it), robot {rep['robot_mass_kg']*1e3:.0f} g, actions {rep['action_dim']}")
    hover_w = (rep["robot_mass_kg"] * 9.81 / 4 / fans._base.kf) ** 0.5
    from microduck_rl.flight import fan_model as F
    u_hover = F.throttle_for_thrust(rep["robot_mass_kg"] * 9.81 / 4, 4, 0.8, fans._base, fans._bat)
    rep["u_hover_model"] = u_hover
    rep["runs"] = {}
    n_act = rep["action_dim"]
    for label, u in (("off", 0.0), ("hover", float(u_hover)), ("full", 1.0)):
        env.reset()
        a = torch.zeros(args.num_envs, n_act, device=env.device)
        a[:, -4:] = u
        z0 = float((robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).mean())
        zs, vz = [], []
        for k_ in range(int(1.5 / env.step_dt)):
            env.step(a)
            zs.append(float((robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).mean()))
            if k_ * env.step_dt < 0.7:      # before the uncontrolled duck has tipped far
                vz.append(float(robot.data.root_link_lin_vel_w.torch[:, 2].mean()))
        rep["runs"][label] = dict(throttle=u, z_start=z0, z_end=zs[-1], z_max=max(zs), thrust_per_fan_n=float(fans.thrust.mean()), v_bus=float(fans.v_bus.mean()),
                                  i_bus=float(fans.i_bus.mean()), rotor_rpm=float(fans.w.mean() * 9.549), vz_max_first_0p7s=max(vz),
                                  composed_fz_n=float(robot.permanent_wrench_composer.composed_force.torch[:, trunk, 2].mean()))
        r = rep["runs"][label]
        print(f"[check] throttle {u:.2f} ({label}): height {z0:.3f} -> {zs[-1]:.3f} m (max {max(zs):.3f}), thrust {r['thrust_per_fan_n']:.2f} N per fan "
              f"({4*r['thrust_per_fan_n']/(rep['robot_mass_kg']*9.81):.2f} x weight), climb rate reached in 0.7 s {r['vz_max_first_0p7s']:+.2f} m/s, bus {r['v_bus']:.1f} V {r['i_bus']:.0f} A, {r['rotor_rpm']:.0f} rpm")
    tw = lambda k: 4 * rep["runs"][k]["thrust_per_fan_n"] / (rep["robot_mass_kg"] * 9.81)
    # Nothing balances the duck here (zero leg action, no attitude control) and the stand with the pack is not passively
    # stable, so it tips while the rotors spin up (about 0.45 s from rest at the 45 A limit): judge by the forces.
    cant = math.cos(math.radians(6.0))
    f_ok = abs(rep["runs"]["full"]["composed_fz_n"] - 4 * rep["runs"]["full"]["thrust_per_fan_n"] * cant) < 0.03 * 4 * rep["runs"]["full"]["thrust_per_fan_n"]
    ok = rep["runs"]["off"]["thrust_per_fan_n"] < 0.01 and 0.9 < tw("hover") < 1.1 and tw("full") > 1.5 and f_ok and abs(rep["trunk_mass_kg"] - 0.2 - pack) < 0.02
    rep["pass"] = bool(ok)
    print("[check]", "PASS" if ok else "FAIL")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
