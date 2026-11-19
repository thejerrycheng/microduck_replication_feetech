"""Measure the EFFECTIVE sole-floor friction the policies train against.

Newton's MuJoCo solver combines two geoms' friction as the MAX unless their ``mjc:priority`` differs, and the
flat terrain is authored at 1.0. So a sole randomised to 0.5 may still touch the floor at 1.0. This script
(1) prints the solver's ``geom_priority`` / ``geom_friction`` for the soles and the ground;
(2) measures the friction physically: the standing robot (zero action = default pose) is pulled forward by a
    slowly growing horizontal force at both ankles; the force at which it starts to slide over the feet's
    measured normal load is the static friction coefficient, ``mu_s = F_onset / N``;
(3) optionally rolls a policy at a fixed command at every friction (``--policy``), reporting falls and the
    planar speed of feet in contact (the slip distribution a slip penalty threshold must be set against).
Sole mu is set per environment to a sweep of values through the same binding the ``foot_friction`` event uses.

Run::

    uv run --project external/IsaacLab python scripts/measure_foot_friction.py --task <task> presets=backlash \
        [--policy checkpoints/velstand_21495/policy.pt --vx 0.3] --out docs/friction/<name>.json
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Microduck-VelStand-Flat-HD1910-AllColl")
    p.add_argument("--num_envs", type=int, default=96)
    p.add_argument("--mus", default="0.2,0.4,0.6,0.8,1.0,1.2")
    p.add_argument("--settle_s", type=float, default=0.6)
    p.add_argument("--ramp_s", type=float, default=2.0, help="the pull grows linearly to --ramp_max_g x the feet's normal load over this time")
    p.add_argument("--ramp_max_g", type=float, default=1.6)
    p.add_argument("--onset_m", type=float, default=0.005, help="displacement along the pull that counts as sliding")
    p.add_argument("--policy", default=None, help="TorchScript policy to roll at each friction")
    p.add_argument("--vx", type=float, default=0.3)
    p.add_argument("--roll_s", type=float, default=8.0)
    p.add_argument("--out", default=None)
    args, extra = p.parse_known_args()
    import gymnasium as gym
    from isaaclab.utils.math import quat_apply_yaw
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from newton import ModelFlags

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=extra)
    env_cfg.play_mode()
    env_cfg.scene.num_envs = args.num_envs  # play_mode shrinks the scene; the sweep needs every row
    env_cfg.curriculum = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    foot_ids, _ = sensor.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
    ankle_ids, _ = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
    mus = [float(m) for m in args.mus.split(",")]
    per = args.num_envs // len(mus)
    mu_env = torch.tensor([mus[min(i // per, len(mus) - 1)] for i in range(args.num_envs)], device=env.device)
    mu_np = mu_env.cpu().numpy()

    obs, _ = env.reset()
    term = env.event_manager.get_term_cfg("foot_friction").func
    impl = term._impl
    report = {"task": args.task, "num_envs": args.num_envs, "mus": mus, "overrides": extra}

    def set_mu():
        fv = wp.to_torch(impl._friction_binding)
        fv[:, impl._shape_indices.to(env.device)] = mu_env[:, None]
        impl._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    set_mu()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(int(args.settle_s / env.step_dt)):
        obs, *_ = env.step(zero)

    # --- (1) solver arrays
    solver = getattr(impl._newton_manager.backend, "solver", None)
    if solver is None or not hasattr(solver, "mjw_model"):
        from newton.solvers import SolverMuJoCo

        solver = next((o for o in gc.get_objects() if isinstance(o, SolverMuJoCo)), None)
    if solver is not None:
        model = impl._newton_manager.backend.model
        names = None
        for attr in ("shape_key", "shape_label", "shape_name"):
            if hasattr(model, attr):
                names = list(getattr(model, attr))
                break
        shape_body = model.shape_body.numpy()
        g2s = solver.mjc_geom_to_newton_shape.numpy()
        g2s = g2s[0] if g2s.ndim == 2 else g2s
        prio = solver.mjw_model.geom_priority.numpy()
        prio = prio[0] if prio.ndim == 2 else prio
        fric = solver.mjw_model.geom_friction.numpy()  # (world, geom, 3)
        rows = []
        for g in range(len(g2s)):
            s = int(g2s[g])
            if s < 0:
                continue
            key = names[s] if names and s < len(names) else ""
            is_sole = "sole_" in key
            is_ground = int(shape_body[s]) < 0
            if is_sole or is_ground:
                rows.append({"geom": g, "shape": s, "key": key, "sole": is_sole, "ground": is_ground, "priority": int(prio[g]),
                             "geom_friction_w0_w1_wlast": [float(fric[0, g, 0]), float(fric[min(1, fric.shape[0] - 1), g, 0]), float(fric[-1, g, 0])]})
        report["solver_geoms"] = rows
        for r in rows:
            print("GEOM", r)
        # where the priority should have come from: the USD attribute on the sole prim and Newton's custom attribute
        try:
            from isaaclab.sim.utils import get_current_stage

            stage = get_current_stage()
            for r in rows:
                if r["sole"]:
                    prim = stage.GetPrimAtPath(r["key"])
                    attr = prim.GetAttribute("mjc:priority")
                    print("USD", r["key"].split("/Robot/")[-1], "valid", prim.IsValid(), "mjc:priority", attr.Get() if attr and attr.IsValid() else None, "apis", [a for a in prim.GetAppliedSchemas()][:6])
        except Exception as e:  # noqa: BLE001
            print("USD probe failed:", repr(e))
        mj = getattr(model, "mujoco", None)
        gp = getattr(mj, "geom_priority", None) if mj is not None else None
        print("MODEL mujoco.geom_priority:", None if gp is None else [int(x) for x in gp.numpy().reshape(-1)[[r["shape"] for r in rows]]], "attrs:", [a for a in dir(mj) if not a.startswith("_")][:12] if mj is not None else None)
        d = solver.mjw_data
        ncon = int((getattr(d, "nacon", None) if hasattr(d, "nacon") else d.ncon).numpy().reshape(-1)[0])
        cg = d.contact.geom.numpy()[:ncon]
        cw = d.contact.worldid.numpy()[:ncon]
        cf = d.contact.friction.numpy()[:ncon, 0]
        sole_geoms = {r["geom"] for r in rows if r["sole"]}
        sel = np.array([(int(a) in sole_geoms) or (int(b) in sole_geoms) for a, b in cg], dtype=bool)
        resolved = []
        for m in mus:
            ws = np.nonzero(mu_np == m)[0]
            pick = sel & np.isin(cw, ws)
            resolved.append({"mu_set": m, "n_contacts": int(pick.sum()), "contact_friction_median": float(np.median(cf[pick])) if pick.any() else None,
                             "contact_friction_min": float(cf[pick].min()) if pick.any() else None, "contact_friction_max": float(cf[pick].max()) if pick.any() else None})
            print(f"CONTACT mu_set {m:.2f}: {resolved[-1]}")
        report["resolved_contacts"] = resolved

    # --- (2) force ramp at the ankles
    heading = quat_apply_yaw(robot.data.root_link_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    start_xy = robot.data.body_link_pos_w.torch[:, ankle_ids, :2].mean(1).clone()  # the feet, not the root
    n_load = sensor.data.net_forces_w.torch[:, foot_ids, 2].sum(-1).clone()  # N on both feet, standing
    dt = env.physics_dt
    steps = int(args.ramp_s / dt)
    onset_force = torch.full((env.num_envs,), float("nan"), device=env.device)
    upright_at_onset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    max_ratio = torch.zeros(env.num_envs, device=env.device)
    wc = robot.permanent_wrench_composer
    ids = torch.arange(env.num_envs, device=env.device)
    for k in range(steps):
        f_total = args.ramp_max_g * n_load * (k + 1) / steps  # (N,)
        forces = torch.zeros(env.num_envs, len(ankle_ids), 3, device=env.device)
        forces[:, :, :] = (heading * (f_total / len(ankle_ids))[:, None])[:, None, :]
        pos = robot.data.body_link_pos_w.torch[:, ankle_ids, :].clone()
        pos[:, :, 2] = 0.0  # at the floor: no moment about the sole edge
        wc.set_forces_and_torques_index(forces=forces, torques=torch.zeros_like(forces), positions=pos, body_ids=ankle_ids, env_ids=ids, is_global=True)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
        disp = ((robot.data.body_link_pos_w.torch[:, ankle_ids, :2].mean(1) - start_xy) * heading[:, :2]).sum(-1)
        nf = sensor.data.net_forces_w.torch[:, foot_ids]
        ratio = nf[:, :, :2].norm(dim=-1).sum(-1) / nf[:, :, 2].sum(-1).clamp_min(1e-3)  # tangential / normal on the feet
        max_ratio = torch.where(torch.isnan(onset_force), torch.maximum(max_ratio, ratio), max_ratio)
        new = torch.isnan(onset_force) & (disp > args.onset_m)
        onset_force[new] = f_total[new]
        upright_at_onset[new] = robot.data.projected_gravity_b.torch[new, 2] < -0.7
    wc.set_forces_and_torques_index(forces=torch.zeros(env.num_envs, len(ankle_ids), 3, device=env.device), torques=torch.zeros(env.num_envs, len(ankle_ids), 3, device=env.device), positions=None, body_ids=ankle_ids, env_ids=ids, is_global=True)
    mu_s = (onset_force / n_load.clamp_min(1e-6)).cpu().numpy()
    upright = upright_at_onset.cpu().numpy()
    mr = max_ratio.cpu().numpy()
    report["normal_load_N_median"] = float(n_load.median())
    ramp = []
    for m in mus:
        sel = mu_np == m
        v = mu_s[sel]
        ramp.append({"mu_set": m, "n": int(sel.sum()), "slid_frac": float(np.isfinite(v).mean()), "mu_s_median": float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                     "mu_s_p10": float(np.nanpercentile(v, 10)) if np.isfinite(v).any() else None, "mu_s_p90": float(np.nanpercentile(v, 90)) if np.isfinite(v).any() else None,
                     "upright_at_onset_frac": float(upright[sel & np.isfinite(mu_s)].mean()) if np.isfinite(v).any() else None,
                     "max_tangential_ratio_median": float(np.median(mr[sel]))})
        print(f"RAMP mu_set {m:.2f}: slid {ramp[-1]['slid_frac']:.2f}, mu_s median {ramp[-1]['mu_s_median']}, p10 {ramp[-1]['mu_s_p10']}, p90 {ramp[-1]['mu_s_p90']}, upright {ramp[-1]['upright_at_onset_frac']}, max Ft/N {ramp[-1]['max_tangential_ratio_median']:.3f}")
    report["ramp"] = ramp

    # --- (3) policy rollout per friction
    if args.policy:
        policy = torch.jit.load(args.policy, map_location=env.device).eval()
        cmd = env.command_manager.get_term("base_velocity")
        obs, _ = env.reset()
        set_mu()
        n = int(args.roll_s / env.step_dt)
        fallen_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        slip_samples = {m: [] for m in mus}
        vx_b = []
        with torch.no_grad():
            for t in range(n):
                cmd.vel_command_b[:, 0] = args.vx
                cmd.vel_command_b[:, 1:] = 0.0
                obs, *_ = env.step(policy(obs["policy"]))
                fallen_any |= robot.data.projected_gravity_b.torch[:, 2] > -0.5
                in_c = sensor.data.current_contact_time.torch[:, foot_ids] > 0
                v = robot.data.body_link_lin_vel_w.torch[:, ankle_ids, :2].norm(dim=-1)
                if t > n // 5:
                    vx_b.append(robot.data.root_link_lin_vel_b.torch[:, 0].clone())
                    for m in mus:
                        sel = torch.from_numpy(mu_np == m).to(env.device)
                        vv = v[sel][in_c[sel]]
                        if vv.numel():
                            slip_samples[m].append(vv)
        vx_all = torch.stack(vx_b).mean(0).cpu().numpy()
        fa = fallen_any.cpu().numpy()
        roll = []
        for m in mus:
            sel = mu_np == m
            s = torch.cat(slip_samples[m]).cpu().numpy() if slip_samples[m] else np.zeros(1)
            roll.append({"mu_set": m, "fallen_frac": float(fa[sel].mean()), "vx_achieved_mean": float(vx_all[sel].mean()),
                         "contact_foot_speed_p50": float(np.percentile(s, 50)), "p90": float(np.percentile(s, 90)), "p99": float(np.percentile(s, 99)),
                         "frac_over_0.05": float((s > 0.05).mean()), "frac_over_0.10": float((s > 0.10).mean()), "frac_over_0.20": float((s > 0.20).mean())})
            print(f"ROLL mu_set {m:.2f}: fallen {roll[-1]['fallen_frac']:.2f}, vx {roll[-1]['vx_achieved_mean']:.3f}, contact foot speed p50/p90/p99 {roll[-1]['contact_foot_speed_p50']:.3f}/{roll[-1]['p90']:.3f}/{roll[-1]['p99']:.3f}, >0.1 {roll[-1]['frac_over_0.10']:.3f}")
        report["rollout"] = {"policy": args.policy, "vx": args.vx, "roll_s": args.roll_s, "per_mu": roll}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
