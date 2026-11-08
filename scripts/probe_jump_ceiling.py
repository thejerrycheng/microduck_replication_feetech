#!/usr/bin/env python
"""Open-loop jump ceiling of the HD1910 duck: how high can a scripted crouch-and-extend throw the trunk?

No policy. Each environment gets one (crouch depth, joint ratios, sign set, overshoot gain, head flick)
combination: 1.0 s at HOME to capture the standing baseline, 1.0 s holding the crouch, then a step to the
extension target held for 0.6 s. The BAM servo saturates its PWM on a large position error, so a big
overshoot target is "maximum effort" on this plant. Reports the best ballistic rise (trunk z above the
baseline while BOTH feet are off the ground), air time and takeoff vertical speed, per variant.

Run (play mode: no pushes, nominal DR-free plant)::

    uv run --project external/IsaacLab python scripts/probe_jump_ceiling.py presets=backlash --out docs/jump_ceiling.json
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

DEPTHS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
ANKLE_RATIOS = (0.3, 0.6, 0.9)
HIP_RATIOS = (0.3, 0.6)
SIGN_SETS = ((1, 1, 1), (1, -1, 1), (-1, 1, -1), (-1, -1, -1))  # (hip, knee, ankle) multipliers on the default-direction guess
GAINS = (1.5, 3.0)
HEAD = (0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Microduck-Sit-Flat-HD1910")
    parser.add_argument("--out", default="docs/jump_ceiling.json")
    parser.add_argument("--hold_s", type=float, default=0.6)
    args, hydra = parser.parse_known_args()

    grid = list(itertools.product(DEPTHS, ANKLE_RATIOS, HIP_RATIOS, SIGN_SETS, GAINS, HEAD))
    n = len(grid)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n, overrides=hydra)
    env_cfg.play_mode()  # resets num_envs to its play default, so set the grid size AFTER it
    env_cfg.scene.num_envs = n
    env_cfg.curriculum = None
    env_cfg.episode_length_s = 10.0
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    names = env.action_manager.get_term("joint_pos")._joint_names
    idx = {k: names.index(k) for k in names}
    feet = [i for i, b in enumerate(sensor.body_names) if "ankle" in b]
    assert len(feet) == 2, sensor.body_names
    dflt = robot.data.default_joint_pos.torch[0]
    # direction guess: push each joint further along its own default offset (knee default ~0 -> use hip's sign)
    dirs = {}
    for side in ("left", "right"):
        hp = float(torch.sign(dflt[idx[f"{side}_hip_pitch"]]))
        an = float(torch.sign(dflt[idx[f"{side}_ankle"]]))
        dirs[side] = (hp, hp, an)

    crouch = torch.zeros(n, len(names), device=env.device)
    extend = torch.zeros(n, len(names), device=env.device)
    meta = []
    for e, (d, ra, rh, ss, g, head) in enumerate(grid):
        for side in ("left", "right"):
            hp, kn, an = dirs[side]
            crouch[e, idx[f"{side}_hip_pitch"]] = ss[0] * hp * rh * d
            crouch[e, idx[f"{side}_knee"]] = ss[1] * kn * d
            crouch[e, idx[f"{side}_ankle"]] = ss[2] * an * ra * d
        extend[e] = -g * crouch[e]
        if head:
            extend[e, idx["neck_pitch"]] = -1.0
            extend[e, idx["head_pitch"]] = -1.0
        meta.append({"depth": d, "ankle_ratio": ra, "hip_ratio": rh, "signs": ss, "gain": g, "head": head})

    dt = env.step_dt
    env.reset()
    zero = torch.zeros_like(crouch)

    def z():
        return robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    def tilt_deg():
        gz = robot.data.projected_gravity_b.torch[:, 2]
        return torch.rad2deg(torch.acos(torch.clamp(-gz, -1.0, 1.0)))

    def airborne():
        f = sensor.data.net_forces_w[:, feet, :].norm(dim=-1)
        return (f < 1.0).all(dim=1)

    with torch.no_grad():
        for _ in range(int(1.0 / dt)):
            env.step(zero)
        base = z().clone()
        for _ in range(int(1.0 / dt)):
            env.step(crouch)
        crouch_z = z().clone()
        crouch_tilt = tilt_deg().clone()
        crouch_ok = (crouch_z < base - 0.01) & (crouch_tilt < 25.0) & (~airborne())
        rise = torch.zeros(n, device=env.device)
        air_steps = torch.zeros(n, device=env.device)
        run = torch.zeros(n, device=env.device)
        vz_peak = torch.zeros(n, device=env.device)
        max_tilt = torch.zeros(n, device=env.device)
        for _ in range(int(args.hold_s / dt)):
            env.step(extend)
            a = airborne()
            rise = torch.where(a, torch.maximum(rise, z() - base), rise)
            run = torch.where(a, run + 1, torch.zeros_like(run))
            air_steps = torch.maximum(air_steps, run)
            vz_peak = torch.maximum(vz_peak, robot.data.root_lin_vel_w.torch[:, 2])
            max_tilt = torch.maximum(max_tilt, tilt_deg())
    rows = []
    for e in range(n):
        rows.append({**meta[e], "crouch_ok": bool(crouch_ok[e]), "crouch_drop_mm": round(float(base[e] - crouch_z[e]) * 1000, 1),
                     "rise_mm": round(float(rise[e]) * 1000, 1), "air_time_s": round(float(air_steps[e]) * dt, 2),
                     "vz_peak_m_s": round(float(vz_peak[e]), 3), "max_tilt_deg": round(float(max_tilt[e]), 1)})
    valid = [r for r in rows if r["crouch_ok"]]
    valid.sort(key=lambda r: -r["rise_mm"])
    best_by = {}
    for key in ("head", "gain", "signs"):
        for r in valid:
            k = f"{key}={r[key]}"
            if k not in best_by or r["rise_mm"] > best_by[k]["rise_mm"]:
                best_by[k] = r
    report = {"task": args.task, "overrides": hydra, "num_variants": n, "num_valid_crouches": len(valid),
              "top10": valid[:10], "best_by": best_by, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"valid crouches {len(valid)}/{n}")
    for r in valid[:10]:
        print(r)
    print("best by:", json.dumps(best_by, indent=1))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
