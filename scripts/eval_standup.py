#!/usr/bin/env python
"""Deterministic (mean-action) get-up evaluation per START FAMILY, one 6 s episode each, no pushes.

Families: prone (face down), supine (face up, +-90 deg roll), left side, right side, random orientation with
random joints, fall bank (if ``docs/fall_bank.json`` exists), sitting, and standing-with-pushes (hold test).
Reports standing fraction (tilt < 20 deg and trunk > 0.095 m) at 3 s and at the end, time-to-stand median/p90,
peak |omega| median, and the fraction that stood and then fell again.

    uv run --project external/IsaacLab python scripts/eval_standup.py --run <run dir> --checkpoint model_4000.pt presets=backlash
"""

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_rl  # noqa: E402, F401
from microduck_rl.mdp import standup as S  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

FAMILY_PARAMS = {
    "prone": dict(standing_prob=0, sitting_prob=0, prone_prob=1, supine_prob=0, random_prob=0, bank_prob=0),
    "supine": dict(standing_prob=0, sitting_prob=0, prone_prob=0, supine_prob=1, random_prob=0, bank_prob=0, face_up_roll_max_deg=20.0),
    "side": dict(standing_prob=0, sitting_prob=0, prone_prob=0, supine_prob=1, random_prob=0, bank_prob=0, face_up_roll_max_deg=90.0),
    "random": dict(standing_prob=0, sitting_prob=0, prone_prob=0, supine_prob=0, random_prob=1, bank_prob=0),
    "bank": dict(standing_prob=0, sitting_prob=0, prone_prob=0, supine_prob=0, random_prob=0, bank_prob=1),
    "sitting": dict(standing_prob=0, sitting_prob=1, prone_prob=0, supine_prob=0, random_prob=0, bank_prob=0),
    "standing_pushed": dict(standing_prob=1, sitting_prob=0, prone_prob=0, supine_prob=0, random_prob=0, bank_prob=0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Microduck-StandUp-Flat-HD1910")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--push", type=float, default=1.0, help="push magnitude [m/s] for the standing_pushed family, every 1.5 s")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", default=None, help="comma-separated subset of the fall families (default: all)")
    ap.add_argument("--zero_twist", action="store_true", help="velocity-command envs (VelStand): command exactly zero twist throughout")
    args, extra = ap.parse_known_args()
    run = args.run.resolve()
    name = args.checkpoint.replace(".pt", "").replace("model_", "")
    export_dir = run / f"exported_model_{name}"
    policy_path = export_dir / "policy.pt"
    if not policy_path.exists():
        play = Path(__file__).resolve().parent / "play.py"
        subprocess.run([sys.executable, str(play), "--rl_library", "rsl_rl", "--task", args.task, "--checkpoint", str(run / args.checkpoint),
                        "--num_envs", "4", "--video", "--video_length", "1", "--video_interval", "1000000", *extra],
                       check=True, cwd=Path(__file__).resolve().parent.parent / "external" / "IsaacLab")
        export_dir.mkdir(exist_ok=True)
        for f in ("policy.pt", "policy.onnx", "policy.onnx.data"):
            if (run / "exported" / f).exists():
                shutil.copy(run / "exported" / f, export_dir / f)
    families = [f for f in FAMILY_PARAMS if f != "bank" or os.path.exists("docs/fall_bank.json")]
    if args.families:
        families = [f for f in args.families.split(",") if f in families]
    results = {}
    for fam in families:
        cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, overrides=extra)
        cfg.play_mode()
        cfg.scene.num_envs = args.num_envs
        cfg.curriculum = None
        cfg.events.set_fallen_state.params.update(FAMILY_PARAMS[fam])
        if args.zero_twist:  # a velocity env: pin every twist range to zero and make every env a standing env
            rng = cfg.commands.base_velocity.ranges
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if hasattr(rng, name):
                    setattr(rng, name, (0.0, 0.0))
            if hasattr(cfg.commands.base_velocity, "rel_standing_envs"):
                cfg.commands.base_velocity.rel_standing_envs = 1.0
        env = gym.make(args.task, cfg=cfg).unwrapped
        robot = env.scene["robot"]
        policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
        obs, _ = env.reset()
        n, dt = env.num_envs, env.step_dt
        steps = int(5.9 / dt)
        stood_step = torch.full((n,), -1, device=env.device)
        peak_w = torch.zeros(n, device=env.device)
        fell_after = torch.zeros(n, dtype=torch.bool, device=env.device)
        at3 = None
        with torch.no_grad():
            for t in range(steps):
                if fam == "standing_pushed" and t % int(1.5 / dt) == int(0.5 / dt):
                    ang = torch.rand(n, device=env.device) * 2 * math.pi
                    v = robot.data.root_lin_vel_w.torch.clone()
                    v[:, 0] += args.push * torch.cos(ang)
                    v[:, 1] += args.push * torch.sin(ang)
                    robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat([v, robot.data.root_ang_vel_w.torch], 1), env_ids=torch.arange(n, device=env.device))
                obs, *_ = env.step(policy(obs["policy"]))
                up = (S._upright(robot) > math.cos(math.radians(20.0))) & (S._trunk_z(env, robot) > 0.095)
                stood_step = torch.where(up & (stood_step < 0), torch.full_like(stood_step, t), stood_step)
                fell_after |= (stood_step >= 0) & (S._upright(robot) < math.cos(math.radians(60.0)))
                peak_w = torch.maximum(peak_w, torch.linalg.norm(robot.data.root_ang_vel_w.torch, dim=-1))
                if t == int(3.0 / dt):
                    at3 = up.clone()
        up_end = (S._upright(robot) > math.cos(math.radians(20.0))) & (S._trunk_z(env, robot) > 0.095)
        tts = stood_step[stood_step >= 0].float() * dt
        # End-state diagnostics: what "not standing" is (a lean past 20 deg, a low trunk, or down past 60 deg).
        tilt_end = torch.rad2deg(torch.acos(torch.clamp(S._upright(robot), -1.0, 1.0)))
        trunk_end = S._trunk_z(env, robot)
        end_diag = {
            "tilt_end_median_deg": round(float(tilt_end.median()), 1),
            "trunk_end_median_m": round(float(trunk_end.median()), 4),
            "end_tilt_20_60_frac": round(float(((tilt_end >= 20.0) & (tilt_end < 60.0)).float().mean()), 3),
            "end_down_frac": round(float((tilt_end >= 60.0).float().mean()), 3),
            "end_low_trunk_upright_frac": round(float(((tilt_end < 20.0) & (trunk_end <= 0.095)).float().mean()), 3),
        }
        low_up = (tilt_end < 20.0) & (trunk_end <= 0.095)
        if bool(low_up.any()):
            v_end = torch.linalg.norm(robot.data.root_lin_vel_w.torch[:, :2], dim=-1)
            end_diag["low_trunk_speed_median_m_s"] = round(float(v_end[low_up].median()), 3)
            end_diag["low_trunk_moving_frac"] = round(float((v_end[low_up] > 0.15).float().mean()), 3)
            end_diag["low_trunk_trunk_median_m"] = round(float(trunk_end[low_up].median()), 4)
            names = list(robot.joint_names)
            q = robot.data.joint_pos.torch
            for key in ("left_knee", "right_knee", "left_hip_pitch", "right_hip_pitch"):
                if key in names:
                    j = names.index(key)
                    end_diag[f"low_trunk_{key}_median_rad"] = round(float(q[low_up, j].median()), 3)
                    end_diag[f"ok_{key}_median_rad"] = round(float(q[~low_up & (tilt_end < 20.0), j].median()), 3) if bool((~low_up & (tilt_end < 20.0)).any()) else None
        if hasattr(env, "command_manager"):
            try:
                cmd = env.command_manager.get_command("base_velocity")
                end_diag["cmd_vx_median"] = round(float(cmd[:, 0].median()), 3)
            except Exception:
                pass
        results[fam] = {
            "standing_at_3s": round(float(at3.float().mean()), 3), "standing_at_end": round(float(up_end.float().mean()), 3),
            "ever_stood": round(float((stood_step >= 0).float().mean()), 3), "fell_after_standing": round(float(fell_after.float().mean()), 3),
            "time_to_stand_s": {"median": round(float(tts.median()), 2) if tts.numel() else None, "p90": round(float(tts.quantile(0.9)), 2) if tts.numel() else None},
            "peak_omega_median": round(float(peak_w.median()), 2),
            **end_diag,
        }
        print(fam, results[fam])
        env.close()
    report = {"checkpoint": str(run / args.checkpoint), "num_envs": args.num_envs, "push_m_s": args.push, "families": results}
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
