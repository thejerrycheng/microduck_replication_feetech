"""Forward roll over the head ("roulade" -- the duck's front flip), ported from Pollen's official
``microduck_rl`` task (``mjlab_microduck/tasks/mdp.py``, roulade section, attempt 3 / run 5).

The official design, kept term for term:

* ONE dense progress signal: paid increments of the max-so-far cumulative forward rotation, integrated only
  while the robot touches the ground (a roulade never leaves the floor; a ballistic whip earns nothing) and
  only while the roll stays in the sagittal plane (a shoulder roll earns nothing). Paid rate capped: faster
  than the cap forfeits the excess.
* Landing rewards gated on ROLL COMPLETION (rotation frontier past ~260 deg) AND an over-the-head latch (the
  flat top of the head touched the floor mid-roll), never on a clock.
* Reverse curriculum: a slice of episodes spawns mid-roll, tucked, with forward angular momentum.

Frame conventions in this build (measured 2026-11-08, ``scripts/`` discovery run): quaternions are xyzw,
forward roll is a POSITIVE body-frame angular velocity about +y, the head body is ``jaw_soft`` and world-up
expressed in its frame at HOME is ``HEAD_TOP_AXIS`` below.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

ROULADE_FWD_SIGN = 1.0
SUPPORT_SENSOR = "robot_ground"
HEAD_SENSOR = "head_ground"
HEAD_LATCH_LO = math.radians(20.0)
HEAD_LATCH_HI = math.radians(170.0)
HEAD_TOP_AXIS = (0.941, -0.011, -0.339)
"""World-up in the ``jaw_soft`` frame with the robot settled at HOME (the flat top of the head points up)."""
HEAD_TOP_DOWN_MIN = 0.3
FLAT_FULL = 0.5  # |lateral-axis z| = sin 30 deg: full rotation credit below
FLAT_ZERO = 0.866  # sin 60 deg: no credit above
CONTACT_FORCE_N = 1.0


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def _trunk_z(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    return torch.nan_to_num(asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _upright(asset: Articulation) -> torch.Tensor:
    q = asset.data.root_link_quat_w.torch  # xyzw
    return 1.0 - 2.0 * (q[:, 0].pow(2) + q[:, 1].pow(2))


def _lateral_axis_z(asset: Articulation) -> torch.Tensor:
    """World-z component of the body's lateral (+y) axis; 0 for any pure pitch, +-1 on a shoulder."""
    q = asset.data.root_link_quat_w.torch  # xyzw: x=q0 y=q1 z=q2 w=q3
    return 2.0 * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0])


def _omega_fwd(asset: Articulation) -> torch.Tensor:
    return ROULADE_FWD_SIGN * torch.nan_to_num(asset.data.root_link_ang_vel_b.torch[:, 1], nan=0.0)


def ground_contact(env: ManagerBasedEnv, sensor_name: str, threshold: float = CONTACT_FORCE_N) -> torch.Tensor:
    """Per-env bool: any tracked body of ``sensor_name`` presses on the filtered partner (the ground)."""
    sensor = env.scene.sensors[sensor_name]
    fm = sensor.data.force_matrix_w
    if fm is not None:
        f = fm.torch
        return (torch.linalg.norm(f, dim=-1) > threshold).flatten(1).any(dim=1)
    f = sensor.data.net_forces_w.torch
    return (torch.linalg.norm(f, dim=-1) > threshold).any(dim=1)


def _head_top_down(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
        env._roulade_head_axis = torch.tensor(HEAD_TOP_AXIS, device=env.device)
    q = asset.data.body_link_quat_w.torch[:, env._roulade_head_body_id]
    axis_w = math_utils.quat_apply(q, env._roulade_head_axis.expand(q.shape[0], 3))
    return axis_w[:, 2] < -HEAD_TOP_DOWN_MIN


def _state(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def update_accum(env: ManagerBasedEnv, asset: Articulation) -> None:
    """Integrate supported, sagittal forward rotation once per control step; latch over-the-head contact."""
    _state(env)
    step = int(env.common_step_counter)
    if step == env._roulade_last_step:
        return
    delta = _omega_fwd(asset) * env.step_dt
    delta = delta * ground_contact(env, SUPPORT_SENSOR).float()
    y_z = torch.nan_to_num(_lateral_axis_z(asset), nan=1.0).abs()
    delta = delta * _smoothstep(torch.clamp((FLAT_ZERO - y_z) / (FLAT_ZERO - FLAT_FULL), 0.0, 1.0))
    env._roulade_accum = env._roulade_accum + delta
    env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)
    in_window = (env._roulade_accum > HEAD_LATCH_LO) & (env._roulade_accum < HEAD_LATCH_HI)
    env._roulade_head_latch |= ground_contact(env, HEAD_SENSOR) & in_window & _head_top_down(env, asset)
    env._roulade_last_step = step


def completion_gate(env: ManagerBasedEnv, gate_lo: float, gate_hi: float, require_head: bool = True) -> torch.Tensor:
    _, max_accum, _ = _state(env)
    gate = _smoothstep(torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0))
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def _resolve_env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
    return env_ids.to(env.device, dtype=torch.long)


def reset_roulade_state(
    env: ManagerBasedEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = math.radians(5.0),
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(340.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 3.0),
    tuck_overrides: dict | None = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
) -> None:
    """Standing start or mid-roll reverse-curriculum spawn; resets the rotation accounting.

    Runs after ``reset_base`` / ``reset_robot_joints``. ``tuck_overrides`` is keyed by servo index (string
    keys) in :data:`MICRODUCK_JOINT_NAMES` order; mid-roll spawns lerp HOME -> tuck by a per-env factor and
    get a body-frame forward angular velocity. The accumulator starts at the spawn pitch, so a 170 deg spawn
    is only ever paid for the remaining rotation, and mid-roll spawns are born with the head latch set.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    num = len(env_ids)
    accum, max_accum, paid = _state(env)
    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=device) < (midroll_prob / max(total, 1e-6))

    yaw = (torch.rand(num, device=device) * 2.0 - 1.0) * math.pi
    pitch = (torch.rand(num, device=device) * 2.0 - 1.0) * standing_tilt_max
    mid_pitch = midroll_pitch_min + torch.rand(num, device=device) * (midroll_pitch_max - midroll_pitch_min)
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=device) * 2.0 - 1.0) * max(standing_tilt_max, math.radians(5.0))
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    z_stand = standing_z_min + torch.rand(num, device=device) * (standing_z_max - standing_z_min)
    z_mid = midroll_z_min + torch.rand(num, device=device) * (midroll_z_max - midroll_z_min)
    pos = asset.data.root_link_pos_w.torch[env_ids].clone()
    pos[:, 2] = torch.where(is_mid, z_mid, z_stand) + env.scene.env_origins[env_ids, 2]
    asset.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)

    vel = torch.zeros(num, 6, device=device)
    if midroll_omega_range[1] > 0.0:
        omega = midroll_omega_range[0] + torch.rand(num, device=device) * (midroll_omega_range[1] - midroll_omega_range[0])
        omega_b = torch.zeros(num, 3, device=device)
        omega_b[:, 1] = ROULADE_FWD_SIGN * omega
        vel[:, 3:6] = torch.where(is_mid.unsqueeze(-1), math_utils.quat_apply(quat, omega_b), torch.zeros_like(omega_b))
    if forward_vel_range[1] > 0.0:
        vx = forward_vel_range[0] + torch.rand(num, device=device) * (forward_vel_range[1] - forward_vel_range[0])
        vx = torch.where(is_mid, torch.zeros_like(vx), vx)
        vel[:, 0] = vx * torch.cos(yaw)
        vel[:, 1] = vx * torch.sin(yaw)
    asset.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    home = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos = home.clone()
    if tuck_overrides:
        u = tuck_factor_range[0] + torch.rand(num, device=device) * (tuck_factor_range[1] - tuck_factor_range[0])
        for index, angle in tuck_overrides.items():
            i = int(index)
            joint_pos[:, i] = home[:, i] + u * (float(angle) - home[:, i])
    if joint_noise_std > 0.0:
        joint_pos = joint_pos + torch.randn_like(joint_pos) * joint_noise_std
    joint_pos = torch.where(is_mid.unsqueeze(-1), joint_pos, home)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )

    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    env._roulade_head_latch[env_ids] = is_mid


# ── rewards ────────────────────────────────────────────────────────────────────────────────────────────


def roulade_progress(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_angle: float = 2 * math.pi, max_paid_rate: float = 5.0
) -> torch.Tensor:
    """Paid increments of the rotation frontier up to one full roll, capped at ``max_paid_rate`` rad/s."""
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    _, max_accum, paid = _state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
) -> torch.Tensor:
    """Head on the floor x mid-roll window x forward rate x (0.3 + 0.7 flat-top-down)."""
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    accum, _, _ = _state(env)
    contact = ground_contact(env, HEAD_SENSOR).float()
    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    rate = torch.clamp(_omega_fwd(asset) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def standing_composite_score(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list[int],
) -> torch.Tensor:
    """height Gaussian x upright Gaussian x HOME-pose Gaussian over ``joint_indices`` of the servo joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = _trunk_z(env, asset)
    height = torch.exp(-((z - target_height) / height_std) ** 2)
    q = asset.data.root_link_quat_w.torch
    tilt_sq = 2.0 * (q[:, 0].pow(2) + q[:, 1].pow(2))
    upright = torch.exp(-tilt_sq / (upright_std * upright_std))
    ids = [asset_cfg.joint_ids[i] for i in joint_indices]
    err = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    pose = torch.exp(-(err.pow(2).mean(dim=-1)) / (pose_std * pose_std))
    return height * upright * pose


def roulade_landing_composite(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list[int],
    gate_lo: float,
    gate_hi: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    score = standing_composite_score(env, asset_cfg, target_height, height_std, upright_std, pose_std, joint_indices)
    return score * completion_gate(env, gate_lo, gate_hi)


def roulade_upright_after_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gate_lo: float, gate_hi: float) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    return torch.clamp(_upright(asset), min=0.0) * completion_gate(env, gate_lo, gate_hi)


def roulade_height_after_roll(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, gate_lo: float, gate_hi: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    g = torch.exp(-((_trunk_z(env, asset) - target_height) / std) ** 2)
    return g * completion_gate(env, gate_lo, gate_hi)


def roulade_landing_sharp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    height_std: float,
    upright_std: float,
    gate_lo: float,
    gate_hi: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    q = asset.data.root_link_quat_w.torch
    tilt_sq = 2.0 * (q[:, 0].pow(2) + q[:, 1].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    height_g = torch.exp(-((_trunk_z(env, asset) - target_height) / height_std) ** 2)
    return upright_g * height_g * completion_gate(env, gate_lo, gate_hi)


def roulade_stand_tax(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, gate_lo: float, gate_hi: float) -> torch.Tensor:
    """SELF-NEGATING: -max(0, target - z) x completion gate; use a POSITIVE weight."""
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    shortfall = torch.clamp(target_height - _trunk_z(env, asset), min=0.0)
    return -shortfall * completion_gate(env, gate_lo, gate_hi)


def roulade_rise_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_height: float, gate_lo: float, gate_hi: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (_trunk_z(env, asset) < max_height).float()
    return reward * completion_gate(env, gate_lo, gate_hi)


def roulade_overspeed_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, omega_max: float = 7.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b.torch[:, 1], nan=0.0)
    return torch.clamp(omega_y.abs() - omega_max, min=0.0).pow(2)


def roulade_flatness_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset), nan=0.0).pow(2)


def roulade_sagittal_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b.torch
    return torch.nan_to_num(w[:, 0].pow(2) + w[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b.torch[:, 1].pow(2), nan=0.0)


def body_ang_vel_at_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_low: float,
    height_high: float,
    tilt_full_deg: float = 20.0,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """Trunk omega_xy^2 gated to standing height AND low tilt (arrival damper; use a negative weight)."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(w[:, :2]), dim=1)
    z = _trunk_z(env, asset)
    gate = _smoothstep(torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0))
    tilt = torch.rad2deg(torch.acos(torch.clamp(_upright(asset), -1.0, 1.0)))
    gate = gate * _smoothstep(torch.clamp((tilt_zero_deg - tilt) / max(tilt_zero_deg - tilt_full_deg, 1e-6), 0.0, 1.0))
    return cost * gate


# ── metrics: curriculum terms run at reset on the ENDING episodes and are logged as Curriculum/<name> ──


def roulade_frontier_deg(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Mean rotation frontier [deg] of the episodes ending now (a zero-weight reward would log 0 x value)."""
    _, max_accum, _ = _state(env)
    return torch.rad2deg(max_accum[env_ids]).mean()


def roulade_completed_frac(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Fraction of ending episodes that rolled past 330 deg with the head latched."""
    _, max_accum, _ = _state(env)
    done = (max_accum[env_ids] >= math.radians(330.0)) & env._roulade_head_latch[env_ids]
    return done.float().mean()


def roulade_standing_frac(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Fraction of ending episodes upright (tilt < 20 deg) above 0.095 m: completed-and-standing when ANDed."""
    asset: Articulation = env.scene[asset_cfg.name]
    up = _upright(asset)[env_ids] > math.cos(math.radians(20.0))
    tall = _trunk_z(env, asset)[env_ids] > 0.095
    _, max_accum, _ = _state(env)
    done = (max_accum[env_ids] >= math.radians(330.0)) & env._roulade_head_latch[env_ids]
    return (up & tall & done).float().mean()


def roulade_head_home_after_roll(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gate_lo: float, gate_hi: float, std: float = 0.25
) -> torch.Tensor:
    """After the roll (completion gate), pay for the head joints (``asset_cfg.joint_ids``: neck, head pitch/yaw/
    roll) being back at their default pose: exp(-(rms deviation / std)^2). The recipe scores only the trunk after
    the roll, so the deployed 6000 checkpoint stood up with the head still where the tuck left it."""
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    dev = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    rms = torch.sqrt(torch.mean(dev**2, dim=1))
    return torch.exp(-((rms / std) ** 2)) * completion_gate(env, gate_lo, gate_hi)


def roulade_head_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """RMS deviation [rad] of the head joints from their default pose (diagnostic)."""
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sqrt(torch.mean(dev**2, dim=1))


def roulade_head_l1_after_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gate_lo: float, gate_hi: float) -> torch.Tensor:
    """Landing-gated RMS head-joint deviation [rad] from the default pose (a positive cost: use a negative
    weight). Constant pull regardless of how far off the head is -- the Gaussian bonus alone moved the error only
    57.6 -> 50.0 deg in 500 iterations because its slope is small at a 1 rad error."""
    asset: Articulation = env.scene[asset_cfg.name]
    update_accum(env, asset)
    dev = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sqrt(torch.mean(dev**2, dim=1)) * completion_gate(env, gate_lo, gate_hi)
