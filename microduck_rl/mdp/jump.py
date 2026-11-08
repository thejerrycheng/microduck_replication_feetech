"""One-shot standing jump, ported from the community recipe with the most careful accounting
(ThomasBurgess2000/microduck-max-height-jump, v6) plus the measured lessons of chenp9527/microduck-jump:

* a one-shot LAUNCH flag in the twist-x command slot (1 while supported and inside the launch window, 0
  after touchdown), so the standing observation before the jump and after the landing are distinguishable;
* nothing is credited before the robot has stood on both feet for 4 control steps (spawn bounces earn nothing);
* flight pays only through paid-once FRONTIERS (launch speed, airborne rise above the per-episode standing
  baseline, air time), so camping, bobbing and repeated mini-hops cannot farm them;
* the largest rewards are reserved for a quiet, centred two-foot landing held for 1 s and, at the horizon, a
  DURABLE stand in the HOME pose -- there is no success termination (it hid delayed falls upstream);
* a failed recovery (not stable within 1.4 s of touchdown, or any body-ground contact after a valid takeoff)
  terminates with an explicit cost.

All per-episode state lives on the env object and is advanced once per control step by :func:`update_state`.
Quaternions are xyzw; the feet are the two ``ankle_*`` bodies of the ``contact_forces`` sensor; body-ground
contact is read from the ``robot_ground`` sensor (all bodies against the ground shape) excluding those feet.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

FEET_SENSOR = "contact_forces"
GROUND_SENSOR = "robot_ground"
CONTACT_N = 1.0
SUPPORT_STEPS = 4
TAKEOFF_VZ_MIN = 0.05
TAKEOFF_TILT_MAX_DEG = 30.0
MIN_RISE_M = 0.003
MIN_AIRTIME_S = 0.08
STABLE_TILT_DEG = 18.0
STABLE_Z_MIN = 0.085
STABLE_LIN_SPEED = 0.20
STABLE_ANG_SPEED = 1.5
STABLE_TIME_S = 1.0
RECOVERY_TIMEOUT_S = 1.40
RESET_POSE_RMS_RAD = 0.30
RESET_HEAD_ERR_RAD = 0.25
RESET_FOOT_DISTANCE_M = 0.070
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def _ids(env: ManagerBasedEnv) -> dict:
    if not hasattr(env, "_jump_ids"):
        robot: Articulation = env.scene["robot"]
        feet_sensor = env.scene.sensors[FEET_SENSOR]
        feet_rows = [i for i, b in enumerate(feet_sensor.body_names) if b.startswith("ankle")]
        ground_sensor = env.scene.sensors[GROUND_SENSOR]
        nonfoot_rows = [i for i, b in enumerate(ground_sensor.body_names) if not b.startswith("ankle")]
        feet_bodies, _ = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
        head_j, _ = robot.find_joints(list(HEAD_JOINTS), preserve_order=True)
        servo_j, _ = robot.find_joints([r"^(?!passive_).*"])
        assert len(feet_rows) == 2 and len(feet_bodies) == 2, (feet_sensor.body_names, feet_bodies)
        env._jump_ids = {"feet_rows": feet_rows, "nonfoot_rows": nonfoot_rows, "feet_bodies": feet_bodies, "head": head_j, "servo": servo_j}
    return env._jump_ids


def _state(env: ManagerBasedEnv) -> dict:
    if not hasattr(env, "_jump"):
        n, d = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=d)  # noqa: E731
        b = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        env._jump = {
            "baseline_z": z(), "grounded_steps": z(), "support": b(), "launch_frontier": z(), "launch_paid": z(),
            "airborne": b(), "air_steps": z(), "airtime_frontier": z(), "airtime_paid": z(), "rise_frontier": z(),
            "rise_paid": z(), "valid_takeoff": b(), "takeoff_pulse": b(), "landed": b(), "touchdown_pulse": b(),
            "post_td_steps": z(), "body_contact_ever": b(), "stable_steps": z(), "stable_latched": b(),
            "stable_pulse": b(), "recovery_prev": z(), "settle_max": z(), "failed": b(), "failed_pulse": b(),
            "descending_spawn": b(), "durable": b(), "last_step": -1,
        }
    return env._jump


def feet_contact(env: ManagerBasedEnv) -> torch.Tensor:
    """(N, 2) bool: each foot presses on something with more than CONTACT_N."""
    f = env.scene.sensors[FEET_SENSOR].data.net_forces_w.torch[:, _ids(env)["feet_rows"], :]
    return torch.linalg.norm(f, dim=-1) > CONTACT_N


def body_ground_force(env: ManagerBasedEnv) -> torch.Tensor:
    """(N,) summed ground force [N] on every non-foot body."""
    fm = env.scene.sensors[GROUND_SENSOR].data.force_matrix_w.torch[:, _ids(env)["nonfoot_rows"]]
    return torch.linalg.norm(fm, dim=-1).flatten(1).sum(dim=1)


def _trunk_z(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    return torch.nan_to_num(asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _upright(asset: Articulation) -> torch.Tensor:
    q = asset.data.root_link_quat_w.torch
    return 1.0 - 2.0 * (q[:, 0].pow(2) + q[:, 1].pow(2))


def _tilt_deg(asset: Articulation) -> torch.Tensor:
    return torch.rad2deg(torch.acos(torch.clamp(_upright(asset), -1.0, 1.0)))


def _pose_rms(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    ids = _ids(env)["servo"]
    err = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    return err.pow(2).mean(dim=1).sqrt()


def _head_err(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    ids = _ids(env)["head"]
    return (asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]).abs().max(dim=1).values


def _foot_distance(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    p = asset.data.body_link_pos_w.torch[:, _ids(env)["feet_bodies"], :]
    return torch.linalg.norm(p[:, 0, :2] - p[:, 1, :2], dim=-1)


def settle_score(env: ManagerBasedEnv, asset: Articulation, stand_z: float) -> torch.Tensor:
    """Product of Gaussians: upright, height, HOME pose, low linear/angular/joint speed, body clear."""
    cos_t = _upright(asset)
    z = _trunk_z(env, asset)
    v = torch.linalg.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    w = torch.linalg.norm(asset.data.root_ang_vel_w.torch, dim=-1)
    pose = _pose_rms(env, asset)
    jv = asset.data.joint_vel.torch[:, _ids(env)["servo"]].pow(2).mean(dim=1).sqrt()
    clear = (body_ground_force(env) <= CONTACT_N).float()
    return (
        torch.exp(-((1.0 - cos_t) / 0.22) ** 2) * torch.exp(-((z - stand_z) / 0.040) ** 2) * torch.exp(-(v / 0.45) ** 2)
        * torch.exp(-(w / 3.0) ** 2) * torch.exp(-(pose / 0.80) ** 2) * torch.exp(-(jv / 4.0) ** 2) * clear
    )


def update_state(env: ManagerBasedEnv, stand_z: float = 0.115) -> dict:
    """Advance the jump state machine once per control step (idempotent within a step)."""
    s = _state(env)
    step = int(env.common_step_counter)
    if step == s["last_step"]:
        return s
    s["last_step"] = step
    asset: Articulation = env.scene["robot"]
    dt = env.step_dt
    feet = feet_contact(env)
    both = feet.all(dim=1)
    any_foot = feet.any(dim=1)
    body = body_ground_force(env) > CONTACT_N
    z = _trunk_z(env, asset)
    vz = torch.nan_to_num(asset.data.root_lin_vel_w.torch[:, 2], nan=0.0)
    tilt = _tilt_deg(asset)
    fresh = env.episode_length_buf <= 1
    s["baseline_z"] = torch.where(fresh & ~s["descending_spawn"], z, s["baseline_z"])

    # support
    s["grounded_steps"] = torch.where(both & ~s["airborne"], s["grounded_steps"] + 1, torch.zeros_like(s["grounded_steps"]))
    s["support"] |= s["grounded_steps"] >= SUPPORT_STEPS
    pre = s["support"] & ~s["airborne"]
    s["launch_frontier"] = torch.where(pre, torch.maximum(s["launch_frontier"], torch.clamp(vz, min=0.0)), s["launch_frontier"])

    # takeoff
    takeoff = pre & ~any_foot & ~body & (vz >= TAKEOFF_VZ_MIN) & (tilt <= TAKEOFF_TILT_MAX_DEG)
    s["airborne"] |= takeoff
    s["takeoff_pulse"] = takeoff
    in_air = s["airborne"] & ~s["landed"] & ~any_foot
    s["air_steps"] = torch.where(in_air, s["air_steps"] + 1, s["air_steps"])
    s["airtime_frontier"] = torch.where(in_air, torch.maximum(s["airtime_frontier"], s["air_steps"] * dt), s["airtime_frontier"])
    rise = torch.clamp(z - s["baseline_z"], min=0.0)
    s["rise_frontier"] = torch.where(in_air & ~s["descending_spawn"], torch.maximum(s["rise_frontier"], rise), s["rise_frontier"])
    s["valid_takeoff"] |= s["airborne"] & (s["airtime_frontier"] >= MIN_AIRTIME_S) & ((s["rise_frontier"] >= MIN_RISE_M) | s["descending_spawn"])
    s["body_contact_ever"] |= s["airborne"] & body

    # touchdown and recovery
    touchdown = s["airborne"] & ~s["landed"] & any_foot & (s["air_steps"] >= 1)
    s["touchdown_pulse"] = touchdown
    s["landed"] |= touchdown
    s["post_td_steps"] = torch.where(s["landed"], s["post_td_steps"] + 1, s["post_td_steps"])
    v = torch.linalg.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    w = torch.linalg.norm(asset.data.root_ang_vel_w.torch, dim=-1)
    stable_now = (
        s["landed"] & both & ~s["body_contact_ever"] & (tilt <= STABLE_TILT_DEG) & (z >= STABLE_Z_MIN)
        & (v <= STABLE_LIN_SPEED) & (w <= STABLE_ANG_SPEED)
    )
    s["stable_now"] = stable_now
    s["stable_steps"] = torch.where(stable_now, s["stable_steps"] + 1, torch.zeros_like(s["stable_steps"]))
    newly = ~s["stable_latched"] & (s["stable_steps"] >= math.ceil(STABLE_TIME_S / dt))
    s["stable_pulse"] = newly
    s["stable_latched"] |= newly
    s["reset_stance"] = (_pose_rms(env, asset) <= RESET_POSE_RMS_RAD) & (_head_err(env, asset) <= RESET_HEAD_ERR_RAD) & (_foot_distance(env, asset) >= RESET_FOOT_DISTANCE_M)
    failed_now = (s["landed"] & ~s["stable_latched"] & (s["post_td_steps"] >= math.ceil(RECOVERY_TIMEOUT_S / dt))) | (s["valid_takeoff"] & s["body_contact_ever"])
    s["failed_pulse"] = failed_now & ~s["failed"]
    s["failed"] |= failed_now
    last = env.episode_length_buf >= env.max_episode_length - 1
    s["durable"] = last & s["valid_takeoff"] & s["stable_latched"] & stable_now & s["reset_stance"] & ~s["body_contact_ever"]
    return s


def reset_jump_state(
    env: ManagerBasedEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    standing_prob: float = 0.35,
    crouch_prob: float = 0.35,
    descending_prob: float = 0.30,
    standing_z_range: tuple = (0.112, 0.115),
    crouch_z_range: tuple = (0.068, 0.071),
    descending_z_range: tuple = (0.125, 0.150),
    descending_vz_range: tuple = (-0.35, -0.05),
    tilt_max_deg: float = 2.0,
    crouch_overrides: dict | None = None,
    joint_noise_std: float = 0.03,
) -> None:
    """Standing / foot-flat crouch / descending (reverse curriculum for the landing half) spawns."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    s = _state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    num = len(env_ids)
    u = torch.rand(num, device=device) * (standing_prob + crouch_prob + descending_prob)
    is_crouch = (u >= standing_prob) & (u < standing_prob + crouch_prob)
    is_desc = u >= standing_prob + crouch_prob

    yaw = (torch.rand(num, device=device) * 2.0 - 1.0) * math.pi
    tilt = math.radians(tilt_max_deg)
    quat = math_utils.quat_from_euler_xyz((torch.rand(num, device=device) * 2 - 1) * tilt, (torch.rand(num, device=device) * 2 - 1) * tilt, yaw)
    rng = lambda r: r[0] + torch.rand(num, device=device) * (r[1] - r[0])  # noqa: E731
    z = torch.where(is_desc, rng(descending_z_range), torch.where(is_crouch, rng(crouch_z_range), rng(standing_z_range)))
    pos = asset.data.root_link_pos_w.torch[env_ids].clone()
    pos[:, 2] = z + env.scene.env_origins[env_ids, 2]
    asset.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    vel = torch.zeros(num, 6, device=device)
    vel[:, 2] = torch.where(is_desc, rng(descending_vz_range), torch.zeros(num, device=device))
    asset.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    home = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids].clone()
    crouch = home.clone()
    for index, angle in (crouch_overrides or {}).items():
        crouch[:, int(index)] = float(angle)
    joint_pos = torch.where(is_crouch.unsqueeze(-1), crouch, home)
    if joint_noise_std > 0.0:
        joint_pos = joint_pos + torch.randn_like(joint_pos) * joint_noise_std
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    for key, val in s.items():
        if isinstance(val, torch.Tensor):
            val[env_ids] = False if val.dtype == torch.bool else 0.0
    s["baseline_z"][env_ids] = float(standing_z_range[1])
    s["descending_spawn"][env_ids] = is_desc
    # a descending spawn is already airborne after a (virtual) valid takeoff: it trains the landing half only
    s["support"][env_ids] = is_desc
    s["airborne"][env_ids] = is_desc
    s["launch_paid"][env_ids] = torch.where(is_desc, torch.full_like(z, 10.0), torch.zeros_like(z))
    s["rise_paid"][env_ids] = torch.where(is_desc, torch.full_like(z, 10.0), torch.zeros_like(z))
    s["airtime_paid"][env_ids] = torch.where(is_desc, torch.full_like(z, 10.0), torch.zeros_like(z))
    s["airtime_frontier"][env_ids] = torch.where(is_desc, torch.full_like(z, MIN_AIRTIME_S), torch.zeros_like(z))
    s["valid_takeoff"][env_ids] = is_desc


class JumpOneShotCommand(CommandTerm):
    """``[launch, 0, 0]`` in the twist slot: 1 while supported, inside the launch window and not yet landed."""

    cfg: "JumpOneShotCommandCfg"

    def __init__(self, cfg: "JumpOneShotCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._support_time = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self._support_time[env_ids] = 0.0

    def _update_command(self):
        s = update_state(self._env)
        self._support_time = torch.where(s["support"], self._support_time + self._env.step_dt, torch.zeros_like(self._support_time))
        launch = s["support"] & (self._support_time <= self.cfg.launch_window_s) & ~s["landed"] & ~s["descending_spawn"]
        self._command[:, 0] = launch.float()


@configclass
class JumpOneShotCommandCfg(CommandTermCfg):
    class_type: type[JumpOneShotCommand] = JumpOneShotCommand
    asset_name: str = "robot"
    launch_window_s: float = 0.75
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


# ── rewards (paid-once frontiers return delta/(dt*norm) so each integrates to weight x 1 per unit) ──────


def jump_launch_velocity_progress(env: ManagerBasedRLEnv, target_velocity: float = 0.60) -> torch.Tensor:
    s = update_state(env)
    delta = torch.clamp(s["launch_frontier"] - s["launch_paid"], min=0.0)
    s["launch_paid"] = torch.maximum(s["launch_paid"], s["launch_frontier"])
    return delta / (env.step_dt * target_velocity)


def jump_takeoff_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return s["takeoff_pulse"].float() / env.step_dt


def jump_visible_height_progress(env: ManagerBasedRLEnv, target_height: float = 0.012) -> torch.Tensor:
    s = update_state(env)
    delta = torch.clamp(s["rise_frontier"] - s["rise_paid"], min=0.0)
    s["rise_paid"] = torch.maximum(s["rise_paid"], s["rise_frontier"])
    return delta / (env.step_dt * target_height)


def jump_airtime_progress(env: ManagerBasedRLEnv, target_airtime: float = 0.10) -> torch.Tensor:
    s = update_state(env)
    frontier = torch.clamp(s["airtime_frontier"], max=target_airtime)
    delta = torch.clamp(frontier - s["airtime_paid"], min=0.0)
    s["airtime_paid"] = torch.maximum(s["airtime_paid"], frontier)
    return delta / (env.step_dt * target_airtime)


def jump_touchdown_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    vz = torch.clamp(-asset.data.root_lin_vel_w.torch[:, 2], min=0.0)
    v = torch.linalg.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    cos_t = _upright(asset)
    clear = (body_ground_force(env) <= CONTACT_N).float()
    score = torch.exp(-((vz / 0.35) ** 2)) * torch.exp(-(((1.0 - cos_t) / 0.12) ** 2)) * torch.exp(-((v / 0.45) ** 2)) * clear
    return s["touchdown_pulse"].float() * score / env.step_dt


def jump_recovery_progress(env: ManagerBasedRLEnv, stand_height: float = 0.115) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    r = torch.clamp(_upright(asset), 0.0, 1.0) * torch.clamp(_trunk_z(env, asset) / stand_height, 0.0, 1.0)
    active = s["landed"] & ~s["stable_latched"]
    delta = torch.where(active, r - s["recovery_prev"], torch.zeros_like(r))
    s["recovery_prev"] = torch.where(s["landed"], r, torch.zeros_like(r))
    return delta / env.step_dt


def jump_settle_progress(env: ManagerBasedRLEnv, stand_height: float = 0.115) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    score = settle_score(env, asset, stand_height)
    active = s["landed"] & ~s["durable"]
    new_max = torch.where(active, torch.maximum(s["settle_max"], score), s["settle_max"])
    delta = new_max - s["settle_max"]
    s["settle_max"] = new_max
    return delta / env.step_dt


def jump_stable_landing_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return (s["stable_pulse"] & (s["valid_takeoff"] | s["descending_spawn"])).float() / env.step_dt


def jump_durable_landing_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return s["durable"].float() / env.step_dt


def jump_durable_visible_height_bonus(env: ManagerBasedRLEnv, target_height: float = 0.012) -> torch.Tensor:
    s = update_state(env)
    return s["durable"].float() * torch.log1p(s["rise_frontier"] / target_height) / env.step_dt


def jump_failed_recovery_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return s["failed_pulse"].float() / env.step_dt


def jump_post_touchdown_pose_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return _pose_rms(env, env.scene["robot"]) * s["landed"].float()


def jump_post_touchdown_head_pose_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    return _head_err(env, env.scene["robot"]) * s["landed"].float()


def jump_post_touchdown_stance_width_cost(env: ManagerBasedRLEnv, min_distance: float = RESET_FOOT_DISTANCE_M) -> torch.Tensor:
    s = update_state(env)
    return torch.clamp(min_distance - _foot_distance(env, env.scene["robot"]), min=0.0) / min_distance * s["landed"].float()


def jump_post_touchdown_motion_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    v = torch.linalg.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    w = torch.linalg.norm(asset.data.root_ang_vel_w.torch, dim=-1)
    return (v / 0.45 + w / 3.0) * s["landed"].float()


def jump_nonfoot_impact_cost(env: ManagerBasedRLEnv, threshold: float = CONTACT_N) -> torch.Tensor:
    update_state(env)
    return torch.clamp(body_ground_force(env) - threshold, min=0.0)


def jump_airborne_tilt_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    in_air = s["airborne"] & ~s["landed"]
    return (1.0 - _upright(asset)) * in_air.float()


def jump_horizontal_drift_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    s = update_state(env)
    asset: Articulation = env.scene["robot"]
    in_air = s["airborne"] & ~s["landed"]
    return asset.data.root_lin_vel_w.torch[:, :2].pow(2).sum(dim=1) * in_air.float()


def jump_failed_recovery(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination: recovery timed out after touchdown, or a body hit the ground after a valid takeoff."""
    s = update_state(env)
    return s["failed"]


# ── metrics (curriculum terms, logged as Curriculum/<name> on the ending episodes) ─────────────────────


def jump_valid_takeoff_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    m = ~s["descending_spawn"][env_ids]
    return (s["valid_takeoff"][env_ids] & m).float().sum() / m.float().sum().clamp(min=1)


def jump_rise_mm(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    m = ~s["descending_spawn"][env_ids]
    return (s["rise_frontier"][env_ids] * m).sum() * 1000.0 / m.float().sum().clamp(min=1)


def jump_airtime_ms(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    m = ~s["descending_spawn"][env_ids]
    return (s["airtime_frontier"][env_ids] * m).sum() * 1000.0 / m.float().sum().clamp(min=1)


def jump_stable_landing_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    return s["stable_latched"][env_ids].float().mean()


def jump_durable_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    return s["durable"][env_ids].float().mean()


def jump_body_contact_rate(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    return s["body_contact_ever"][env_ids].float().mean()
