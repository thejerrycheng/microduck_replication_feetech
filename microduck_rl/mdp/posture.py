"""Sit/stand posture terms, ported from ``pollen-robotics/microduck_rl`` (mjlab) to Isaac Lab.

Source: ``src/mjlab_microduck/tasks/mdp.py`` (``SitStandCommand``, ``posture_*``, the trunk
speed caps, the seated fall-back hardening events) and ``microduck_sitstand_env_cfg.py``.
Function names, semantics and sign conventions are kept so the upstream reward weights
apply unchanged. Note the upstream convention: the ``*_penalty`` functions already return
negative values and are used with POSITIVE weights.

Every term takes a ``asset_cfg`` that selects the 14 servo joints in servo order
(``preserve_order=True``); the sit keyframe is keyed by servo index.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.mdp.observations import zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "trunk_height_error_l1",
    "forward_speed_reward",
    "joint_speed_saturation_frac",
    "SitStandCommand",
    "SitStandCommandCfg",
    "posture_pose_match",
    "posture_pose_l1",
    "head_pose_l1",
    "posture_height_gaussian",
    "posture_height_l1",
    "posture_composite",
    "posture_stillness",
    "posture_rise_bootstrap",
    "trunk_downward_velocity_penalty",
    "trunk_upward_velocity_penalty",
    "trunk_vertical_accel_penalty",
    "body_upright_linear",
    "upright_while_tall",
    "joint_torque_rate_l2",
    "neck_torque_when_tilted",
    "set_random_ground_state",
    "seated_head_command_jerk",
    "seated_backward_tip",
]


"""
Posture command.
"""


class SitStandCommand(CommandTerm):
    """Posture command ``[sit_flag, 0, 0]`` with dwell-time resampling and a slewed target blend.

    ``sit_flag`` is 0 (STAND) or 1 (SIT); it occupies the twist slot so the observation keeps the
    61-wide deployment layout. ``alpha`` slews toward the flag at a constant rate (a full STAND<->SIT
    transition in ``ramp_s``) and is what the posture rewards track: with a binary target, arriving
    early pays the goal-state reward for every step saved, and the speed caps cannot out-price that.
    On episode start ``alpha`` is initialised from the ACTUAL trunk height, so a seated spawn is not
    dragged upward by a stand-initialised ramp.
    """

    cfg: SitStandCommandCfg

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """``[sit_flag, 0, 0]``, shape (num_envs, 3)."""
        return self._command

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed target blend, 0 = STAND target, 1 = SIT target, shape (num_envs,)."""
        return self._alpha

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        sit = (torch.rand(len(env_ids), device=self.device) < self.cfg.sit_prob).float()
        self._command[env_ids] = 0.0
        self._command[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2], nan=self.cfg.stand_z
        )
        return torch.clamp((self.cfg.stand_z - z) / max(self.cfg.stand_z - self.cfg.sit_z, 1e-6), 0.0, 1.0)

    def _update_command(self):
        # Freshly reset environments re-seed the blend from where the robot actually is.
        fresh = self._env.episode_length_buf <= 1
        self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        step = self._env.step_dt / max(self.cfg.ramp_s, 1e-6)
        delta = self._command[:, 0] - self._alpha
        self._alpha = self._alpha + torch.clamp(delta, -step, step)


@configclass
class SitStandCommandCfg(CommandTermCfg):
    """Configuration for :class:`SitStandCommand`."""

    class_type: type[SitStandCommand] = SitStandCommand
    asset_name: str = "robot"
    sit_prob: float = 0.5
    """Probability that a resample commands SIT (1.0: always sit, 0.0: always stand)."""
    ramp_s: float = 2.0
    """Seconds for the target blend to traverse STAND<->SIT in full."""
    sit_z: float = 0.060
    """Seated trunk height [m], used to initialise the blend from the spawn state."""
    stand_z: float = 0.115
    """Standing trunk height [m]."""


"""
Shared helpers.
"""


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """Event managers pass a slice for "all environments"; turn any form into a long index tensor."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _trunk_z(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    return torch.nan_to_num(asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _posture_blend(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _sit_target(asset: Articulation, asset_cfg: SceneEntityCfg, sit_overrides: dict) -> torch.Tensor:
    stand = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    sit = stand.clone()
    for index, value in sit_overrides.items():
        sit[:, int(index)] = float(value)
    return stand, sit


def _posture_targets(
    env: ManagerBasedRLEnv, asset: Articulation, asset_cfg: SceneEntityCfg, command_name: str, sit_overrides: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env servo target): HOME blended toward the SIT keyframe by the slewed alpha."""
    blend = _posture_blend(env, command_name)
    stand, sit = _sit_target(asset, asset_cfg, sit_overrides)
    return blend, stand + blend.unsqueeze(-1) * (sit - stand)


def _posture_height(
    env: ManagerBasedRLEnv, asset: Articulation, command_name: str, sit_z: float, stand_z: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    return stand_z + blend * (sit_z - stand_z), _trunk_z(env, asset)


def _tilt_sq(asset: Articulation) -> torch.Tensor:
    """``1 - cos(tilt)`` of the trunk (= ``2 (qx^2 + qy^2)`` upstream), a tilt-squared proxy for small angles.

    Read from the projected gravity rather than the quaternion: Newton stores ``root_link_quat_w`` as
    ``(x, y, z, w)``, so upstream's component indices would read yaw. Upright gives 0, horizontal 1, inverted 2.
    """
    gravity_b = torch.nan_to_num(asset.data.projected_gravity_b.torch, nan=0.0)
    return torch.clamp(1.0 + gravity_b[:, 2], 0.0, 2.0)


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


"""
Posture rewards.
"""


def posture_pose_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list[int],
    asset_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian match of the selected servo joints against the commanded posture's target pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, asset_cfg, command_name, sit_overrides)
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids][:, joint_indices]
    error = joint_pos - target[:, joint_indices]
    return zero_non_finite(torch.exp(-((error / std) ** 2)).mean(dim=-1))


def posture_pose_l1(
    env: ManagerBasedRLEnv, command_name: str, sit_overrides: dict, joint_indices: list[int], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """L1 companion to :func:`posture_pose_match` (constant gradient to target); returns <= 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, asset_cfg, command_name, sit_overrides)
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids][:, joint_indices]
    return zero_non_finite(-torch.abs(joint_pos - target[:, joint_indices]).mean(dim=-1))


def head_pose_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "head_pose"
) -> torch.Tensor:
    """L1 distance of the neck/head joints from their commanded offsets; returns <= 0.

    Not in upstream. Both upstream head terms (``head_pose_tracking``, the composite's head factor) are
    Gaussians with std 0.4--0.5 rad; once the policy parks the head on the floor (1.3 rad away, measured
    at iterations 250--2000 of the first HD1910 sit run) they are numerically flat and nothing pulls the
    head back up. This is the head counterpart of ``posture_pose_l1``: a constant gradient everywhere.
    ``asset_cfg`` selects the 4 head joints in servo order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    error = actual - env.command_manager.get_command(command_name)
    return zero_non_finite(-torch.abs(error).mean(dim=-1))


def posture_height_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian on trunk z against the commanded posture's (slewed) target height."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, asset, command_name, sit_z, stand_z)
    return zero_non_finite(torch.exp(-((z - target_z) / std) ** 2))


def posture_height_l1(
    env: ManagerBasedRLEnv, command_name: str, sit_z: float, stand_z: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """L1 companion to :func:`posture_height_gaussian`, the transition driver; returns <= 0.

    Resting in the WRONG posture charges a constant per-step cost (~55 mm), which is what makes
    ignoring the command a net-negative strategy in both directions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, asset, command_name, sit_z, stand_z)
    return zero_non_finite(-torch.abs(z - target_z))


def posture_composite(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list[int],
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg | None = None,
    head_command_name: str = "head_pose",
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
) -> torch.Tensor:
    """Multiplicative goal score vs the commanded posture: height * upright * pose [* head].

    A deficiency in any factor collapses the whole term, so partial-sum compromises (plank, flop,
    lean, park 1 cm short) never pay. With ``head_std`` and ``head_cfg`` the goal state also requires
    the head at its commanded pose, so transient head assist mid-transition stays free but must be
    retracted to collect the goal reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, asset_cfg, command_name, sit_overrides)
    target_z, z = _posture_height(env, asset, command_name, sit_z, stand_z)
    height_score = torch.exp(-((z - target_z) / height_std) ** 2)
    upright_score = torch.exp(-_tilt_sq(asset) / (upright_std * upright_std))
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids][:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    score = height_score * upright_score * torch.exp(-pose_err_sq / (pose_std * pose_std))
    if head_std is not None and head_cfg is not None:
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = (
            asset.data.joint_pos.torch[:, head_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, head_cfg.joint_ids]
        )
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))
    return zero_non_finite(score)


def posture_stillness(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
) -> torch.Tensor:
    """Reward trunk stillness while AT the commanded posture, upright, with the target ramp complete.

    ``exp(-(|v|/std)^2)`` gated by a smoothstep on ``|z - commanded z|`` (inactive during
    transitions), by trunk tilt (a tilted rest earns nothing) and by ``|flag - alpha| < 0.02``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, asset, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch, nan=0.0).norm(dim=-1)
    flag = env.command_manager.get_command(command_name)[:, 0]
    ramp_done = ((flag - _posture_blend(env, command_name)).abs() < 0.02).float()
    z_gate = _smoothstep((band_zero - torch.abs(z - target_z)) / max(band_zero - band_full, 1e-6))
    cos_tilt = 1.0 - _tilt_sq(asset)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    tilt_gate = _smoothstep((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6))
    return zero_non_finite(torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done)


def posture_rise_bootstrap(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_height: float,
    asset_cfg: SceneEntityCfg,
    max_vz: float | None = None,
) -> torch.Tensor:
    """Upward trunk velocity, paid only while STAND is commanded and ``z < max_height``.

    Destination-only rewards have zero gradient at zero motion, so "stay seated and eat the L1"
    is a local optimum; paying for the rise motion itself makes any attempt immediately positive.
    ``max_vz`` caps the rewarded speed so an explosive launch cannot out-earn a gentle rise.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = _trunk_z(env, asset)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


"""
Gentleness (these return <= 0 and take POSITIVE weights upstream).
"""


def trunk_downward_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_down_vel: float = 0.05
) -> torch.Tensor:
    """``-max(0, -vz - max_down_vel)``: every step of a too-fast descent costs reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_up_vel: float = 0.08
) -> torch.Tensor:
    """``-max(0, vz - max_up_vel)``: the mirror cap for a violent stand-up."""
    asset: Articulation = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)


class trunk_vertical_accel_penalty(ManagerTermBase):
    """``-|a_z|`` of the trunk (finite difference of vz), zeroed on the step after a reset."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_vz = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_vz[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
        a_z = (vz - self._prev_vz) / env.step_dt
        a_z = torch.where(env.episode_length_buf <= 1, torch.zeros_like(a_z), a_z)
        self._prev_vz = vz.clone()
        return -torch.abs(a_z)


"""
Uprightness.
"""


def body_upright_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """``cos(tilt)``: +1 upright, 0 horizontal, -1 inverted; non-zero gradient at every tilt."""
    asset: Articulation = env.scene[asset_cfg.name]
    return zero_non_finite(1.0 - _tilt_sq(asset))


def upright_while_tall(
    env: ManagerBasedRLEnv, height_low: float, height_high: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """:func:`body_upright_linear` weighted by a smoothstep on trunk z: full standing tall, zero once seated.

    Blocks the "tip backward while still high" descent exploit; doubles as an arrival-uprightness pull.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = _trunk_z(env, asset)
    smooth = _smoothstep((z - height_low) / max(height_high - height_low, 1e-6))
    return zero_non_finite((1.0 - _tilt_sq(asset)) * smooth)


"""
Servo load.
"""


class joint_torque_rate_l2(ManagerTermBase):
    """Sum of squared per-step changes in applied joint torque (gearbox-shock proxy), >= 0."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        num_joints = len(cfg.params["asset_cfg"].joint_ids)
        self._prev = torch.zeros(env.num_envs, num_joints, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        torque = torch.nan_to_num(asset.data.applied_torque.torch[:, asset_cfg.joint_ids], nan=0.0)
        rate = torque - self._prev
        rate = torch.where((env.episode_length_buf <= 1).unsqueeze(-1), torch.zeros_like(rate), rate)
        self._prev = torque.clone()
        return torch.sum(torch.square(rate), dim=1)


def neck_torque_when_tilted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    tilt_start_deg: float = 12.0,
    tilt_full_deg: float = 25.0,
    torque_scale: float = 0.1,
) -> torch.Tensor:
    """Cost >= 0 (NEGATIVE weight): mean ``(|tau_neck| / torque_scale)^2``, gated on a backward lean while seated.

    ``asset_cfg`` selects the 4 neck/head joints. Gate = posture blend (SIT target) x backward lean
    (projected gravity x < 0; trunk +x is forward) x tilt ramp from ``tilt_start_deg`` to ``tilt_full_deg``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = torch.nan_to_num(asset.data.applied_torque.torch[:, asset_cfg.joint_ids], nan=0.0)
    cost = torch.square(tau / torque_scale).mean(dim=1)
    g = torch.nan_to_num(asset.data.projected_gravity_b.torch, nan=0.0)
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
    gate = torch.clamp((tilt - tilt_start_deg) / max(tilt_full_deg - tilt_start_deg, 1e-6), 0.0, 1.0)
    gate = gate * (g[:, 0] < 0.0).float() * _posture_blend(env, command_name)
    return zero_non_finite(cost * gate)


"""
Reset and interval events.
"""


def set_random_ground_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    sitting_prob: float = 0.5,
    standing_prob: float = 0.5,
    sitting_z_min: float = 0.06,
    sitting_z_max: float = 0.075,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: dict | None = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
) -> None:
    """Reset each environment to a seated or a standing start (upstream's sit/stand buckets).

    Runs AFTER ``reset_base`` and ``reset_robot_joints``: keeps their xy, draws a fresh yaw, writes an
    absolute trunk height above the env origin, zero velocity, and for the seated bucket the SIT keyframe
    (``asset_cfg`` servo indices -> angles) with per-joint Gaussian noise and +-``sitting_tilt_max`` tilt.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    num = len(env_ids)
    total = sitting_prob + standing_prob
    is_sit = torch.rand(num, device=device) < (sitting_prob / total)

    yaw = (torch.rand(num, device=device) * 2.0 - 1.0) * math.pi
    pitch = (torch.rand(num, device=device) * 2.0 - 1.0) * sitting_tilt_max
    roll = (torch.rand(num, device=device) * 2.0 - 1.0) * sitting_tilt_max
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    z_sit = sitting_z_min + torch.rand(num, device=device) * (sitting_z_max - sitting_z_min)
    z_stand = standing_z_min + torch.rand(num, device=device) * (standing_z_max - standing_z_min)
    pos = asset.data.root_link_pos_w.torch[env_ids].clone()
    pos[:, 2] = torch.where(is_sit, z_sit, z_stand) + env.scene.env_origins[env_ids, 2]
    asset.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=device), env_ids=env_ids)

    stand_pose = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids].clone()
    sit_pose = stand_pose.clone()
    for index, value in (sitting_joint_overrides or {}).items():
        sit_pose[:, int(index)] = float(value)
    if sitting_joint_noise_std > 0.0:
        sit_pose += torch.randn_like(sit_pose) * sitting_joint_noise_std
    joint_pos = torch.where(is_sit.unsqueeze(-1), sit_pose, stand_pose)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def _seated_mask(env: ManagerBasedRLEnv, command_name: str, min_alpha: float) -> torch.Tensor:
    sit = env.command_manager.get_command(command_name)[:, 0] > 0.5
    return sit & (_posture_blend(env, command_name) >= min_alpha)


def seated_head_command_jerk(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prob: float,
    flip_frac: float = 0.5,
    posture_command_name: str = "base_velocity",
    head_command_name: str = "head_pose",
    min_alpha: float = 0.98,
) -> None:
    """Interval event: abrupt head-command changes for SEATED environments.

    Each selected seated env (probability ``prob``) gets either a sign flip of its head command
    (``flip_frac`` of them: a full-range swing in one step) or a fresh uniform sample in the command's
    live ranges. Supplies the sub-second head motion a gamepad produces, which the dwell-resampled
    command never trains; on the real robot that is what tips a seated duck onto its back.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if prob <= 0.0 or len(env_ids) == 0:
        return
    pick = _seated_mask(env, posture_command_name, min_alpha)[env_ids]
    pick &= torch.rand(len(env_ids), device=env.device) < prob
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    term = env.command_manager.get_term(head_command_name)
    low, high = torch.tensor(term.cfg.ranges, device=env.device).unbind(-1)
    fresh = low + (high - low) * torch.rand(len(ids), term.dim, device=env.device)
    flip = (torch.rand(len(ids), device=env.device) < flip_frac).unsqueeze(-1)
    current = term.command[ids]
    term.command[ids] = torch.where(flip, -current, fresh)


def seated_backward_tip(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ang_vel_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    prob: float = 1.0,
    lateral_frac: float = 0.3,
    posture_command_name: str = "base_velocity",
    min_alpha: float = 0.98,
) -> None:
    """Interval event: backward (nose-up) pitch kick of ``|omega|`` in ``ang_vel_range`` for SEATED envs.

    ``lateral_frac`` mixes a random +- roll component for backward-diagonal falls. Generates the leaned-back
    seated states no spawn covers (teleporting into a lean at sit height penetrates the floor).
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if len(env_ids) == 0 or ang_vel_range[1] <= 0.0:
        return
    pick = _seated_mask(env, posture_command_name, min_alpha)[env_ids]
    pick &= torch.rand(len(env_ids), device=env.device) < prob
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(ids)
    mag = torch.empty(n, device=env.device).uniform_(*ang_vel_range)
    roll = (torch.rand(n, device=env.device) * 2.0 - 1.0) * lateral_frac
    # trunk +x is forward and +omega_y pitches nose-down, so the backward kick is -omega_y.
    omega_b = torch.stack((roll * mag, -mag, torch.zeros_like(mag)), dim=1)
    omega_w = math_utils.quat_apply(asset.data.root_link_quat_w.torch[ids], omega_b)
    vel = asset.data.root_link_vel_w.torch[ids].clone()
    vel[:, 3:6] += omega_w
    asset.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=ids)


def trunk_height_error_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.115) -> torch.Tensor:
    """|trunk z - target| [m], a positive cost (use a negative weight). Always on: a walking duck keeps its
    trunk near standing height, and this is what removes the squat the walk policy adopted at zero command
    (z 0.067-0.092 measured) and at the onset of a turn."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (_trunk_z(env, asset) - target_height).abs()


def forward_speed_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, cap: float = 2.0) -> torch.Tensor:
    """Body-frame forward speed [m/s], clipped at ``cap``: the sprint-max objective (pays speed itself, not
    tracking a command, so the policy is pushed to whatever the servos allow)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_b.torch[:, 0], min=-cap, max=cap)


def joint_speed_saturation_frac(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, limit_rad_s: float = 10.0) -> torch.Tensor:
    """Curriculum-logged metric: fraction of servo joints currently above ``limit_rad_s`` (the HD1910 no-load
    speed is ~11.9 rad/s at 7.4 V): how hard the gait leans on the servo speed wall."""
    asset: Articulation = env.scene[asset_cfg.name]
    jv = asset.data.joint_vel.torch[:, asset_cfg.joint_ids].abs()
    return (jv > limit_rad_s).float().mean()
