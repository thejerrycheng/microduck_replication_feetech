"""Get-up from the floor and hold a robust stand, ported from Pollen's official ``microduck_standup_env_cfg.py``
(their deployed ``alpha_stand``) with the start distribution widened to what a real fall produces.

Start families (``reset_fallen_state``): standing, sitting, face-down, face-up (with the official +-90 deg roll
about the long axis), a uniformly random orientation with random joint angles (covers both sides and every
in-between pose), and a BANK of settled poses captured from pushed walking/running falls
(``scripts/capture_fall_states.py``). Every family is labelled per environment so the training log and the
evaluator report success per family, never averaged into one number.

Reward design kept from upstream (weights already in their /4 scale): destination terms (height Gaussian +
sharp + L1, leg pose Gaussian + L1, upright linear + height-gated sharp, multiplicative composite) plus the
one motion term that makes any rising attempt immediately positive (upward CoM velocity below 0.125 m); the
anti-violence terms (arrival damping, head bias, torque rate, accel) start at zero and ramp in late. No rise
speed cap (two upstream runs with one never learned the flips). Pushes during the hold make the stand robust.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .roulade import _smoothstep, _trunk_z, _upright

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

FAMILIES = ("standing", "sitting", "prone", "supine", "random", "bank")
STAND_Z = 0.115


def _state(env: ManagerBasedEnv) -> dict:
    if not hasattr(env, "_standup"):
        env._standup = {
            "family": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "stood_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
            "peak_omega": torch.zeros(env.num_envs, device=env.device),
            "bank": None,
        }
    return env._standup


def _load_bank(env: ManagerBasedEnv, path: str):
    s = _state(env)
    if s["bank"] is None:
        try:
            data = json.load(open(path))
            s["bank"] = {
                "quat": torch.tensor([d["quat_xyzw"] for d in data], dtype=torch.float32, device=env.device),
                "z": torch.tensor([d["z"] for d in data], dtype=torch.float32, device=env.device),
                "joint_pos": torch.tensor([d["joint_pos"] for d in data], dtype=torch.float32, device=env.device),
            }
        except (OSError, ValueError, KeyError):
            s["bank"] = {}
    return s["bank"]


def reset_fallen_state(
    env: ManagerBasedEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    standing_prob: float = 0.40,
    sitting_prob: float = 0.40,
    prone_prob: float = 0.20,
    supine_prob: float = 0.0,
    random_prob: float = 0.0,
    bank_prob: float = 0.0,
    sitting_joint_overrides: dict | None = None,
    sitting_joint_noise_std: float = 0.12,
    tilt_noise_deg: float = 10.0,
    ground_z_range: tuple = (0.05, 0.09),
    standing_z_range: tuple = (0.11, 0.12),
    face_up_roll_max_deg: float = 90.0,
    random_joint_frac: float = 0.6,
    random_z_range: tuple = (0.06, 0.10),
    fall_bank_path: str = "docs/fall_bank.json",
) -> None:
    """Spawn each environment in one of the start families (probabilities are normalised)."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    s = _state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    dev = env.device
    num = len(env_ids)
    probs = torch.tensor([standing_prob, sitting_prob, prone_prob, supine_prob, random_prob, bank_prob], device=dev)
    bank = _load_bank(env, fall_bank_path) if bank_prob > 0 else {}
    if bank_prob > 0 and not bank:
        probs[5] = 0.0  # no bank on disk yet: fold its share into the random family
        probs[4] += bank_prob
    fam = torch.multinomial(probs / probs.sum(), num, replacement=True)

    yaw = (torch.rand(num, device=dev) * 2.0 - 1.0) * math.pi
    tilt = math.radians(tilt_noise_deg)
    roll = (torch.rand(num, device=dev) * 2.0 - 1.0) * tilt
    pitch = (torch.rand(num, device=dev) * 2.0 - 1.0) * tilt
    pitch = torch.where(fam == 2, torch.full_like(pitch, math.pi / 2), pitch)  # face-down: nose to the floor
    pitch = torch.where(fam == 3, torch.full_like(pitch, -math.pi / 2), pitch)  # face-up
    long_roll = (torch.rand(num, device=dev) * 2.0 - 1.0) * math.radians(face_up_roll_max_deg)
    roll = torch.where(fam == 3, long_roll, roll)
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    # random family: uniform random orientation
    u = torch.rand(num, 3, device=dev)
    q_rand = torch.stack([
        torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]), torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
        torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]), torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
    ], dim=1)  # xyzw
    quat = torch.where((fam == 4).unsqueeze(-1), q_rand, quat)

    rng = lambda r: r[0] + torch.rand(num, device=dev) * (r[1] - r[0])  # noqa: E731
    z = torch.where(fam == 0, rng(standing_z_range), rng(ground_z_range))
    z = torch.where(fam == 4, rng(random_z_range), z)

    home = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids].clone()
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
    joint_pos = home.clone()
    sit = home.clone()
    for index, value in (sitting_joint_overrides or {}).items():
        sit[:, int(index)] = float(value)
    sit = sit + torch.randn_like(sit) * sitting_joint_noise_std
    joint_pos = torch.where((fam == 1).unsqueeze(-1), sit, joint_pos)
    centre = 0.5 * (limits[..., 0] + limits[..., 1])
    half = 0.5 * (limits[..., 1] - limits[..., 0]) * random_joint_frac
    rand_joints = centre + (torch.rand_like(home) * 2.0 - 1.0) * half
    joint_pos = torch.where((fam == 4).unsqueeze(-1), rand_joints, joint_pos)
    # prone/supine spawns get a little joint noise too (upstream used HOME exactly; real falls are not at HOME)
    noisy = home + torch.randn_like(home) * 0.15
    joint_pos = torch.where(((fam == 2) | (fam == 3)).unsqueeze(-1), noisy, joint_pos)
    if bank:
        idx = torch.randint(0, bank["quat"].shape[0], (num,), device=dev)
        is_bank = (fam == 5)
        quat = torch.where(is_bank.unsqueeze(-1), bank["quat"][idx], quat)
        z = torch.where(is_bank, bank["z"][idx] + 0.01, z)
        joint_pos = torch.where(is_bank.unsqueeze(-1), bank["joint_pos"][idx], joint_pos)
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

    pos = asset.data.root_link_pos_w.torch[env_ids].clone()
    pos[:, 2] = z + env.scene.env_origins[env_ids, 2]
    asset.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=dev), env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    s["family"][env_ids] = fam
    s["stood_step"][env_ids] = -1
    s["peak_omega"][env_ids] = 0.0


# ── rewards ────────────────────────────────────────────────────────────────────────────────────────────


def _legs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, joint_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    ids = [asset_cfg.joint_ids[i] for i in joint_indices]
    return asset.data.joint_pos.torch[:, ids], asset.data.default_joint_pos.torch[:, ids]


def pose_stand_legs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_indices: list[int], std: float = 0.5) -> torch.Tensor:
    q, home = _legs(env, asset_cfg, joint_indices)
    return torch.exp(-((q - home) / std) ** 2).mean(dim=1)


def pose_stand_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_indices: list[int]) -> torch.Tensor:
    """SELF-NEGATING: -mean|q - HOME| over the leg joints (positive weight)."""
    q, home = _legs(env, asset_cfg, joint_indices)
    return -(q - home).abs().mean(dim=1)


def height_stand_gaussian(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = STAND_Z, std: float = 0.04) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-((_trunk_z(env, asset) - target_height) / std) ** 2)


def height_stand_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = STAND_Z) -> torch.Tensor:
    """SELF-NEGATING: -|z - target| (positive weight). THE anti-lying / anti-sitting term."""
    asset: Articulation = env.scene[asset_cfg.name]
    return -(_trunk_z(env, asset) - target_height).abs()


def com_upward_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_height: float = 0.125) -> torch.Tensor:
    """clamp(vz, 0) while below ``max_height``: pays for the MOTION of rising, uncapped (upstream lesson)."""
    asset: Articulation = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0) * (_trunk_z(env, asset) < max_height).float()


def upright_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _upright(env.scene[asset_cfg.name])


def upright_sharp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.3, z_low: float = 0.06, z_high: float = STAND_Z) -> torch.Tensor:
    """exp(-tilt^2/std^2) gated by height: uprightness only pays near standing height (crouch-basin lesson)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_link_quat_w.torch
    tilt_sq = 2.0 * (q[:, 0].pow(2) + q[:, 1].pow(2))
    gate = _smoothstep(torch.clamp((_trunk_z(env, asset) - z_low) / max(z_high - z_low, 1e-6), 0.0, 1.0))
    return torch.exp(-tilt_sq / (std * std)) * gate


# ── bookkeeping + metrics ───────────────────────────────────────────────────────────────────────────────


def track_standup(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Zero-valued reward term that records time-to-stand and peak angular speed per episode."""
    s = _state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    up = (_upright(asset) > math.cos(math.radians(20.0))) & (_trunk_z(env, asset) > 0.095)
    first = up & (s["stood_step"] < 0)
    s["stood_step"] = torch.where(first, env.episode_length_buf.long(), s["stood_step"])
    s["peak_omega"] = torch.maximum(s["peak_omega"], torch.linalg.norm(asset.data.root_ang_vel_w.torch, dim=-1))
    return torch.zeros(env.num_envs, device=env.device)


def _family_metric(env: ManagerBasedRLEnv, env_ids: torch.Tensor, family: int) -> torch.Tensor:
    s = _state(env)
    asset: Articulation = env.scene["robot"]
    m = s["family"][env_ids] == family
    if m.sum() == 0:
        return torch.tensor(float("nan"))
    up = (_upright(asset)[env_ids] > math.cos(math.radians(20.0))) & (_trunk_z(env, asset)[env_ids] > 0.095)
    return (up & m).float().sum() / m.float().sum()


def stood_prone(env, env_ids):
    return _family_metric(env, env_ids, 2)


def stood_supine(env, env_ids):
    return _family_metric(env, env_ids, 3)


def stood_random(env, env_ids):
    return _family_metric(env, env_ids, 4)


def stood_bank(env, env_ids):
    return _family_metric(env, env_ids, 5)


def stood_sitting(env, env_ids):
    return _family_metric(env, env_ids, 1)


def still_standing(env, env_ids):
    return _family_metric(env, env_ids, 0)


def time_to_stand_s(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Median time to first upright over the ending episodes that started on the ground and did stand."""
    s = _state(env)
    m = (s["family"][env_ids] >= 1) & (s["stood_step"][env_ids] >= 0)
    if m.sum() == 0:
        return torch.tensor(float("nan"))
    return (s["stood_step"][env_ids][m].float() * env.step_dt).median()


def peak_omega(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    s = _state(env)
    return s["peak_omega"][env_ids].median()
