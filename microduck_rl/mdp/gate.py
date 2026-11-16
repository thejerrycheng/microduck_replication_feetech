"""Gate observations for the distillation: quantities the TEACHER GATE needs that the policy must not see.

An extra observation group (``gate``) rides along in the rollout storage (rsl_rl stores every group the env
returns and feeds the networks only the mapped ones), so the actor's 61-wide deploy contract and the critic's
input are untouched and a warm start loads unchanged. ``PpoWithExpertBc`` reads it to decide which frames the
stand-up expert teaches. The symmetry augmentation of the contrib package rejects unknown groups, hence the
wrapper below.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from tensordict import TensorDict

from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_symmetric_states

__all__ = ["trunk_height_obs", "compute_symmetric_states_with_extras", "track_linear_velocity_upright", "track_angular_velocity_upright"]


def trunk_height_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Trunk (root link) height above the environment origin [m], shape (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.nan_to_num(z, nan=0.0).unsqueeze(-1)


@torch.no_grad()
def compute_symmetric_states_with_extras(
    env: ManagerBasedRLEnv | None,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """The contrib mirror augmentation on the ``policy``/``critic`` groups; every other group is duplicated as is
    (a trunk height is its own mirror image)."""
    if obs is None:
        return compute_symmetric_states(env, None, actions)
    known = {k: v for k, v in obs.items() if k in ("policy", "critic")}
    extras = {k: v for k, v in obs.items() if k not in ("policy", "critic")}
    obs_aug, actions_aug = compute_symmetric_states(env, TensorDict(known, batch_size=obs.batch_size), actions)
    for k, v in extras.items():
        obs_aug[k] = torch.cat((v, v), dim=0)
    return obs_aug, actions_aug


def _tilt_gate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, full_deg: float, zero_deg: float) -> torch.Tensor:
    """1 below ``full_deg`` of trunk tilt, 0 above ``zero_deg``, linear in between."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b.torch
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
    return torch.clamp((zero_deg - tilt) / max(zero_deg - full_deg, 1e-6), 0.0, 1.0)


def track_linear_velocity_upright(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    full_deg: float = 15.0, zero_deg: float = 25.0,
) -> torch.Tensor:
    """The contrib linear-velocity tracking reward, paid only when the trunk is upright.

    Measured on RunStand v2 (`docs/standup_runstand_v2_7000_refall.json`): after a get-up the ducks that fall again
    never straighten -- they shuffle forward at 0.3 m/s in a 30-48 deg stoop for 1-2 s and go down -- and the
    ungated tracking term pays for that shuffle. The running gait sits at 3-6 deg and is unaffected."""
    from isaaclab_tasks.contrib.microduck.mdp.rewards import track_linear_velocity

    return track_linear_velocity(env, std, command_name, asset_cfg) * _tilt_gate(env, asset_cfg, full_deg, zero_deg)


def track_angular_velocity_upright(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    full_deg: float = 15.0, zero_deg: float = 25.0,
) -> torch.Tensor:
    """The contrib angular-velocity tracking reward, paid only when the trunk is upright (see above)."""
    from isaaclab_tasks.contrib.microduck.mdp.rewards import track_angular_velocity

    return track_angular_velocity(env, std, command_name, asset_cfg) * _tilt_gate(env, asset_cfg, full_deg, zero_deg)
